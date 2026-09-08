"""Tests for the core loop, memory, and policies."""

from __future__ import annotations

import json
import pathlib
import random

import pytest

from tactics import (
    Agent,
    Context,
    EpsilonGreedyPolicy,
    Goal,
    InMemoryStore,
    Outcome,
    Tactic,
    Target,
    UCBPolicy,
)
from tactics.core.tactic import FunctionTactic


# --- tiny fixtures -----------------------------------------------------------


class Counter(Target):
    """A target whose state is a single integer that tactics increment."""

    name = "counter"

    def __init__(self, start: int = 0) -> None:
        self.value = start

    def observe(self) -> dict:
        return {"value": self.value}

    def features(self, data: dict) -> dict:
        return {"phase": "low" if data["value"] < 3 else "high"}

    def bump(self, by: int) -> None:
        self.value += by


class Bump(Tactic):
    def __init__(self, name: str, by: int, reward: float) -> None:
        super().__init__(name=name)
        self.by = by
        self.reward = reward

    def execute(self, ctx) -> Outcome:
        ctx.target.bump(self.by)
        return Outcome(success=True, reward=self.reward)


# --- Outcome -----------------------------------------------------------------


def test_outcome_helpers():
    assert Outcome.win(2.0, vulns=3).success is True
    assert Outcome.win(2.0).reward == 2.0
    assert Outcome.loss().success is False
    assert Outcome.win(metrics_check=1).metrics == {"metrics_check": 1}


# --- Memory ------------------------------------------------------------------


def test_inmemory_store_accumulates():
    m = InMemoryStore()
    m.record("t", "sig", reward=1.0, success=True)
    m.record("t", "sig", reward=0.0, success=False)
    st = m.stats("t", "sig")
    assert st.trials == 2
    assert st.successes == 1
    assert st.mean_reward == 0.5
    assert st.success_rate == 0.5
    assert m.total_trials("sig") == 2


def test_empty_stats_are_safe():
    st = InMemoryStore().stats("missing", "sig")
    assert st.trials == 0
    assert st.mean_reward == 0.0
    assert st.success_rate == 0.0


def test_json_store_persists(tmp_path):
    from tactics import JsonStore

    path = str(tmp_path / "mem.json")
    a = JsonStore(path)
    a.record("t", "sig", reward=3.0, success=True)

    b = JsonStore(path)  # reload from disk
    assert b.stats("t", "sig").trials == 1
    assert b.stats("t", "sig").total_reward == 3.0


# --- Context -----------------------------------------------------------------


def test_signature_is_stable_and_feature_scoped():
    g = Goal(name="g")
    t = Counter()
    c1 = Context(target=t, goal=g, features={"a": 1, "b": 2})
    c2 = Context(target=t, goal=g, features={"b": 2, "a": 1})  # order differs
    assert c1.signature() == c2.signature()
    c3 = Context(target=t, goal=g, features={"a": 9})
    assert c1.signature() != c3.signature()


# --- Policy ------------------------------------------------------------------


def test_ucb_tries_each_tactic_before_repeating():
    m = InMemoryStore()
    tactics = [Bump("a", 1, 1.0), Bump("b", 1, 1.0), Bump("c", 1, 1.0)]
    ctx = Context(target=Counter(), goal=Goal(name="g"))
    policy = UCBPolicy()
    picked = set()
    for _ in range(3):
        t = policy.choose(tactics, ctx, m)
        picked.add(t.name)
        m.record(t.name, ctx.signature(), reward=1.0, success=True)
    assert picked == {"a", "b", "c"}


def test_ucb_converges_to_best():
    m = InMemoryStore()
    good = Bump("good", 1, 1.0)
    bad = Bump("bad", 1, 0.0)
    ctx = Context(target=Counter(), goal=Goal(name="g"))
    policy = UCBPolicy(c=0.5)
    counts = {"good": 0, "bad": 0}
    for _ in range(200):
        t = policy.choose([good, bad], ctx, m)
        counts[t.name] += 1
        reward = 1.0 if t.name == "good" else 0.0
        m.record(t.name, ctx.signature(), reward=reward, success=bool(reward))
    assert counts["good"] > counts["bad"] * 5  # strongly prefers the winner


def test_epsilon_greedy_is_deterministic_with_seed():
    rng = random.Random(0)
    policy = EpsilonGreedyPolicy(epsilon=0.2, rng=rng)
    m = InMemoryStore()
    tactics = [Bump("a", 1, 1.0), Bump("b", 1, 0.0)]
    ctx = Context(target=Counter(), goal=Goal(name="g"))
    # untried first
    for t in tactics:
        m.record(t.name, ctx.signature(), reward=1.0 if t.name == "a" else 0.0, success=True)
    seq = [policy.choose(tactics, ctx, m).name for _ in range(10)]
    # reproducible given the seed
    rng2 = random.Random(0)
    policy2 = EpsilonGreedyPolicy(epsilon=0.2, rng=rng2)
    seq2 = [policy2.choose(tactics, ctx, m).name for _ in range(10)]
    assert seq == seq2


# --- Engine ------------------------------------------------------------------


def test_agent_stops_when_goal_satisfied():
    target = Counter()
    goal = Goal(name="reach_5", is_satisfied=lambda ctx: ctx.get("value", 0) >= 5)
    agent = Agent(target, [Bump("inc", 1, 1.0)], max_steps=100)
    result = agent.pursue(goal)
    assert result.satisfied is True
    assert target.value == 5
    assert len(result.steps) == 5


def test_agent_respects_step_budget():
    target = Counter()
    goal = Goal(name="never", is_satisfied=lambda ctx: False)
    agent = Agent(target, [Bump("inc", 1, 1.0)], max_steps=4)
    result = agent.pursue(goal)
    assert result.satisfied is False
    assert len(result.steps) == 4


def test_agent_skips_inapplicable_tactics():
    target = Counter()
    calls = {"n": 0}

    def only_high(ctx):
        return ctx.get("value", 0) >= 100

    def do(ctx):
        calls["n"] += 1
        return Outcome.win()

    blocked = FunctionTactic("blocked", do, applies=only_high)
    inc = Bump("inc", 1, 1.0)
    goal = Goal(name="reach_3", is_satisfied=lambda ctx: ctx.get("value", 0) >= 3)
    agent = Agent(target, [blocked, inc], max_steps=50)
    result = agent.pursue(goal)
    assert result.satisfied is True
    assert calls["n"] == 0  # the inapplicable tactic was never executed


def test_agent_requires_a_tactic():
    with pytest.raises(ValueError):
        Agent(Counter(), [])


def test_run_result_summary_reports_progress():
    target = Counter()
    goal = Goal(name="reach_3", is_satisfied=lambda ctx: ctx.get("value", 0) >= 3)
    result = Agent(target, [Bump("inc", 1, 1.0)]).pursue(goal)
    assert result.wins == 3
    assert result.total_reward == 3.0
    assert "satisfied" in result.summary()


def test_learning_carries_across_runs_via_shared_memory():
    """Two separate pursuits share one memory; the winner's lead grows."""
    memory = InMemoryStore()
    good, bad = Bump("good", 1, 1.0), Bump("bad", 1, 0.0)
    goal = Goal(name="open")  # never satisfied -> runs to budget

    Agent(Counter(), [good, bad], memory=memory, max_steps=40).pursue(goal)
    sig_runs = [k for k in memory.snapshot()]
    assert sig_runs  # something was learned
    # the good tactic should have a higher mean reward in every learned bucket
    for sig, table in memory.snapshot().items():
        if "good" in table and "bad" in table:
            assert table["good"]["total_reward"] >= table["bad"]["total_reward"]


# --- WithoutReplacement: spreading a parallel round across tactics ------------


def _tactics(*names):
    from tactics import FunctionTactic, Outcome

    return [FunctionTactic(n, lambda ctx: Outcome.win(1.0)) for n in names]


def _policy_ctx():
    from tactics import Context, Goal, Target

    class Nothing(Target):
        name = "nothing"

        def observe(self) -> dict:
            return {}

    return Context(target=Nothing(), goal=Goal(name="g"))


def test_an_unwrapped_policy_gives_every_ant_the_same_tactic():
    # The bug WithoutReplacement exists to fix, pinned so it can't come back
    # silently: one memory snapshot, one conclusion, N times over.
    from tactics import InMemoryStore, UCBPolicy

    policy, ctx, memory = UCBPolicy(), _policy_ctx(), InMemoryStore()
    tactics = _tactics("a", "b", "c")
    for t in tactics:  # give every tactic a record so UCB stops exploring
        memory.record(t.name, ctx.signature(), reward=1.0 if t.name == "b" else 0.0, success=True)
    assert len({policy.choose(tactics, ctx, memory).name for _ in range(3)}) == 1


def test_a_parallel_round_spreads_across_tactics():
    from tactics import InMemoryStore, UCBPolicy, WithoutReplacement

    policy = WithoutReplacement(UCBPolicy())
    ctx, memory = _policy_ctx(), InMemoryStore()
    tactics = _tactics("a", "b", "c", "d")
    policy.begin_round()
    picks = [policy.choose(tactics, ctx, memory).name for _ in range(3)]
    assert len(set(picks)) == 3


def test_the_next_round_starts_the_slate_clean():
    from tactics import InMemoryStore, UCBPolicy, WithoutReplacement

    policy = WithoutReplacement(UCBPolicy())
    ctx, memory = _policy_ctx(), InMemoryStore()
    tactics = _tactics("a", "b")
    policy.begin_round()
    first = [policy.choose(tactics, ctx, memory).name for _ in range(2)]
    policy.begin_round()
    second = [policy.choose(tactics, ctx, memory).name for _ in range(2)]
    assert sorted(first) == sorted(second) == ["a", "b"]


def test_more_ants_than_tactics_share_out_evenly():
    # Six ants over three tactics should come out two apiece, not four piled on
    # the winner and one each on the rest.
    from collections import Counter

    from tactics import InMemoryStore, UCBPolicy, WithoutReplacement

    policy = WithoutReplacement(UCBPolicy())
    ctx, memory = _policy_ctx(), InMemoryStore()
    tactics = _tactics("a", "b", "c")
    policy.begin_round()
    counts = Counter(policy.choose(tactics, ctx, memory).name for _ in range(6))
    assert set(counts.values()) == {2}


def test_the_wrapped_policy_still_picks_the_best_of_what_is_left():
    # Spreading must not become "ignore what we know" — the first ant still gets
    # the winner; the others get the best of the rest.
    from tactics import InMemoryStore, UCBPolicy, WithoutReplacement

    ctx, memory = _policy_ctx(), InMemoryStore()
    tactics = _tactics("poor", "great", "ok")
    for name, reward in (("poor", 0.0), ("great", 1.0), ("ok", 0.5)):
        for _ in range(4):
            memory.record(name, ctx.signature(), reward=reward, success=reward > 0.4)

    policy = WithoutReplacement(UCBPolicy(c=0.0))  # pure exploitation, to read the order
    policy.begin_round()
    assert [policy.choose(tactics, ctx, memory).name for _ in range(3)] == ["great", "ok", "poor"]


def test_concurrent_ants_never_collide():
    import threading

    from tactics import InMemoryStore, UCBPolicy, WithoutReplacement

    policy = WithoutReplacement(UCBPolicy())
    ctx, memory = _policy_ctx(), InMemoryStore()
    tactics = _tactics("a", "b", "c", "d", "e", "f", "g", "h")
    policy.begin_round()

    picks, lock, ready = [], threading.Lock(), threading.Barrier(8)

    def pick():
        ready.wait()  # maximize the overlap on the read-choose-mark window
        chosen = policy.choose(tactics, ctx, memory)
        with lock:
            picks.append(chosen.name)

    threads = [threading.Thread(target=pick) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(picks)) == 8


def test_begin_round_is_a_no_op_on_a_plain_policy():
    from tactics import EpsilonGreedyPolicy, UCBPolicy

    UCBPolicy().begin_round()
    EpsilonGreedyPolicy().begin_round()


# --- persistent recency: the two halves markets need at once ------------------


def test_a_persistent_recency_store_decays_and_survives_restart(tmp_path):
    from tactics import JsonRecencyStore

    path = str(tmp_path / "mem.json")
    store = JsonRecencyStore(path, decay=0.5)
    for _ in range(3):
        store.record("t", "sig", reward=1.0, success=True)

    reopened = JsonRecencyStore(path, decay=0.5)          # a fresh process
    assert reopened.stats("t", "sig").trials == store.stats("t", "sig").trials
    assert reopened.stats("t", "sig").mean_reward == store.stats("t", "sig").mean_reward


def test_old_experience_fades_where_a_plain_json_store_would_hoard_it(tmp_path):
    from tactics import JsonRecencyStore, JsonStore

    fading = JsonRecencyStore(str(tmp_path / "r.json"), decay=0.5)
    hoarding = JsonStore(str(tmp_path / "j.json"))
    for store in (fading, hoarding):
        for _ in range(10):                      # the old regime: this worked
            store.record("t", "sig", reward=1.0, success=True)
        for _ in range(3):                       # the regime turned
            store.record("t", "sig", reward=0.0, success=False)

    # The recency store has largely moved on; the plain one is still anchored.
    assert fading.stats("t", "sig").mean_reward < 0.2
    assert hoarding.stats("t", "sig").mean_reward > 0.7


def test_reopening_with_a_different_decay_governs_from_then_on(tmp_path):
    from tactics import JsonRecencyStore

    path = str(tmp_path / "m.json")
    JsonRecencyStore(path, decay=0.9).record("t", "sig", reward=1.0, success=True)
    reopened = JsonRecencyStore(path, decay=0.1)   # decay is config, not data
    assert reopened.decay == 0.1
    assert reopened.stats("t", "sig").total_reward == 1.0   # what was written stands


# --- decay by the clock, not by the number of observations --------------------


class _Clock:
    """A hand-cranked clock, because a test that sleeps is a test nobody runs."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds
        return self


def test_a_half_life_is_required_because_the_right_one_is_domain_specific():
    from tactics import TimeDecayStore

    with __import__("pytest").raises(ValueError, match="positive"):
        TimeDecayStore(0)
    with __import__("pytest").raises(TypeError):
        TimeDecayStore()          # no default: guessing it mis-weights everything


def test_experience_halves_over_one_half_life():
    from tactics import TimeDecayStore

    clock = _Clock()
    store = TimeDecayStore(half_life=60.0, clock=clock)
    store.record("t", "sig", reward=1.0, success=True)
    assert store.stats("t", "sig").trials == 1.0

    clock.advance(60)
    assert store.stats("t", "sig").trials == pytest.approx(0.5)
    clock.advance(60)
    assert store.stats("t", "sig").trials == pytest.approx(0.25)


def test_an_idle_store_fades_without_being_written_to():
    # The failure this exists to prevent: a store nobody touched over a weekend
    # handing the policy full confidence in a world that has since moved.
    from tactics import RecencyStore, TimeDecayStore

    clock = _Clock()
    timed = TimeDecayStore(half_life=3600.0, clock=clock)
    counted = RecencyStore(decay=0.5)
    for store in (timed, counted):
        store.record("t", "sig", reward=1.0, success=True)

    clock.advance(86_400)                       # a day passes; nothing happens
    assert timed.stats("t", "sig").trials < 0.001
    assert counted.stats("t", "sig").trials == 1.0   # count-based decay noticed nothing


def test_the_mean_still_reads_correctly_while_the_weight_fades():
    from tactics import TimeDecayStore

    clock = _Clock()
    store = TimeDecayStore(half_life=60.0, clock=clock)
    for _ in range(4):
        store.record("t", "sig", reward=1.0, success=True)
    clock.advance(300)
    st = store.stats("t", "sig")
    assert st.trials < 0.2                      # barely any weight left
    assert st.mean_reward == pytest.approx(1.0) # but it still says what it learned


def test_a_recent_result_outweighs_an_old_one():
    from tactics import TimeDecayStore

    clock = _Clock()
    store = TimeDecayStore(half_life=60.0, clock=clock)
    store.record("t", "sig", reward=1.0, success=True)     # the old regime
    clock.advance(600)
    store.record("t", "sig", reward=0.0, success=False)    # the regime turned
    assert store.stats("t", "sig").mean_reward < 0.02


def test_a_clock_that_jumps_backwards_never_amplifies_old_experience():
    from tactics import TimeDecayStore

    clock = _Clock()
    store = TimeDecayStore(half_life=60.0, clock=clock)
    store.record("t", "sig", reward=1.0, success=True)
    clock.advance(-3600)                        # an NTP correction, say
    assert store.stats("t", "sig").trials == pytest.approx(1.0)


def test_totals_and_entries_fade_too_or_the_policy_reads_stale_confidence():
    from tactics import TimeDecayStore

    clock = _Clock()
    store = TimeDecayStore(half_life=60.0, clock=clock)
    store.record("a", "sig", reward=1.0, success=True, features={"x": 1}, goal="g")
    store.record("b", "sig", reward=1.0, success=True, features={"x": 1}, goal="g")
    clock.advance(60)
    assert store.total_trials("sig") == pytest.approx(1.0)      # 2 × 0.5
    assert all(e.stats.trials == pytest.approx(0.5) for e in store.entries())


def test_the_time_a_process_spent_dead_still_counts(tmp_path):
    # The reload case is the whole point: decay that only runs while the process
    # is alive has no opinion about the week it was not.
    from tactics import JsonTimeDecayStore

    path = str(tmp_path / "m.json")
    clock = _Clock()
    store = JsonTimeDecayStore(path, half_life=60.0, clock=clock)
    store.record("t", "sig", reward=1.0, success=True)

    later = _Clock(clock.t + 180)               # three half-lives later
    reopened = JsonTimeDecayStore(path, half_life=60.0, clock=later)
    assert reopened.stats("t", "sig").trials == pytest.approx(0.125)


def test_a_persisted_time_store_round_trips_its_timestamps(tmp_path):
    from tactics import JsonTimeDecayStore

    path = str(tmp_path / "m.json")
    clock = _Clock()
    JsonTimeDecayStore(path, half_life=60.0, clock=clock).record(
        "t", "sig", reward=1.0, success=True)
    saved = json.loads(pathlib.Path(path).read_text())
    assert saved[0]["last_seen"] == clock.t

    same_moment = JsonTimeDecayStore(path, half_life=60.0, clock=_Clock(clock.t))
    assert same_moment.stats("t", "sig").trials == pytest.approx(1.0)


def test_the_other_stores_are_unaffected_by_the_new_row_field(tmp_path):
    from tactics import JsonRecencyStore, JsonStore

    for cls, kw in ((JsonStore, {}), (JsonRecencyStore, {"decay": 0.9})):
        path = str(tmp_path / f"{cls.__name__}.json")
        cls(path, **kw).record("t", "sig", reward=1.0, success=True)
        assert "last_seen" not in json.loads(pathlib.Path(path).read_text())[0]
        assert cls(path, **kw).stats("t", "sig").trials > 0
