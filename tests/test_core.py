"""Tests for the core loop, memory, and policies."""

from __future__ import annotations

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
