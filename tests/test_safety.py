"""Tests for the trust layer: failure isolation, budgets, approval gate,
journal, and recency-weighted memory."""

from __future__ import annotations

import pytest

from tactics import (
    Agent,
    AutoApprove,
    Budget,
    CallbackGate,
    DryRun,
    Goal,
    InMemoryStore,
    Journal,
    Outcome,
    PolicyGate,
    Proposal,
    RecencyStore,
    Tactic,
    Target,
)
from tactics.colony import Colony, FunctionPlanner


# --- fixtures ----------------------------------------------------------------


class Counter(Target):
    name = "counter"

    def __init__(self) -> None:
        self.value = 0

    def observe(self) -> dict:
        return {"value": self.value}

    def bump(self) -> None:
        self.value += 1


class Boom(Tactic):
    """A tactic that always raises — to prove the loop survives it."""

    def execute(self, ctx) -> Outcome:
        raise RuntimeError("kaboom")


class Bump(Tactic):
    def __init__(self, name="bump", reward=1.0, cost=0.0) -> None:
        super().__init__(name=name)
        self.reward = reward
        self.cost = cost

    def execute(self, ctx) -> Outcome:
        ctx.target.bump()
        return Outcome(success=True, reward=self.reward, cost=self.cost)


# --- failure isolation -------------------------------------------------------


def test_agent_survives_a_raising_tactic():
    agent = Agent(Counter(), [Boom()], max_steps=3, memory=InMemoryStore())
    result = agent.pursue(Goal(name="g"))  # never satisfied
    assert len(result.steps) == 3  # didn't crash; logged losses instead
    assert all(s.outcome.success is False for s in result.steps)
    assert result.journal.of_kind("error")


def test_colony_survives_a_raising_tactic():
    target = Counter()
    planner = FunctionPlanner(
        lambda g, b, t: [b.post_task("x")] if not b.tasks else []
    )
    colony = Colony(
        target, [Boom()], planner,
        budget=Budget(max_attempts_per_task=2), max_workers=2, max_rounds=4,
    )
    result = colony.run(Goal(name="g"))
    assert result.satisfied is False  # nothing got done, but no crash
    assert result.journal.of_kind("error")


# --- budget ------------------------------------------------------------------


def test_budget_stops_agent_on_cost():
    agent = Agent(
        Counter(), [Bump(cost=2.0)],
        budget=Budget(max_cost=5.0), max_steps=100, memory=InMemoryStore(),
    )
    result = agent.pursue(Goal(name="never"))  # open-ended
    # spends 2 per step; stops once cumulative >= 5 -> after 3 steps (cost 6)
    assert result.total_cost >= 5.0
    assert "cost" in result.stop_reason


def test_budget_deadline_uses_injected_clock():
    ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Budget(max_seconds=3.0, clock=lambda: next(ticks))
    agent = Agent(Counter(), [Bump()], budget=b, max_steps=100, memory=InMemoryStore())
    result = agent.pursue(Goal(name="never"))
    assert "time" in result.stop_reason


def test_budget_attempts_cap_fails_a_stuck_task():
    target = Counter()
    planner = FunctionPlanner(lambda g, b, t: [b.post_task("stuck")] if not b.tasks else [])
    # critic rejects forever; attempt cap should fail the task rather than loop
    from tactics.colony import FunctionCritic

    colony = Colony(
        target, [Bump()], planner,
        critic=FunctionCritic(lambda o, c: False),
        budget=Budget(max_attempts_per_task=3), max_workers=1, max_rounds=20,
    )
    result = colony.run(Goal(name="g"))
    assert result.board.counts()["failed"] == 1
    assert result.journal.of_kind("task.gaveup")


def test_budget_remaining_cost():
    b = Budget(max_cost=5.0)
    assert b.remaining_cost() == 5.0
    b.spend(2.0)
    assert b.remaining_cost() == 3.0
    b.spend(10.0)
    assert b.remaining_cost() == -7.0
    assert Budget().remaining_cost() is None


# --- approval gate -----------------------------------------------------------


def test_auto_approve_commits():
    committed = []
    gate = AutoApprove()
    res = gate.submit(Proposal("send", commit=lambda: committed.append(1)))
    assert res.committed is True
    assert committed == [1]


def test_dry_run_never_commits():
    committed = []
    gate = DryRun()
    res = gate.submit(Proposal("deploy", commit=lambda: committed.append(1)))
    assert res.committed is False
    assert committed == []


def test_policy_gate_auto_approves_reversible_low_risk_only():
    asked = []
    gate = PolicyGate(escalate=lambda p, c: asked.append(p.action) or False)
    safe = gate.submit(Proposal("toggle flag", commit=lambda: "ok", reversible=True, risk="low"))
    risky = gate.submit(Proposal("wire money", commit=lambda: "sent", reversible=False, risk="high"))
    assert safe.committed is True
    assert risky.committed is False
    assert asked == ["wire money"]  # the risky one was escalated, the safe one wasn't


def test_gate_writes_to_journal():
    class _Ctx:
        journal = Journal()

    ctx = _Ctx()
    CallbackGate(lambda p, c: True).submit(Proposal("act", commit=lambda: None), ctx)
    CallbackGate(lambda p, c: False).submit(Proposal("act2", commit=lambda: None), ctx)
    kinds = [e.kind for e in ctx.journal.events]
    assert "gate.commit" in kinds
    assert "gate.hold" in kinds


def test_tactic_uses_gate_to_hold_irreversible_action():
    """A tactic proposes; with DryRun nothing fires and it reports held."""
    sent = []

    class SendEmail(Tactic):
        def execute(self, ctx) -> Outcome:
            res = ctx.gate.submit(
                Proposal("send outreach", commit=lambda: sent.append("mail"), reversible=False),
                ctx,
            )
            return Outcome.win(1.0) if res.committed else Outcome.loss(notes="held")

    agent = Agent(Counter(), [SendEmail()], gate=DryRun(), max_steps=1, memory=InMemoryStore())
    result = agent.pursue(Goal(name="g"))
    assert sent == []  # the irreversible action was gated off
    assert result.steps[0].outcome.success is False


# --- journal -----------------------------------------------------------------


def test_journal_orders_and_filters_events():
    j = Journal()
    j.record("a", x=1)
    j.record("b", y=2)
    j.record("a", x=3)
    assert [e.seq for e in j.events] == [1, 2, 3]
    assert len(j.of_kind("a")) == 2
    assert "a" in j.explain()


# --- recency-weighted memory -------------------------------------------------


def test_recency_store_tracks_a_regime_shift():
    plain = InMemoryStore()
    recent = RecencyStore(decay=0.6)
    sig = "s"
    # Phase 1: tactic looks great.
    for _ in range(10):
        plain.record("t", sig, reward=1.0, success=True)
        recent.record("t", sig, reward=1.0, success=True)
    # Phase 2: the world changes; it now fails.
    for _ in range(10):
        plain.record("t", sig, reward=0.0, success=False)
        recent.record("t", sig, reward=0.0, success=False)

    # Plain memory still thinks it's ~50/50 (all history weighted equally)...
    assert plain.stats("t", sig).mean_reward == pytest.approx(0.5)
    # ...recency memory has largely forgotten the good old days.
    assert recent.stats("t", sig).mean_reward < 0.2


def test_recency_rejects_bad_decay():
    with pytest.raises(ValueError):
        RecencyStore(decay=0.0)
    with pytest.raises(ValueError):
        RecencyStore(decay=1.5)
