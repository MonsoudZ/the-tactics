"""Regression tests for issues found in the full system audit."""

from __future__ import annotations

import threading

from tactics import Budget, Goal, Outcome, Tactic, Target
from tactics.colony import Colony, FunctionCritic, FunctionPlanner


class Counter(Target):
    name = "counter"

    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def observe(self) -> dict:
        return {"value": self.value}

    def bump(self) -> None:
        with self._lock:
            self.value += 1


class Bump(Tactic):
    def execute(self, ctx) -> Outcome:
        ctx.target.bump()
        return Outcome.win(1.0)


def _one_task_planner():
    return FunctionPlanner(lambda g, b, t: [b.post_task("x")] if not b.tasks else [])


# --- Finding 1: planner / critic failures must not crash the colony ----------


def test_colony_isolates_a_raising_critic():
    def boom(outcome, ctx):
        raise RuntimeError("critic exploded")

    colony = Colony(
        Counter(), [Bump()], _one_task_planner(),
        critic=FunctionCritic(boom), budget=Budget(max_attempts_per_task=2),
        max_workers=1, max_rounds=4,
    )
    result = colony.run(Goal(name="g"))  # must not raise
    assert result.board.counts()["done"] == 0  # nothing trusted
    errors = [e for e in result.journal.of_kind("error") if e.data.get("stage") == "critic"]
    assert errors


def test_colony_isolates_a_raising_planner():
    class BoomPlanner(FunctionPlanner):
        def __init__(self):
            super().__init__(lambda g, b, t: None)

        def plan(self, goal, board, target):
            raise RuntimeError("planner exploded")

    colony = Colony(Counter(), [Bump()], BoomPlanner(), max_workers=1, max_rounds=3)
    result = colony.run(Goal(name="g"))  # must not raise
    assert result.satisfied is False
    errors = [e for e in result.journal.of_kind("error") if e.data.get("stage") == "planner"]
    assert errors


# --- Finding 2: Budget.start() is idempotent (lifetime allowance) ------------


def test_budget_start_is_idempotent_on_the_clock():
    ticks = iter([10.0, 50.0])
    b = Budget(max_seconds=100.0, clock=lambda: next(ticks))
    b.start()          # anchors at 10.0
    b.start()          # idempotent — no re-anchor, consumes no tick
    assert b.elapsed() == 40.0  # 50.0 - 10.0, proving the anchor stayed at 10.0


def test_budget_cost_accumulates_across_starts():
    b = Budget(max_cost=10.0)
    b.start()
    b.spend(4.0)
    b.start()  # idempotent — does not reset spend
    b.spend(4.0)
    assert b.spent == 8.0
    assert not b.exhausted()
    b.spend(3.0)
    assert b.exhausted()
