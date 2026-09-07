"""Tests for the colony layer: blackboard, planner, critic, parallel swarm."""

from __future__ import annotations

import threading

from tactics import Goal, InMemoryStore, Outcome, Tactic, Target
from tactics.colony import (
    AcceptCritic,
    Blackboard,
    Colony,
    FunctionCritic,
    FunctionPlanner,
    SingleTaskPlanner,
)


# --- fixtures ----------------------------------------------------------------


class Items(Target):
    """Processes a fixed number of items; thread-safe so parallel ants are fine."""

    name = "items"

    def __init__(self, n: int) -> None:
        self.n = n
        self.done = 0
        self._lock = threading.Lock()

    def observe(self) -> dict:
        return {"done": self.done, "n": self.n}

    def process(self) -> None:
        with self._lock:
            self.done += 1


class Process(Tactic):
    def execute(self, ctx) -> Outcome:
        ctx.target.process()
        return Outcome.win(1.0, item=ctx.task.id if ctx.task else None)


def seed_n_tasks(n: int):
    def planner(goal, board, target):
        if board.tasks:
            return []
        return [board.post_task(f"item-{i}", signal="process") for i in range(n)]

    return FunctionPlanner(planner)


def all_done(n: int):
    return Goal(name="process_all", is_satisfied=lambda ctx: ctx.get("done", 0) >= n)


# --- blackboard --------------------------------------------------------------


def test_blackboard_claim_is_priority_ordered_and_exclusive():
    b = Blackboard()
    b.post_task("low", priority=1.0)
    hi = b.post_task("high", priority=5.0)
    first = b.claim_batch(1)
    assert first == [hi]  # highest priority first
    # claimed tasks aren't handed out again
    second = b.claim_batch(5)
    assert hi not in second
    assert len(second) == 1


def test_pheromones_boost_then_evaporate():
    b = Blackboard()
    b.post_task("plain", priority=1.0, signal="x")
    b.post_task("plain2", priority=1.0, signal="y")
    b.deposit("x", 10.0)  # strong trail on x
    picked = b.claim_batch(1)
    assert picked[0].signal == "x"  # pheromone won the tie
    b.deposit("x", 10.0)
    before = b.sense("x")
    b.evaporate(0.5)
    assert b.sense("x") == before * 0.5


def test_single_task_planner_seeds_once():
    b = Blackboard()
    p = SingleTaskPlanner()
    g = Goal(name="g")
    assert len(p.plan(g, b, None)) == 1
    assert p.plan(g, b, None) == []  # idempotent on later rounds


# --- colony end to end -------------------------------------------------------


def test_colony_completes_all_tasks_in_parallel():
    target = Items(8)
    colony = Colony(
        target, [Process()], seed_n_tasks(8),
        memory=InMemoryStore(), critic=AcceptCritic(), max_workers=4, max_rounds=10,
    )
    result = colony.run(all_done(8))
    assert result.satisfied is True
    assert target.done == 8
    assert result.board.counts()["done"] == 8


def test_colony_sequential_mode_is_equivalent():
    target = Items(5)
    colony = Colony(
        target, [Process()], seed_n_tasks(5),
        max_workers=1, max_rounds=10,
    )
    result = colony.run(all_done(5))
    assert result.satisfied is True
    assert target.done == 5


class ClaimSuccess(Tactic):
    """Reports a win without changing the target — to test the critic's gate."""

    def execute(self, ctx) -> Outcome:
        return Outcome.win(1.0)


def test_colony_learns_only_from_verified_outcomes():
    target = Items(4)
    memory = InMemoryStore()
    # A critic that rejects everything -> nothing learned, nothing completed.
    reject = FunctionCritic(lambda outcome, ctx: False)
    colony = Colony(
        target, [ClaimSuccess()], seed_n_tasks(4),
        memory=memory, critic=reject, max_workers=2, max_rounds=3,
    )
    result = colony.run(all_done(4))
    assert result.satisfied is False
    assert list(memory.entries()) == []  # no poisoned learning
    assert result.board.counts()["done"] == 0


def test_colony_records_learning_for_accepted_work():
    target = Items(3)
    memory = InMemoryStore()
    colony = Colony(target, [Process()], seed_n_tasks(3), memory=memory, max_workers=1)
    colony.run(all_done(3))
    entries = list(memory.entries())
    assert entries
    assert all(e.tactic == "Process" for e in entries)
    assert all(e.goal == "process_all" for e in entries)


def test_colony_requires_a_tactic():
    import pytest

    with pytest.raises(ValueError):
        Colony(Items(1), [], seed_n_tasks(1))


# --- per-worker views (Target.session / Target.release) ----------------------
#
# Domain-free: a Target may hand each ant a private view of the domain so
# parallel ants don't collide. The git-worktree fan-out in the agent-sdk playbook
# is one implementation; nothing here knows that.


class Shared(Target):
    """A target that hands out a numbered private view per task."""

    name = "shared"

    def __init__(self) -> None:
        self.handed_out: list[str] = []
        self.released: list[str] = []
        self._lock = threading.Lock()

    def observe(self) -> dict:
        return {"view": "main"}

    def session(self, task=None):  # noqa: ANN001
        with self._lock:
            view = _View(self, f"view-{len(self.handed_out)}")
            self.handed_out.append(view.label)
        return view

    def release(self, session) -> None:  # noqa: ANN001
        with self._lock:
            self.released.append(session.label)


class _View(Target):
    name = "view"

    def __init__(self, parent: Shared, label: str) -> None:
        self.parent, self.label = parent, label

    def observe(self) -> dict:
        return {"view": self.label}


def test_a_target_defaults_to_one_shared_view():
    target = Items(1)
    assert target.session(None) is target
    target.release(target)  # no-op by default


def test_each_ant_acts_on_its_own_view_and_it_is_given_back():
    target = Shared()
    seen: list[str] = []

    class Look(Tactic):
        def execute(self, ctx) -> Outcome:
            seen.append(ctx.data["view"])
            return Outcome.win(1.0)

    colony = Colony(
        target, [Look()],
        FunctionPlanner(lambda g, b, t: [b.post_task(f"t{i}") for i in range(3)] if not b.tasks else []),
        memory=InMemoryStore(), max_workers=3, max_rounds=1,
    )
    colony.run(Goal(name="g"))

    assert sorted(seen) == ["view-0", "view-1", "view-2"]   # never the shared main view
    assert sorted(target.released) == sorted(target.handed_out)  # every view given back


def test_the_critic_verifies_through_the_same_view_the_ant_used():
    # If the Critic re-measured the shared target instead, every parallel result
    # would be judged against work it did not do.
    target = Shared()
    verified: list[str] = []

    class Noop(Tactic):
        def execute(self, ctx) -> Outcome:
            return Outcome.win(1.0)

    def critic(outcome, ctx):
        verified.append(ctx.target.label)
        return True

    colony = Colony(
        target, [Noop()],
        FunctionPlanner(lambda g, b, t: [b.post_task("t0"), b.post_task("t1")] if not b.tasks else []),
        memory=InMemoryStore(), critic=FunctionCritic(critic), max_workers=2, max_rounds=1,
    )
    colony.run(Goal(name="g"))
    assert sorted(verified) == ["view-0", "view-1"]


def test_a_view_is_released_even_when_the_ant_errors():
    class Exploding(Shared):
        def session(self, task=None):  # noqa: ANN001
            view = super().session(task)
            view.observe = lambda: 1 / 0  # blow up after the view is handed out
            return view

    target = Exploding()
    colony = Colony(
        target, [Process()], SingleTaskPlanner(),
        memory=InMemoryStore(), max_workers=1, max_rounds=1,
    )
    colony.run(Goal(name="g"))
    assert target.released == target.handed_out != []  # no leak on the error path
