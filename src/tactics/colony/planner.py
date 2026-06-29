"""Planner — turns a Goal into Tasks the colony can swarm.

A planner is called once at the start (to seed the work) and again every round
(to react to Findings ants have posted — the colony writing its own to-do list).
It should be idempotent-ish: only post tasks that don't already exist, since it
runs repeatedly.

The base class is domain-free. Real planners live in playbooks; for quick cases
use :class:`FunctionPlanner` (wrap a function) or :class:`SingleTaskPlanner`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..core.goal import Goal
from .blackboard import Blackboard, Task


class Planner(ABC):
    @abstractmethod
    def plan(self, goal: Goal, board: Blackboard, target) -> list[Task]:  # noqa: ANN001
        """Post any new tasks onto ``board`` and return the ones just added."""


class FunctionPlanner(Planner):
    """Wrap ``fn(goal, board, target) -> list[Task] | None`` as a planner."""

    def __init__(self, fn: Callable[..., list[Task] | None]) -> None:
        self._fn = fn

    def plan(self, goal: Goal, board: Blackboard, target) -> list[Task]:  # noqa: ANN001
        return self._fn(goal, board, target) or []


class SingleTaskPlanner(Planner):
    """Seed exactly one task for the goal, once. Good for simple pursuits."""

    def __init__(self, description: str | None = None, signal: str | None = None) -> None:
        self.description = description
        self.signal = signal
        self._seeded = False

    def plan(self, goal: Goal, board: Blackboard, target) -> list[Task]:  # noqa: ANN001
        if self._seeded:
            return []
        self._seeded = True
        task = board.post_task(
            self.description or f"pursue:{goal.name}", signal=self.signal
        )
        return [task]
