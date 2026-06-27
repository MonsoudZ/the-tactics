"""The Blackboard — the colony's shared environment.

Three things live here, and every ant can read and write them safely from any
thread:

  * **Tasks** — units of work. Ants claim the highest-value open task, do it, and
    mark it done or failed.
  * **Findings** — what ants learned ("auth controller looks risky"). A Planner
    can read findings and post follow-up tasks — this is how the colony grows its
    own to-do list.
  * **Pheromones** — named signals with a strength that **evaporates** each round
    and is **reinforced** when work pays off. A task tagged with a strong signal
    gets claimed sooner. That's stigmergy: the trail, not a manager, directs the
    swarm.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.outcome import Outcome

_OPEN = "open"
_CLAIMED = "claimed"
_DONE = "done"
_FAILED = "failed"


@dataclass
class Task:
    id: str
    description: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    signal: str | None = None  # pheromone tag that boosts this task's urgency
    status: str = _OPEN
    attempts: int = 0
    result: Outcome | None = None


@dataclass
class Finding:
    source: str  # which tactic/ant reported it
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


class Blackboard:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self.tasks: list[Task] = []
        self.findings: list[Finding] = []
        self._pheromones: dict[str, float] = {}

    # --- tasks ---------------------------------------------------------------

    def post_task(
        self,
        description: str,
        *,
        payload: dict[str, Any] | None = None,
        priority: float = 1.0,
        signal: str | None = None,
        task_id: str | None = None,
    ) -> Task:
        with self._lock:
            task = Task(
                id=task_id or f"t{next(self._ids)}",
                description=description,
                payload=payload or {},
                priority=priority,
                signal=signal,
            )
            self.tasks.append(task)
            return task

    def _effective_priority(self, task: Task) -> float:
        boost = 1.0 + self.sense(task.signal) if task.signal else 1.0
        return task.priority * boost

    def claim_batch(self, limit: int, where: Callable[[Task], bool] | None = None) -> list[Task]:
        """Claim up to ``limit`` open tasks, highest effective priority first."""
        with self._lock:
            ready = [
                t for t in self.tasks
                if t.status == _OPEN and (where is None or where(t))
            ]
            ready.sort(key=self._effective_priority, reverse=True)
            chosen = ready[:limit]
            for t in chosen:
                t.status = _CLAIMED
                t.attempts += 1
            return chosen

    def complete(self, task: Task, outcome: Outcome) -> None:
        with self._lock:
            task.status = _DONE
            task.result = outcome

    def reopen(self, task: Task, outcome: Outcome | None = None, *, decay: float = 0.5) -> None:
        """Return a task to the pool (e.g. critic rejected it), with lower priority."""
        with self._lock:
            task.status = _OPEN
            task.result = outcome
            task.priority *= decay

    def fail(self, task: Task, outcome: Outcome | None = None) -> None:
        with self._lock:
            task.status = _FAILED
            task.result = outcome

    def open_tasks(self) -> list[Task]:
        with self._lock:
            return [t for t in self.tasks if t.status == _OPEN]

    def has_open_work(self) -> bool:
        return bool(self.open_tasks())

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {_OPEN: 0, _CLAIMED: 0, _DONE: 0, _FAILED: 0}
            for t in self.tasks:
                out[t.status] = out.get(t.status, 0) + 1
            return out

    # --- findings ------------------------------------------------------------

    def post_finding(self, source: str, kind: str, **detail: Any) -> Finding:
        with self._lock:
            f = Finding(source=source, kind=kind, detail=detail)
            self.findings.append(f)
            return f

    # --- pheromones (stigmergy) ---------------------------------------------

    def deposit(self, signal: str | None, amount: float = 1.0) -> None:
        if not signal:
            return
        with self._lock:
            self._pheromones[signal] = self._pheromones.get(signal, 0.0) + amount

    def sense(self, signal: str | None) -> float:
        if not signal:
            return 0.0
        with self._lock:
            return self._pheromones.get(signal, 0.0)

    def evaporate(self, rate: float = 0.2) -> None:
        """Decay every trail by ``rate`` so stale paths fade and fresh wins lead."""
        with self._lock:
            for k in list(self._pheromones):
                self._pheromones[k] *= (1.0 - rate)
                if self._pheromones[k] < 1e-6:
                    del self._pheromones[k]
