"""Budget — the governor that keeps autonomy from running away.

"The right win, not the quick win" still needs a ceiling: don't loop forever,
don't spend unbounded money/API calls, don't grind a hopeless task. A Budget caps
three things, any subset:

  * ``max_cost``              — total of every Outcome.cost (dollars, tokens, calls).
  * ``max_seconds``          — wall-clock for the whole pursuit/run.
  * ``max_attempts_per_task``— how many times the colony retries one task before
    giving up on it (so a stuck task can't soak the swarm).

Pass a ``clock`` for deterministic tests. Nothing here is domain-specific — a
tactic just reports what it consumed via ``Outcome.cost``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Budget:
    max_cost: float | None = None
    max_seconds: float | None = None
    max_attempts_per_task: int | None = None
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _spent: float = field(default=0.0, init=False, repr=False)
    _start: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        # Idempotent: a Budget is a fixed allowance for its lifetime, so the clock
        # anchors on first use and cost accumulates across runs. For per-run limits,
        # construct a fresh Budget per run.
        if self._start is None:
            self._start = self.clock()

    def spend(self, amount: float) -> None:
        self._spent += max(0.0, amount)

    @property
    def spent(self) -> float:
        return self._spent

    def elapsed(self) -> float:
        return 0.0 if self._start is None else self.clock() - self._start

    def over_cost(self) -> bool:
        return self.max_cost is not None and self._spent >= self.max_cost

    def over_time(self) -> bool:
        return self.max_seconds is not None and self.elapsed() >= self.max_seconds

    def exhausted(self) -> bool:
        return self.over_cost() or self.over_time()

    def reason(self) -> str | None:
        if self.over_cost():
            return "cost budget exhausted"
        if self.over_time():
            return "time budget exhausted"
        return None

    def attempts_exceeded(self, attempts: int) -> bool:
        return self.max_attempts_per_task is not None and attempts >= self.max_attempts_per_task
