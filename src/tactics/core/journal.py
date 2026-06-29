"""Journal — the audit trail that makes autonomy explainable.

When the system acts on its own, "why did it do that?" must be answerable. The
Journal records every decision as an ordered event: what was observed, which
tactic was chosen and its estimated value, the outcome, the critic's verdict, any
gate commit/hold, and any error. It's thread-safe so the parallel colony can all
write to one.

It's deliberately plain (kind + data). Pass a ``clock`` to stamp wall-clock time;
otherwise events are ordered by a monotonic sequence number, which keeps tests
deterministic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    seq: int
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    t: float | None = None


class Journal:
    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._lock = threading.Lock()
        self._events: list[Event] = []
        self._clock = clock
        self._seq = 0

    def record(self, kind: str, **data: Any) -> Event:
        with self._lock:
            self._seq += 1
            ev = Event(self._seq, kind, data, self._clock() if self._clock else None)
            self._events.append(ev)
            return ev

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def of_kind(self, *kinds: str) -> list[Event]:
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]

    def explain(self, limit: int | None = None) -> str:
        evs = self.events
        if limit is not None:
            evs = evs[-limit:]
        return "\n".join(f"#{e.seq:>3} {e.kind:<16} {e.data}" for e in evs)
