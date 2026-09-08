"""Memory — the framework's accumulated experience.

Memory records every Outcome keyed by (tactic name, context signature) and serves
back the running stats the Policy uses to decide. This is the substrate of
"learns and gets better": nothing is hard-coded about which tactic is best — it
emerges from results, and persists across runs when you use :class:`JsonStore`.

Each signature also stores the raw ``features`` and ``goal`` it came from, so an
:class:`~tactics.core.estimator.SimilarityEstimator` can generalize a tactic's
track record across *similar* situations, not just identical ones.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from collections.abc import Callable
from typing import Any, Protocol


@dataclass
class TacticStats:
    trials: int = 0
    successes: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.trials if self.trials else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


@dataclass
class Entry:
    """One learning row, enriched with the situation it came from."""

    signature: str
    tactic: str
    stats: TacticStats
    features: dict[str, Any]
    goal: str | None


class MemoryStore(Protocol):
    """How the engine reads and writes experience. Swap the backend freely."""

    def record(
        self,
        tactic_name: str,
        signature: str,
        *,
        reward: float,
        success: bool,
        features: dict[str, Any] | None = ...,
        goal: str | None = ...,
    ) -> None: ...

    def stats(self, tactic_name: str, signature: str) -> TacticStats: ...

    def total_trials(self, signature: str) -> int: ...

    def entries(self) -> Iterator[Entry]: ...


class InMemoryStore:
    """Non-persistent store. Great for tests and single-process runs."""

    def __init__(self) -> None:
        self._table: dict[tuple[str, str], TacticStats] = {}  # (signature, tactic) -> stats
        self._meta: dict[str, tuple[dict[str, Any], str | None]] = {}  # signature -> (features, goal)

    def record(
        self,
        tactic_name: str,
        signature: str,
        *,
        reward: float,
        success: bool,
        features: dict[str, Any] | None = None,
        goal: str | None = None,
    ) -> None:
        st = self._table.setdefault((signature, tactic_name), TacticStats())
        st.trials += 1
        st.successes += 1 if success else 0
        st.total_reward += reward
        if features is not None or goal is not None:
            self._meta[signature] = (features or {}, goal)

    def stats(self, tactic_name: str, signature: str) -> TacticStats:
        return self._table.get((signature, tactic_name), TacticStats())

    def total_trials(self, signature: str) -> int:
        return sum(st.trials for (sig, _), st in self._table.items() if sig == signature)

    def entries(self) -> Iterator[Entry]:
        for (sig, name), st in self._table.items():
            features, goal = self._meta.get(sig, ({}, None))
            yield Entry(signature=sig, tactic=name, stats=st, features=features, goal=goal)

    def snapshot(self) -> dict[str, dict[str, dict]]:
        """Human-readable dump: {signature: {tactic_name: stats}}."""
        out: dict[str, dict[str, dict]] = {}
        for (sig, name), st in self._table.items():
            out.setdefault(sig, {})[name] = asdict(st)
        return out


class RecencyStore(InMemoryStore):
    """Recency-weighted store for *non-stationary* worlds (markets shift, code changes).

    Before adding each new observation, the existing stats for that (situation,
    tactic) are scaled by ``decay`` (0..1). So old experience fades and the mean
    tracks what's working *now*, with an effective window of ~ 1/(1-decay)
    observations. ``trials`` becomes a fractional "effective count" — the policy
    and estimators already handle that.
    """

    def __init__(self, decay: float = 0.9) -> None:
        super().__init__()
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay

    def record(
        self,
        tactic_name: str,
        signature: str,
        *,
        reward: float,
        success: bool,
        features: dict[str, Any] | None = None,
        goal: str | None = None,
    ) -> None:
        st = self._table.setdefault((signature, tactic_name), TacticStats())
        st.trials = st.trials * self.decay + 1
        st.successes = st.successes * self.decay + (1 if success else 0)
        st.total_reward = st.total_reward * self.decay + reward
        if features is not None or goal is not None:
            self._meta[signature] = (features or {}, goal)


class TimeDecayStore(InMemoryStore):
    """Recency by *clock time* rather than by observation count.

    :class:`RecencyStore` fades old experience one step per new observation,
    which is the right model when the world moves as you act on it. It is the
    wrong model when the world moves while you are not looking: a store left
    idle over a weekend reloads at full weight and the policy acts on a market,
    or a codebase, that has since changed underneath it.

    Here each cell decays continuously by its own age. ``half_life`` is in
    seconds and has no default on purpose — the right value is a property of the
    domain (an hour for intraday signals, a month for which lint rule matters),
    and quietly guessing it would mis-weight everything the policy ever reads.

    Decay is applied on *read* as well as on write, so a store that has simply
    been sitting there reports faded numbers without needing a write to notice
    the time. ``clock`` is injectable, which is what makes any of this testable.
    """

    def __init__(self, half_life: float, *, clock: Callable[[], float] = time.time) -> None:
        super().__init__()
        if half_life <= 0:
            raise ValueError("half_life must be positive (seconds)")
        self.half_life = half_life
        self.clock = clock
        self._seen: dict[tuple[str, str], float] = {}  # (signature, tactic) -> last touched

    def _weight(self, key: tuple[str, str], now: float) -> float:
        last = self._seen.get(key)
        if last is None:
            return 1.0
        # A clock that jumps backwards (an NTP correction) must never *amplify*
        # old experience, so age is floored at zero.
        age = max(0.0, now - last)
        return math.pow(0.5, age / self.half_life)

    def _faded(self, key: tuple[str, str], st: TacticStats, now: float) -> TacticStats:
        w = self._weight(key, now)
        return TacticStats(
            trials=st.trials * w,
            successes=st.successes * w,
            total_reward=st.total_reward * w,
        )

    def record(
        self,
        tactic_name: str,
        signature: str,
        *,
        reward: float,
        success: bool,
        features: dict[str, Any] | None = None,
        goal: str | None = None,
    ) -> None:
        key = (signature, tactic_name)
        now = self.clock()
        st = self._table.setdefault(key, TacticStats())
        faded = self._faded(key, st, now)
        st.trials = faded.trials + 1
        st.successes = faded.successes + (1 if success else 0)
        st.total_reward = faded.total_reward + reward
        self._seen[key] = now
        if features is not None or goal is not None:
            self._meta[signature] = (features or {}, goal)

    # Reads fade too — otherwise an idle store hands the policy stale confidence.

    def stats(self, tactic_name: str, signature: str) -> TacticStats:
        key = (signature, tactic_name)
        st = self._table.get(key)
        return self._faded(key, st, self.clock()) if st else TacticStats()

    def total_trials(self, signature: str) -> int:
        now = self.clock()
        return sum(self._faded(k, st, now).trials
                   for k, st in self._table.items() if k[0] == signature)

    def entries(self) -> Iterator[Entry]:
        now = self.clock()
        for key, st in self._table.items():
            sig, name = key
            features, goal = self._meta.get(sig, ({}, None))
            yield Entry(signature=sig, tactic=name, stats=self._faded(key, st, now),
                        features=features, goal=goal)


class _JsonBacked:
    """Mixin: persist a store's table to a JSON file, atomically, on every record.

    Kept separate from the store it wraps so persistence and *how stats are
    updated* stay orthogonal — which is what lets a recency-weighted store be
    persistent without either one knowing about the other.
    """

    def __init__(self, path: str, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.path = path
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for entry in raw:
            sig, name = entry["signature"], entry["tactic"]
            self._table[(sig, name)] = TacticStats(
                trials=entry["trials"],
                successes=entry["successes"],
                total_reward=entry["total_reward"],
            )
            self._meta[sig] = (entry.get("features") or {}, entry.get("goal"))
            self._load_extra((sig, name), entry)

    def _flush(self) -> None:
        rows = [
            {
                "signature": sig,
                "tactic": name,
                "trials": st.trials,
                "successes": st.successes,
                "total_reward": st.total_reward,
                "features": self._meta.get(sig, ({}, None))[0],
                "goal": self._meta.get(sig, ({}, None))[1],
                **self._row_extra((sig, name)),
            }
            for (sig, name), st in self._table.items()
        ]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        os.replace(tmp, self.path)  # atomic write — never leaves a half file

    def record(self, tactic_name: str, signature: str, **kw: Any) -> None:
        super().record(tactic_name, signature, **kw)
        self._flush()

    # Hooks so a store with more per-cell state than the stats themselves can
    # round-trip it. Without this a time-decayed store would reload with no idea
    # how old its numbers are — which is the exact bug it exists to prevent.
    def _row_extra(self, key: tuple[str, str]) -> dict[str, Any]:
        return {}

    def _load_extra(self, key: tuple[str, str], entry: dict[str, Any]) -> None:
        return None


class JsonStore(_JsonBacked, InMemoryStore):
    """Persistent store backed by a JSON file. Experience survives restarts."""


class JsonRecencyStore(_JsonBacked, RecencyStore):
    """Persistent *and* recency-weighted — for worlds that are both.

    A market is non-stationary (what worked last quarter may be actively wrong
    now) and long-running (the process restarts, the experience should not).
    Either store alone forces a bad trade: `JsonStore` remembers a regime that
    has ended, `RecencyStore` forgets everything when the process dies.

    ``decay`` is configuration rather than data, so it is not written to the
    file: reopen with a different decay and it governs from that point on.

    Note the decay is per *observation*, not per unit of time — a store left
    idle for a month reloads at full weight. Time-based decay is the right model
    for a market that moved while you were not looking, and is not implemented.
    """




class JsonTimeDecayStore(_JsonBacked, TimeDecayStore):
    """Persistent *and* time-decayed. The combination a market actually wants.

    The timestamps ride along in the file, so reopening a store after a week
    finds week-old experience already faded rather than restored to full
    confidence. That reload is the whole point: a store that only decays while
    the process is alive has no opinion about the time it spent dead.
    """

    def _row_extra(self, key: tuple[str, str]) -> dict[str, Any]:
        return {"last_seen": self._seen.get(key)}

    def _load_extra(self, key: tuple[str, str], entry: dict[str, Any]) -> None:
        seen = entry.get("last_seen")
        if seen is not None:
            self._seen[key] = float(seen)
