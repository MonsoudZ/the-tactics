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
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
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


class JsonStore(InMemoryStore):
    """Persistent store backed by a JSON file. Experience survives restarts."""

    def __init__(self, path: str) -> None:
        super().__init__()
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
