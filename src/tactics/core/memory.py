"""Memory — the framework's accumulated experience.

Memory records every Outcome keyed by (tactic name, context signature) and serves
back the running stats the Policy uses to decide. This is the substrate of
"learns and gets better": nothing is hard-coded about which tactic is best — it
emerges from results, and persists across runs when you use :class:`JsonStore`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Protocol


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


class MemoryStore(Protocol):
    """How the engine reads and writes experience. Swap the backend freely."""

    def record(self, tactic_name: str, signature: str, *, reward: float, success: bool) -> None: ...

    def stats(self, tactic_name: str, signature: str) -> TacticStats: ...

    def total_trials(self, signature: str) -> int: ...


class InMemoryStore:
    """Non-persistent store. Great for tests and single-process runs."""

    def __init__(self) -> None:
        # key: (signature, tactic_name) -> TacticStats
        self._table: dict[tuple[str, str], TacticStats] = {}

    def record(self, tactic_name: str, signature: str, *, reward: float, success: bool) -> None:
        st = self._table.setdefault((signature, tactic_name), TacticStats())
        st.trials += 1
        st.successes += 1 if success else 0
        st.total_reward += reward

    def stats(self, tactic_name: str, signature: str) -> TacticStats:
        return self._table.get((signature, tactic_name), TacticStats())

    def total_trials(self, signature: str) -> int:
        return sum(st.trials for (sig, _), st in self._table.items() if sig == signature)

    def snapshot(self) -> dict[str, dict[str, dict]]:
        """Human-readable dump: {signature: {tactic_name: stats}}."""
        out: dict[str, dict[str, dict]] = {}
        for (sig, name), st in self._table.items():
            out.setdefault(sig, {})[name] = asdict(st)
        return out


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
            self._table[(entry["signature"], entry["tactic"])] = TacticStats(
                trials=entry["trials"],
                successes=entry["successes"],
                total_reward=entry["total_reward"],
            )

    def _flush(self) -> None:
        rows = [
            {
                "signature": sig,
                "tactic": name,
                "trials": st.trials,
                "successes": st.successes,
                "total_reward": st.total_reward,
            }
            for (sig, name), st in self._table.items()
        ]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        os.replace(tmp, self.path)  # atomic write — never leaves a half file

    def record(self, tactic_name: str, signature: str, *, reward: float, success: bool) -> None:
        super().record(tactic_name, signature, reward=reward, success=success)
        self._flush()
