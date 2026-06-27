"""The Outcome — the learning signal.

Every Tactic returns an Outcome. The two fields that drive learning are:

  * ``success`` — did the tactic do its job? (used for reporting and goal checks)
  * ``reward``  — a number the policy maximizes. Bigger is better.

Keep ``reward`` on a stable, comparable scale *within a goal* (e.g. roughly 0..1,
or money earned, or vulns-fixed). The framework never assumes a fixed range; it
only compares rewards of competing tactics in the same context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Outcome:
    """Result of executing a tactic."""

    success: bool
    reward: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def win(cls, reward: float = 1.0, **metrics: Any) -> "Outcome":
        return cls(success=True, reward=reward, metrics=metrics)

    @classmethod
    def loss(cls, reward: float = 0.0, **metrics: Any) -> "Outcome":
        return cls(success=False, reward=reward, metrics=metrics)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        flag = "win " if self.success else "loss"
        note = f" — {self.notes}" if self.notes else ""
        return f"[{flag}] reward={self.reward:.3f}{note}"
