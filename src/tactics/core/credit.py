"""CreditAssigner — who gets the reward when payoff is delayed.

A bandit assumes the reward lands the instant you act. Real life doesn't: a buy
sets up a sell three steps later; a refactor makes a later fix possible. **Delayed
credit assignment** spreads a reward back over the moves that earned it.

  * :class:`ImmediateCredit` — each move is scored by its own reward, written the
    moment it happens (online). The default; identical to the original loop.
  * :class:`DiscountedReturn` — buffers the episode, then scores each move by the
    discounted sum of everything that came after it
    (Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + …). A move that triggered a later win gets credit
    for it. This is Monte-Carlo return; learning happens across episodes.

The assigner controls *when* memory is written, so both styles share one engine:
``on_step`` emits records to write immediately, ``on_finish`` emits records to
write at episode end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryStep:
    signature: str
    features: dict[str, Any]
    goal: str | None
    tactic: str
    reward: float
    success: bool


@dataclass
class Record:
    """One thing to write into memory."""

    tactic: str
    signature: str
    features: dict[str, Any]
    goal: str | None
    value: float  # the learning signal (immediate reward or discounted return)
    success: bool


class CreditAssigner(ABC):
    @abstractmethod
    def on_step(self, step: TrajectoryStep) -> list[Record]:
        """Records to write right after a step (online learning). May be empty."""

    @abstractmethod
    def on_finish(self, trajectory: list[TrajectoryStep]) -> list[Record]:
        """Records to write once the episode ends (delayed learning). May be empty."""


class ImmediateCredit(CreditAssigner):
    def on_step(self, step: TrajectoryStep) -> list[Record]:
        return [
            Record(step.tactic, step.signature, step.features, step.goal,
                   value=step.reward, success=step.success)
        ]

    def on_finish(self, trajectory: list[TrajectoryStep]) -> list[Record]:
        return []


@dataclass
class DiscountedReturn(CreditAssigner):
    """Spread reward backward with discount factor ``gamma`` (0=myopic, →1=patient)."""

    gamma: float = 0.9
    _: Any = field(default=None, repr=False)  # keep dataclass happy with the methods

    def on_step(self, step: TrajectoryStep) -> list[Record]:
        return []  # wait for the whole episode

    def on_finish(self, trajectory: list[TrajectoryStep]) -> list[Record]:
        records: list[Record] = []
        g = 0.0
        for step in reversed(trajectory):
            g = step.reward + self.gamma * g
            records.append(
                Record(step.tactic, step.signature, step.features, step.goal,
                       value=g, success=step.success)
            )
        records.reverse()
        return records
