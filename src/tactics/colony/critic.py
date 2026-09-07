"""Critic — verify before the colony trusts an outcome.

Two jobs, both essential once the colony acts autonomously:

  1. **Learning integrity.** A tactic that reports a win it didn't earn would
     poison memory and the whole swarm would chase a mirage. The Critic
     independently judges each Outcome; only accepted ones are learned from.
  2. **Safety gate.** For irreversible or outward-facing actions (deploy, send,
     sell), the Critic is where a verification — or a human approval hook — goes,
     so nothing risky is committed on a tactic's say-so alone.

The base is domain-free. :class:`AcceptCritic` trusts everything (fine for pure
simulations/tests). Real playbooks supply a critic that re-checks the work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ..core.context import Context
from ..core.outcome import Outcome


@dataclass
class Verdict:
    accepted: bool
    reason: str = ""
    # Optionally override the reward the colony learns from (e.g. discount an
    # unverified-but-plausible result). Defaults to the outcome's own reward.
    reward: float | None = None
    # Whether this outcome *finishes* the task, which is a different question
    # from whether it can be trusted. "Run this check" is done once the check
    # has run, pass or fail; "make the suite pass" is not done until it passes.
    # ``None`` keeps the historical default (an accepted outcome completes its
    # task); a critic that knows better returns False to send it back for retry.
    done: bool | None = None


class Critic(ABC):
    @abstractmethod
    def verify(self, outcome: Outcome, ctx: Context) -> Verdict: ...


class AcceptCritic(Critic):
    """Trust every outcome. Only safe in simulation/tests."""

    def verify(self, outcome: Outcome, ctx: Context) -> Verdict:
        return Verdict(accepted=True, reason="accept-all")


class FunctionCritic(Critic):
    """Wrap ``fn(outcome, ctx) -> bool | Verdict`` as a critic."""

    def __init__(self, fn: Callable[[Outcome, Context], "bool | Verdict"]) -> None:
        self._fn = fn

    def verify(self, outcome: Outcome, ctx: Context) -> Verdict:
        res = self._fn(outcome, ctx)
        if isinstance(res, Verdict):
            return res
        return Verdict(accepted=bool(res), reason="fn")
