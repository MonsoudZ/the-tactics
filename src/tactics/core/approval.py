"""ApprovalGate — propose, then commit only if allowed.

This is how "no breaking down, no leaks" becomes a mechanism instead of a hope.
A tactic that wants to do something irreversible (deploy, send an email, place a
trade, write to a repo) does its analysis, then hands the *commit step* to the
gate instead of running it directly:

    proposal = Proposal("deploy v12", commit=lambda: do_deploy(plan),
                        reversible=False, risk="high")
    result = ctx.gate.submit(proposal, ctx)
    return Outcome.win(1.0) if result.committed else Outcome.loss(notes="held")

The gate decides whether ``commit`` actually runs. Swap the gate to change the
safety posture without touching the tactic:

  * :class:`AutoApprove` — run everything (simulations, fully trusted runs).
  * :class:`DryRun`      — never run; just record what *would* have happened.
  * :class:`CallbackGate`— ask a function (your human-approval hook).
  * :class:`PolicyGate`  — auto-run reversible/low-risk; defer the rest to a callback.

Every decision is written to ``ctx.journal`` when one is present, so the audit
trail shows exactly what was committed, held, or skipped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Proposal:
    action: str
    commit: Callable[[], Any] | None = field(default=None, repr=False)
    reversible: bool = False
    risk: str = "low"  # "low" | "medium" | "high"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    approved: bool
    committed: bool
    result: Any = None
    reason: str = ""


class ApprovalGate(ABC):
    @abstractmethod
    def decide(self, proposal: Proposal, ctx) -> bool:  # noqa: ANN001
        """Return True to allow the proposal's commit to run."""

    def submit(self, proposal: Proposal, ctx=None) -> GateResult:  # noqa: ANN001
        journal = getattr(ctx, "journal", None)
        approved = self.decide(proposal, ctx)
        if not approved:
            if journal:
                journal.record("gate.hold", action=proposal.action, risk=proposal.risk)
            return GateResult(approved=False, committed=False, reason="held for approval")
        result = proposal.commit() if proposal.commit else None
        if journal:
            journal.record("gate.commit", action=proposal.action, risk=proposal.risk)
        return GateResult(approved=True, committed=True, result=result, reason="approved")


class AutoApprove(ApprovalGate):
    def decide(self, proposal: Proposal, ctx) -> bool:  # noqa: ANN001
        return True


class DryRun(ApprovalGate):
    """Never commit — record intentions so you can review what the colony wanted."""

    def decide(self, proposal: Proposal, ctx) -> bool:  # noqa: ANN001
        return False


class CallbackGate(ApprovalGate):
    """Defer every decision to ``fn(proposal, ctx) -> bool`` (e.g. a human prompt)."""

    def __init__(self, fn: Callable[[Proposal, Any], bool]) -> None:
        self._fn = fn

    def decide(self, proposal: Proposal, ctx) -> bool:  # noqa: ANN001
        return bool(self._fn(proposal, ctx))


class PolicyGate(ApprovalGate):
    """Auto-approve reversible/low-risk actions; escalate the rest.

    ``allow_risk`` is the set of risk levels safe to auto-commit when reversible.
    Anything else goes to ``escalate(proposal, ctx) -> bool`` (default: deny).
    """

    def __init__(
        self,
        escalate: Callable[[Proposal, Any], bool] | None = None,
        allow_risk: tuple[str, ...] = ("low",),
    ) -> None:
        self._escalate = escalate
        self._allow_risk = set(allow_risk)

    def decide(self, proposal: Proposal, ctx) -> bool:  # noqa: ANN001
        if proposal.reversible and proposal.risk in self._allow_risk:
            return True
        return False if self._escalate is None else bool(self._escalate(proposal, ctx))
