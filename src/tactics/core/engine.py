"""The Agent — the loop that ties everything together.

    while not goal met and budget remains:
        observe the target          -> Context
        keep the applicable tactics
        let the policy choose one
        execute it (errors become a loss, never a crash)
        spend its cost against the Budget
        buffer the step; the CreditAssigner decides when it becomes learning

The Agent owns no domain knowledge. Give it a Target, Tactics, a Policy, a
CreditAssigner, a Memory — and optionally a Budget (a governor), a gate (the
approval mechanism injected into each Context), and a Journal (the audit trail).
With the defaults it learns online and never stops early; add a Budget and it
respects ceilings on cost and time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .approval import ApprovalGate, AutoApprove
from .budget import Budget
from .context import Context
from .credit import CreditAssigner, ImmediateCredit, Record, TrajectoryStep
from .goal import Goal
from .journal import Journal
from .memory import InMemoryStore, MemoryStore
from .outcome import Outcome
from .policy import Policy, UCBPolicy
from .tactic import Tactic


@dataclass
class Step:
    index: int
    tactic: str
    outcome: Outcome
    signature: str


@dataclass
class RunResult:
    goal: Goal
    steps: list[Step] = field(default_factory=list)
    satisfied: bool = False
    stop_reason: str = ""
    journal: Journal | None = None

    @property
    def total_reward(self) -> float:
        return sum(s.outcome.reward for s in self.steps)

    @property
    def total_cost(self) -> float:
        return sum(s.outcome.cost for s in self.steps)

    @property
    def wins(self) -> int:
        return sum(1 for s in self.steps if s.outcome.success)

    def summary(self) -> str:
        status = "satisfied" if self.satisfied else f"stopped ({self.stop_reason})"
        return (
            f"Goal '{self.goal.name}': {status} in {len(self.steps)} steps, "
            f"{self.wins} wins, total reward {self.total_reward:.3f}"
        )


class Agent:
    """Runs the goal-seeking loop over a Target using its Tactics."""

    def __init__(
        self,
        target,  # noqa: ANN001 - any Target
        tactics: Sequence[Tactic],
        policy: Policy | None = None,
        memory: MemoryStore | None = None,
        credit: CreditAssigner | None = None,
        budget: Budget | None = None,
        gate: ApprovalGate | None = None,
        journal: Journal | None = None,
        max_steps: int = 50,
    ) -> None:
        if not tactics:
            raise ValueError("an Agent needs at least one tactic")
        self.target = target
        self.tactics = list(tactics)
        self.policy = policy or UCBPolicy()
        self.memory = memory or InMemoryStore()
        self.credit = credit or ImmediateCredit()
        self.budget = budget
        self.gate = gate or AutoApprove()
        self.journal = journal or Journal()
        self.max_steps = max_steps

    def _context(self, goal: Goal) -> Context:
        data = self.target.observe()
        features = self.target.features(data)
        return Context(
            target=self.target, goal=goal, data=data, features=features,
            gate=self.gate, journal=self.journal,
        )

    def _write(self, records: list[Record]) -> None:
        for r in records:
            self.memory.record(
                r.tactic, r.signature, reward=r.value, success=r.success,
                features=r.features, goal=r.goal,
            )

    def _safe_execute(self, tactic: Tactic, ctx: Context) -> Outcome:
        try:
            return tactic.execute(ctx)
        except Exception as exc:  # failure isolation: one bad tactic can't crash the run
            self.journal.record("error", tactic=tactic.name, error=repr(exc))
            return Outcome.failed(exc)

    def pursue(self, goal: Goal) -> RunResult:
        result = RunResult(goal=goal, journal=self.journal)
        trajectory: list[TrajectoryStep] = []
        if self.budget:
            self.budget.start()

        for i in range(self.max_steps):
            if self.budget and self.budget.exhausted():
                result.stop_reason = self.budget.reason() or "budget exhausted"
                break

            ctx = self._context(goal)
            if goal.satisfied_by(ctx):
                result.satisfied = True
                result.stop_reason = "goal satisfied"
                break

            applicable = [t for t in self.tactics if t.is_applicable(ctx)]
            if not applicable:
                result.stop_reason = "no applicable tactic"
                break

            tactic = self.policy.choose(applicable, ctx, self.memory)
            outcome = self._safe_execute(tactic, ctx)
            if self.budget:
                self.budget.spend(outcome.cost)

            sig = ctx.signature()
            tstep = TrajectoryStep(
                signature=sig, features=ctx.features, goal=goal.name,
                tactic=tactic.name, reward=outcome.reward, success=outcome.success,
            )
            trajectory.append(tstep)
            self._write(self.credit.on_step(tstep))  # online learning (if any)
            self.journal.record(
                "step", index=i, tactic=tactic.name,
                reward=outcome.reward, cost=outcome.cost, success=outcome.success,
            )
            result.steps.append(Step(i, tactic.name, outcome, sig))
        else:
            result.satisfied = goal.satisfied_by(self._context(goal))
            result.stop_reason = "goal satisfied" if result.satisfied else "step budget"

        self._write(self.credit.on_finish(trajectory))  # delayed learning (if any)
        return result
