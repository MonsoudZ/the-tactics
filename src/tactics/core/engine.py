"""The Agent — the loop that ties everything together.

    while not goal met and budget remains:
        observe the target          -> Context
        keep the applicable tactics
        let the policy choose one
        execute it                  -> Outcome
        buffer the step; the CreditAssigner decides when it becomes learning

The Agent owns no domain knowledge. Give it a Target, a set of Tactics, a Policy,
a CreditAssigner, and a Memory, then call ``pursue(goal)``. With the default
:class:`ImmediateCredit` it learns online (every step). Swap in
:class:`DiscountedReturn` and it learns from delayed payoff across episodes —
each ``pursue`` call is one episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .context import Context
from .credit import CreditAssigner, ImmediateCredit, Record, TrajectoryStep
from .goal import Goal
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

    @property
    def total_reward(self) -> float:
        return sum(s.outcome.reward for s in self.steps)

    @property
    def wins(self) -> int:
        return sum(1 for s in self.steps if s.outcome.success)

    def summary(self) -> str:
        status = "satisfied" if self.satisfied else "stopped (budget)"
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
        max_steps: int = 50,
    ) -> None:
        if not tactics:
            raise ValueError("an Agent needs at least one tactic")
        self.target = target
        self.tactics = list(tactics)
        self.policy = policy or UCBPolicy()
        self.memory = memory or InMemoryStore()
        self.credit = credit or ImmediateCredit()
        self.max_steps = max_steps

    def _context(self, goal: Goal) -> Context:
        data = self.target.observe()
        features = self.target.features(data)
        return Context(target=self.target, goal=goal, data=data, features=features)

    def _write(self, records: list[Record]) -> None:
        for r in records:
            self.memory.record(
                r.tactic, r.signature, reward=r.value, success=r.success,
                features=r.features, goal=r.goal,
            )

    def pursue(self, goal: Goal) -> RunResult:
        result = RunResult(goal=goal)
        trajectory: list[TrajectoryStep] = []
        for i in range(self.max_steps):
            ctx = self._context(goal)
            if goal.satisfied_by(ctx):
                result.satisfied = True
                break
            applicable = [t for t in self.tactics if t.is_applicable(ctx)]
            if not applicable:
                break  # nothing can act on this situation
            tactic = self.policy.choose(applicable, ctx, self.memory)
            outcome = tactic.execute(ctx)
            sig = ctx.signature()
            tstep = TrajectoryStep(
                signature=sig, features=ctx.features, goal=goal.name,
                tactic=tactic.name, reward=outcome.reward, success=outcome.success,
            )
            trajectory.append(tstep)
            self._write(self.credit.on_step(tstep))  # online learning (if any)
            result.steps.append(Step(i, tactic.name, outcome, sig))
        else:
            result.satisfied = goal.satisfied_by(self._context(goal))
        self._write(self.credit.on_finish(trajectory))  # delayed learning (if any)
        return result
