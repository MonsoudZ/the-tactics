"""The Agent — the loop that ties everything together.

    while not goal met and budget remains:
        observe the target          -> Context
        keep the applicable tactics
        let the policy choose one
        execute it                  -> Outcome
        record the outcome in memory (this is the learning)

The Agent owns no domain knowledge. Give it a Target, a set of Tactics, a Policy,
and a Memory, then call ``pursue(goal)``. Run it again and again — each run starts
smarter because Memory carried the lessons forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .context import Context
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
        max_steps: int = 50,
    ) -> None:
        if not tactics:
            raise ValueError("an Agent needs at least one tactic")
        self.target = target
        self.tactics = list(tactics)
        self.policy = policy or UCBPolicy()
        self.memory = memory or InMemoryStore()
        self.max_steps = max_steps

    def _context(self, goal: Goal) -> Context:
        data = self.target.observe()
        features = self.target.features(data)
        return Context(target=self.target, goal=goal, data=data, features=features)

    def pursue(self, goal: Goal) -> RunResult:
        result = RunResult(goal=goal)
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
            self.memory.record(
                tactic.name, sig, reward=outcome.reward, success=outcome.success
            )
            result.steps.append(Step(i, tactic.name, outcome, sig))
        else:
            # loop finished without break — re-check satisfaction one last time
            result.satisfied = goal.satisfied_by(self._context(goal))
        return result
