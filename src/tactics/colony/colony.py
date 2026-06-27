"""Colony — the orchestrator that runs many ants in parallel.

Each round:

  1. the **Planner** seeds or expands the task list (reacting to findings),
  2. the colony **claims** the highest-value open tasks (pheromones bias this),
  3. ants run their tasks **in parallel** — each ant observes the target, the
     policy picks a tactic for the task, the tactic executes,
  4. the **Critic** verifies each outcome; accepted ones are **learned from** and
     **reinforce their pheromone trail**, rejected ones go back in the pool,
  5. pheromones **evaporate** a little so stale trails fade.

Repeat until the goal is satisfied, the work runs dry, or the round budget is hit.

Parallelism uses threads; tactics that touch shared external state must be safe,
or run with ``max_workers=1`` (which executes deterministically, ideal for tests).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..core.context import Context
from ..core.estimator import Estimator, ExactEstimator
from ..core.goal import Goal
from ..core.memory import InMemoryStore, MemoryStore
from ..core.outcome import Outcome
from ..core.policy import Policy, UCBPolicy
from ..core.tactic import Tactic
from .blackboard import Blackboard, Finding, Task
from .critic import AcceptCritic, Critic
from .planner import Planner


@dataclass
class RoundReport:
    index: int
    dispatched: int
    accepted: int
    rejected: int
    counts: dict[str, int]


@dataclass
class ColonyResult:
    goal: Goal
    board: Blackboard
    rounds: int = 0
    satisfied: bool = False
    history: list[RoundReport] = field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return self.board.findings

    def summary(self) -> str:
        status = "satisfied" if self.satisfied else "stopped"
        c = self.board.counts()
        accepted = sum(r.accepted for r in self.history)
        return (
            f"Goal '{self.goal.name}': {status} after {self.rounds} rounds — "
            f"{c.get('done', 0)} tasks done, {c.get('failed', 0)} failed, "
            f"{accepted} verified outcomes learned from"
        )


@dataclass
class _AntResult:
    task: Task
    tactic: Tactic | None
    ctx: Context | None
    outcome: Outcome | None


class Colony:
    def __init__(
        self,
        target,  # noqa: ANN001 - any Target
        tactics: list[Tactic],
        planner: Planner,
        *,
        policy: Policy | None = None,
        memory: MemoryStore | None = None,
        critic: Critic | None = None,
        estimator: Estimator | None = None,
        max_workers: int = 4,
        max_rounds: int = 20,
        evaporation: float = 0.2,
        board: Blackboard | None = None,
    ) -> None:
        if not tactics:
            raise ValueError("a Colony needs at least one tactic")
        self.target = target
        self.tactics = list(tactics)
        self.planner = planner
        self.memory = memory or InMemoryStore()
        self.policy = policy or UCBPolicy(estimator=estimator or ExactEstimator())
        self.critic = critic or AcceptCritic()
        self.max_workers = max(1, max_workers)
        self.max_rounds = max_rounds
        self.evaporation = evaporation
        self.board = board or Blackboard()

    # --- one ant's turn on one task -----------------------------------------

    def _work(self, task: Task, goal: Goal) -> _AntResult:
        data = self.target.observe()
        features = self.target.features(data)
        ctx = Context(target=self.target, goal=goal, data=data, features=features, task=task)
        applicable = [t for t in self.tactics if t.is_applicable(ctx)]
        if not applicable:
            return _AntResult(task, None, ctx, None)
        tactic = self.policy.choose(applicable, ctx, self.memory)
        outcome = tactic.execute(ctx)
        return _AntResult(task, tactic, ctx, outcome)

    def _run_batch(self, batch: list[Task], goal: Goal) -> list[_AntResult]:
        if self.max_workers == 1 or len(batch) == 1:
            return [self._work(t, goal) for t in batch]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda t: self._work(t, goal), batch))

    # --- main loop -----------------------------------------------------------

    def run(self, goal: Goal) -> ColonyResult:
        result = ColonyResult(goal=goal, board=self.board)

        for rnd in range(self.max_rounds):
            self.planner.plan(goal, self.board, self.target)

            if self._goal_met(goal):
                result.satisfied = True
                break

            batch = self.board.claim_batch(self.max_workers)
            if not batch:
                break  # no open work and planner added none — colony is done

            ant_results = self._run_batch(batch, goal)

            accepted = rejected = 0
            for r in ant_results:
                if r.tactic is None or r.outcome is None or r.ctx is None:
                    self.board.fail(r.task)  # nothing could act on it
                    continue
                verdict = self.critic.verify(r.outcome, r.ctx)
                if verdict.accepted:
                    reward = verdict.reward if verdict.reward is not None else r.outcome.reward
                    self.memory.record(
                        r.tactic.name, r.ctx.signature(),
                        reward=reward, success=r.outcome.success,
                        features=r.ctx.features, goal=goal.name,
                    )
                    self.board.complete(r.task, r.outcome)
                    self.board.deposit(r.task.signal, max(reward, 0.0))  # reinforce trail
                    self.board.post_finding(
                        r.tactic.name, "task_done", task=r.task.id, reward=reward
                    )
                    accepted += 1
                else:
                    self.board.reopen(r.task, r.outcome)
                    self.board.post_finding(
                        r.tactic.name, "rejected", task=r.task.id, reason=verdict.reason
                    )
                    rejected += 1

            self.board.evaporate(self.evaporation)
            result.history.append(
                RoundReport(rnd, len(batch), accepted, rejected, self.board.counts())
            )
            result.rounds = rnd + 1

            if self._goal_met(goal):
                result.satisfied = True
                break

        return result

    def _goal_met(self, goal: Goal) -> bool:
        data = self.target.observe()
        features = self.target.features(data)
        ctx = Context(target=self.target, goal=goal, data=data, features=features)
        return goal.satisfied_by(ctx)
