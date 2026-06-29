"""Colony — the orchestrator that runs many ants in parallel.

Each round:

  1. the **Planner** seeds or expands the task list (reacting to findings),
  2. the colony **claims** the highest-value open tasks (pheromones bias this),
  3. ants run their tasks **in parallel** — each ant observes the target, the
     policy picks a tactic for the task, the tactic executes (a tactic that
     *raises* becomes a loss, never a crash that takes down the swarm),
  4. the **Critic** verifies each outcome; accepted ones are **learned from** and
     **reinforce their pheromone trail**; rejected/errored ones go back in the
     pool — until a task exceeds the Budget's attempt cap, when it's failed,
  5. the **Budget** is charged each outcome's cost; the run stops if it's spent,
  6. pheromones **evaporate** so stale trails fade.

Every decision is written to the **Journal** for an auditable "why". Parallelism
uses threads; run with ``max_workers=1`` for deterministic, ordered execution.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..core.approval import ApprovalGate, AutoApprove
from ..core.budget import Budget
from ..core.context import Context
from ..core.estimator import Estimator, ExactEstimator
from ..core.goal import Goal
from ..core.journal import Journal
from ..core.memory import InMemoryStore, MemoryStore
from ..core.outcome import Outcome
from ..core.policy import Policy, UCBPolicy
from ..core.tactic import Tactic
from .blackboard import Blackboard, Finding, Task
from .critic import AcceptCritic, Critic, Verdict


@dataclass
class RoundReport:
    index: int
    dispatched: int
    accepted: int
    rejected: int
    errored: int
    counts: dict[str, int]


@dataclass
class ColonyResult:
    goal: Goal
    board: Blackboard
    rounds: int = 0
    satisfied: bool = False
    stop_reason: str = ""
    journal: Journal | None = None
    history: list[RoundReport] = field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return self.board.findings

    def summary(self) -> str:
        status = "satisfied" if self.satisfied else f"stopped ({self.stop_reason})"
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
    errored: bool = False


class Colony:
    def __init__(
        self,
        target,  # noqa: ANN001 - any Target
        tactics: list[Tactic],
        planner,  # noqa: ANN001 - any Planner
        *,
        policy: Policy | None = None,
        memory: MemoryStore | None = None,
        critic: Critic | None = None,
        estimator: Estimator | None = None,
        budget: Budget | None = None,
        gate: ApprovalGate | None = None,
        journal: Journal | None = None,
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
        self.budget = budget
        self.gate = gate or AutoApprove()
        self.journal = journal or Journal()
        self.max_workers = max(1, max_workers)
        self.max_rounds = max_rounds
        self.evaporation = evaporation
        self.board = board or Blackboard()

    # --- one ant's turn on one task -----------------------------------------

    def _work(self, task: Task, goal: Goal) -> _AntResult:
        ctx = None
        try:
            data = self.target.observe()
            features = self.target.features(data)
            ctx = Context(
                target=self.target, goal=goal, data=data, features=features,
                task=task, gate=self.gate, journal=self.journal,
            )
            applicable = [t for t in self.tactics if t.is_applicable(ctx)]
            if not applicable:
                return _AntResult(task, None, ctx, None)
            tactic = self.policy.choose(applicable, ctx, self.memory)
            outcome = tactic.execute(ctx)
            return _AntResult(task, tactic, ctx, outcome)
        except Exception as exc:  # failure isolation across the parallel batch
            self.journal.record("error", task=task.id, error=repr(exc))
            return _AntResult(task, None, ctx, Outcome.failed(exc), errored=True)

    def _run_batch(self, batch: list[Task], goal: Goal) -> list[_AntResult]:
        if self.max_workers == 1 or len(batch) == 1:
            return [self._work(t, goal) for t in batch]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda t: self._work(t, goal), batch))

    # --- retry / give-up -----------------------------------------------------

    def _verify(self, outcome: Outcome, ctx: Context) -> Verdict:
        try:
            return self.critic.verify(outcome, ctx)
        except Exception as exc:  # fail-closed: a broken critic rejects, never rubber-stamps
            self.journal.record("error", stage="critic", error=repr(exc))
            return Verdict(accepted=False, reason=f"critic error: {exc!r}")

    def _retry_or_fail(self, task: Task, outcome: Outcome | None) -> None:
        if self.budget and self.budget.attempts_exceeded(task.attempts):
            self.board.fail(task, outcome)
            self.journal.record("task.gaveup", task=task.id, attempts=task.attempts)
        else:
            self.board.reopen(task, outcome)

    # --- main loop -----------------------------------------------------------

    def run(self, goal: Goal) -> ColonyResult:
        result = ColonyResult(goal=goal, board=self.board, journal=self.journal)
        if self.budget:
            self.budget.start()

        for rnd in range(self.max_rounds):
            if self.budget and self.budget.exhausted():
                result.stop_reason = self.budget.reason() or "budget exhausted"
                break

            try:
                self.planner.plan(goal, self.board, self.target)
            except Exception as exc:  # a broken planner degrades, doesn't crash the swarm
                self.journal.record("error", stage="planner", error=repr(exc))

            if self._goal_met(goal):
                result.satisfied = True
                result.stop_reason = "goal satisfied"
                break

            batch = self.board.claim_batch(self.max_workers)
            if not batch:
                result.stop_reason = "no open work"
                break

            self.journal.record("round", index=rnd, dispatched=len(batch))
            ant_results = self._run_batch(batch, goal)

            accepted = rejected = errored = 0
            for r in ant_results:
                if r.outcome is not None and self.budget:
                    self.budget.spend(r.outcome.cost)

                if r.errored:
                    self._retry_or_fail(r.task, r.outcome)
                    errored += 1
                    continue
                if r.tactic is None or r.ctx is None:
                    self.board.fail(r.task)  # nothing could act on it — permanent
                    continue

                verdict = self._verify(r.outcome, r.ctx)
                self.journal.record(
                    "verify", task=r.task.id, tactic=r.tactic.name,
                    accepted=verdict.accepted, reason=verdict.reason,
                )
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
                    self.board.post_finding(
                        r.tactic.name, "rejected", task=r.task.id, reason=verdict.reason
                    )
                    self._retry_or_fail(r.task, r.outcome)
                    rejected += 1

            self.board.evaporate(self.evaporation)
            result.history.append(
                RoundReport(rnd, len(batch), accepted, rejected, errored, self.board.counts())
            )
            result.rounds = rnd + 1

            if self._goal_met(goal):
                result.satisfied = True
                result.stop_reason = "goal satisfied"
                break
        else:
            result.stop_reason = result.stop_reason or "round budget"

        if not result.stop_reason:
            result.stop_reason = "round budget"
        return result

    def _goal_met(self, goal: Goal) -> bool:
        data = self.target.observe()
        features = self.target.features(data)
        ctx = Context(target=self.target, goal=goal, data=data, features=features)
        return goal.satisfied_by(ctx)
