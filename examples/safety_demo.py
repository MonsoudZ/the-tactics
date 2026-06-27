"""Safety demo: budgets, the approval gate, and the audit journal.

Shows the Tier-1 trust layer doing its job, domain-free:

  * a tactic proposes an *irreversible* action (a "deploy") through ``ctx.gate``,
  * a DryRun gate holds it — nothing fires — while a PolicyGate would auto-approve
    only reversible/low-risk work and escalate the rest,
  * a Budget caps spend so the colony can't run away,
  * the Journal records every decision so you can answer "why did it do that?".

Run it:  python examples/safety_demo.py
"""

from __future__ import annotations

from tactics import Budget, DryRun, Goal, Outcome, PolicyGate, Proposal, Tactic, Target
from tactics.colony import Colony, FunctionPlanner


class Release(Target):
    name = "release"

    def __init__(self) -> None:
        self.deployed = False
        self.checks_run = 0

    def observe(self) -> dict:
        return {"deployed": self.deployed, "checks_run": self.checks_run}

    def run_checks(self) -> None:
        self.checks_run += 1

    def do_deploy(self) -> None:
        self.deployed = True


class RunChecks(Tactic):
    """Reversible, low-risk: safe to auto-approve."""

    def execute(self, ctx) -> Outcome:
        res = ctx.gate.submit(
            Proposal("run pre-release checks", commit=ctx.target.run_checks,
                     reversible=True, risk="low"),
            ctx,
        )
        return Outcome.win(0.5, cost=1.0) if res.committed else Outcome.loss(notes="held")


class Deploy(Tactic):
    """Irreversible, high-risk: must pass the gate before it can fire."""

    def execute(self, ctx) -> Outcome:
        res = ctx.gate.submit(
            Proposal("deploy to production", commit=ctx.target.do_deploy,
                     reversible=False, risk="high"),
            ctx,
        )
        if res.committed:
            return Outcome.win(1.0, cost=1.0)
        return Outcome.loss(notes="deploy held for approval", cost=0.0)


def planner(goal, board, target):
    if board.tasks:
        return []
    return [
        board.post_task("checks", payload={"kind": "checks"}, priority=2.0, signal="checks"),
        board.post_task("deploy", payload={"kind": "deploy"}, priority=1.0, signal="deploy"),
    ]


def run(gate, label: str) -> None:
    target = Release()
    tactics = [
        # each tactic only claims its matching task
        type("RC", (RunChecks,), {"is_applicable": lambda self, ctx: ctx.task.payload["kind"] == "checks"})(),
        type("DP", (Deploy,), {"is_applicable": lambda self, ctx: ctx.task.payload["kind"] == "deploy"})(),
    ]
    colony = Colony(
        target, tactics, FunctionPlanner(planner),
        gate=gate, budget=Budget(max_cost=10.0, max_attempts_per_task=3),
        max_workers=1, max_rounds=10,
    )
    goal = Goal(name="released", is_satisfied=lambda ctx: ctx.get("deployed"))
    result = colony.run(goal)

    print(f"\n=== {label} ===")
    print(result.summary())
    print(f"checks_run={target.checks_run}  deployed={target.deployed}")
    gate_events = result.journal.of_kind("gate.commit", "gate.hold")
    for e in gate_events:
        verb = "COMMIT" if e.kind == "gate.commit" else "HOLD  "
        print(f"  {verb} {e.data['action']} (risk={e.data['risk']})")


def main() -> None:
    # DryRun: nothing irreversible ever fires — you review intentions first.
    run(DryRun(), "DryRun gate (review-only)")
    # PolicyGate: auto-approve reversible/low-risk; the high-risk deploy is denied
    # here because no human escalation hook is wired in.
    run(PolicyGate(allow_risk=("low",)), "PolicyGate (no human hook)")
    # PolicyGate with an approval hook that says yes to the deploy.
    run(PolicyGate(escalate=lambda p, c: True), "PolicyGate (human approves deploy)")


if __name__ == "__main__":
    main()
