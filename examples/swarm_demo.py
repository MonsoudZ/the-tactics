"""Colony demo: a swarm of ants hardening a (simulated) service.

Shows every v0.2 idea at once, domain-free:

  * a Planner posts one task per component that needs work,
  * ants run in parallel, each picking a tactic via the learning policy,
  * a Critic rejects shaky outcomes so only verified fixes count,
  * pheromones concentrate effort on the components that keep paying off,
  * a SimilarityEstimator lets lessons on one component transfer to similar ones.

Run it:  python examples/swarm_demo.py
"""

from __future__ import annotations

import random

from tactics import Goal, InMemoryStore, Outcome, SimilarityEstimator, Tactic, Target, UCBPolicy
from tactics.colony import Colony, FunctionCritic, FunctionPlanner, Verdict


class Service(Target):
    """A fake service with components that each need N fixes to be 'hardened'."""

    name = "service"

    def __init__(self, components: dict[str, int], rng: random.Random) -> None:
        self.remaining = dict(components)  # component -> fixes still needed
        self.rng = rng

    def observe(self) -> dict:
        return {"remaining": dict(self.remaining), "open": sum(self.remaining.values())}

    def features(self, data: dict) -> dict:
        # how risky things still look — lets learning generalize across components
        return {"risk": round(min(1.0, data["open"] / 12), 1)}

    def fix(self, component: str) -> bool:
        if self.remaining.get(component, 0) <= 0:
            return False
        self.remaining[component] -= 1
        return True


class Harden(Tactic):
    """Attempt a fix on the component named by the task. Sometimes flaky."""

    def __init__(self, name: str, reliability: float) -> None:
        super().__init__(name=name)
        self.reliability = reliability

    def execute(self, ctx) -> Outcome:
        component = ctx.task.payload["component"]
        solid = ctx.target.rng.random() < self.reliability
        applied = ctx.target.fix(component) if solid else False
        return Outcome(success=applied, reward=1.0 if applied else 0.0,
                       metrics={"component": component, "solid": solid})


def main() -> None:
    rng = random.Random(11)
    service = Service({"auth": 3, "billing": 3, "api": 3, "uploads": 3}, rng)

    def planner(goal, board, target):
        # one standing task per component still needing work; re-posted as needed
        existing = {t.payload.get("component") for t in board.tasks if t.status != "done"}
        added = []
        for comp, left in target.remaining.items():
            if left > 0 and comp not in existing:
                added.append(board.post_task(
                    f"harden {comp}", payload={"component": comp}, signal=f"comp:{comp}"
                ))
        return added

    # Critic: only trust a fix that was applied solidly (the safety/quality gate).
    critic = FunctionCritic(
        lambda outcome, ctx: Verdict(
            accepted=outcome.success and outcome.metrics.get("solid", False),
            reason="verified fix" if outcome.success else "no-op/ flaky",
        )
    )

    tactics = [Harden("careful_patch", 0.9), Harden("quick_patch", 0.5)]
    goal = Goal(name="hardened", description="No open issues anywhere",
                is_satisfied=lambda ctx: ctx.get("open", 1) == 0)

    colony = Colony(
        service, tactics, FunctionPlanner(planner),
        policy=UCBPolicy(c=1.2, estimator=SimilarityEstimator(sharpness=3.0)),
        memory=InMemoryStore(), critic=critic,
        max_workers=4, max_rounds=40, evaporation=0.25,
    )
    result = colony.run(goal)

    print(result.summary())
    print(f"Components remaining: {service.remaining}")
    print(f"Rounds run: {result.rounds}, findings posted: {len(result.findings)}")
    accepted = sum(1 for f in result.findings if f.kind == "task_done")
    rejected = sum(1 for f in result.findings if f.kind == "rejected")
    print(f"Verified fixes: {accepted}   rejected attempts (caught by critic): {rejected}")


if __name__ == "__main__":
    main()
