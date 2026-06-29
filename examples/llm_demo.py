"""LLM tactics + LLM critic, running offline.

This uses ScriptedClient so it runs with no API key and no network — it shows the
*wiring*, not a live model. The lead-finder shape: an LLM tactic drafts/judges
outreach and reports a reward; an adversarial LLM critic verifies before the
colony trusts it. Swapping in a real model is a one-line change (see bottom).

Run it:  python examples/llm_demo.py
"""

from __future__ import annotations

from tactics import Goal, InMemoryStore, Target
from tactics.colony import Colony, FunctionPlanner
from tactics.llm import LLMCritic, LLMTactic, ScriptedClient


class Leads(Target):
    """A tiny lead list; each task is one prospect to write outreach for."""

    name = "leads"

    def __init__(self, prospects: list[str]) -> None:
        self.prospects = prospects
        self.handled: set[str] = set()

    def observe(self) -> dict:
        return {"remaining": [p for p in self.prospects if p not in self.handled]}


class DraftOutreach(LLMTactic):
    """Ask the model to assess a prospect and score the outreach it would send."""

    def build_prompt(self, ctx) -> str:
        prospect = ctx.task.payload["prospect"]
        return (
            f"Prospect: {prospect}. Decide whether their site is weak enough to be "
            "a good lead, and draft a one-line outreach hook. Return JSON "
            '{"success": <bool>, "reward": <0..1 quality>, "notes": "<the hook>"}.'
        )

    def interpret(self, data, resp, ctx):  # mark the prospect handled on success
        outcome = super().interpret(data, resp, ctx)
        if outcome.success:
            ctx.target.handled.add(ctx.task.payload["prospect"])
        return outcome


def main() -> None:
    target = Leads(["sloppy-plumbing.example", "great-bakery.example"])

    # Scripted "model": strong lead for the first prospect, weak for the second;
    # the critic then rubber-stamps the first and rejects the thin second.
    tactic_client = ScriptedClient([
        '{"success": true, "reward": 0.9, "notes": "Your booking page 404s on mobile — I can fix it."}',
        '{"success": true, "reward": 0.2, "notes": "Hi, want a new website?"}',
    ])
    critic_client = ScriptedClient([
        '{"accepted": true, "reason": "Specific, credible, high-value hook."}',
        '{"accepted": false, "reason": "Generic spam; would hurt the brand."}',
    ])

    def planner(goal, board, target):
        if board.tasks:
            return []
        return [
            board.post_task(f"reach {p}", payload={"prospect": p}, signal="outreach")
            for p in target.prospects
        ]

    colony = Colony(
        target,
        [DraftOutreach("draft_outreach", tactic_client)],
        FunctionPlanner(planner),
        memory=InMemoryStore(),
        critic=LLMCritic(critic_client),
        max_workers=1,  # deterministic for the demo
        max_rounds=5,
    )
    goal = Goal(name="win_clients", is_satisfied=lambda ctx: not ctx.get("remaining"))
    result = colony.run(goal)

    print(result.summary())
    for f in result.findings:
        print(f"  {f.kind:10s} {f.detail}")
    print("\nThe critic kept the strong hook and rejected the generic one — only "
          "verified outreach was learned from.")
    print("\nTo go live, replace ScriptedClient(...) with:")
    print("    from tactics.llm import ClaudeClient")
    print("    ClaudeClient()           # uses claude-opus-4-8 via ANTHROPIC_API_KEY")


if __name__ == "__main__":
    main()
