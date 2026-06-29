"""End-to-end demo: the framework *learning* which outreach tactic wins.

This mirrors your lead-finder idea. We have several cold-email tactics, each with
a hidden true reply-rate the agent does not know. The agent tries them, watches
the outcomes (replies), and the policy steadily shifts toward the best performer —
no rule told it which one; it learned from results.

Run it:  python examples/lead_finder_demo.py
"""

from __future__ import annotations

import random
from collections import Counter

from tactics import Agent, Goal, InMemoryStore, Outcome, Tactic, Target, UCBPolicy


class LeadOutreach(Target):
    """Stand-in for a real lead list. Each step "sends" to the next prospect."""

    name = "lead_outreach"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.sent = 0
        self.replies = 0

    def observe(self) -> dict:
        return {"sent": self.sent, "replies": self.replies}

    # A real Target would expose send_email(); here tactics simulate a send.
    def simulate_send(self, true_reply_rate: float) -> bool:
        self.sent += 1
        replied = self.rng.random() < true_reply_rate
        self.replies += 1 if replied else 0
        return replied


class EmailTactic(Tactic):
    """A cold-email template with a hidden true reply-rate the agent can't see."""

    def __init__(self, name: str, true_reply_rate: float) -> None:
        super().__init__(name=name)
        self._rate = true_reply_rate

    def execute(self, ctx) -> Outcome:
        replied = ctx.target.simulate_send(self._rate)
        return Outcome(success=replied, reward=1.0 if replied else 0.0,
                       notes="reply" if replied else "no reply")


def main() -> None:
    rng = random.Random(7)
    target = LeadOutreach(rng)

    tactics = [
        EmailTactic("blunt_pitch", true_reply_rate=0.05),
        EmailTactic("free_audit_offer", true_reply_rate=0.28),  # the real winner
        EmailTactic("generic_followup", true_reply_rate=0.12),
    ]

    memory = InMemoryStore()
    goal = Goal(name="win_clients", description="Maximize replies from cold outreach")
    agent = Agent(target, tactics, policy=UCBPolicy(c=1.4), memory=memory, max_steps=300)

    result = agent.pursue(goal)

    chosen = Counter(s.tactic for s in result.steps)
    print(result.summary())
    print(f"Overall reply rate: {target.replies}/{target.sent} "
          f"= {target.replies / target.sent:.1%}\n")
    print("Times each tactic was chosen (the agent learned to favor the winner):")
    for name, n in chosen.most_common():
        st = memory.stats(name, result.steps[0].signature)
        print(f"  {name:18s} chosen {n:4d}x   learned reply-rate {st.success_rate:.1%}")


if __name__ == "__main__":
    main()
