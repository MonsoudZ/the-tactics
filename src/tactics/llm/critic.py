"""LLMCritic — an adversarial verifier powered by a model.

The colony only learns from outcomes the Critic accepts. An LLM critic re-checks
a tactic's claimed result and, by design, **defaults to rejecting when unsure** —
that bias is what keeps plausible-but-wrong work out of memory and out of any
committed action. If the model call fails, that too is a rejection (fail-closed).

Subclass and override :meth:`build_prompt` to give the verifier the real evidence
(test output, the drafted email, the diff). The default prompt summarizes the
outcome generically.
"""

from __future__ import annotations

from ..colony.critic import Critic, Verdict
from .client import LLMClient, extract_json

_SYSTEM = (
    "You are a strict, adversarial verifier inside an autonomous agent. Your job "
    "is to catch results that are wrong, fabricated, unsafe, or merely plausible. "
    "Assume the claim is wrong until the evidence convinces you otherwise. If you "
    "are uncertain, REJECT. Respond ONLY with the requested JSON object."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "accepted": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["accepted", "reason"],
    "additionalProperties": False,
}


class LLMCritic(Critic):
    SYSTEM = _SYSTEM
    SCHEMA: dict = _SCHEMA

    def __init__(
        self,
        client: LLMClient,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.system = system or self.SYSTEM
        self.max_tokens = max_tokens

    def build_prompt(self, outcome, ctx) -> str:  # noqa: ANN001
        task = getattr(ctx, "task", None)
        task_desc = getattr(task, "description", "(no task)") if task else "(no task)"
        return (
            "Verify the result of an autonomous action before it is trusted.\n\n"
            f"Task: {task_desc}\n"
            f"Reported success: {outcome.success}\n"
            f"Reported reward: {outcome.reward}\n"
            f"Notes/metrics: {outcome.notes} {outcome.metrics}\n\n"
            "Is this result genuinely correct and safe to trust? Reply with JSON: "
            '{"accepted": <bool>, "reason": "<short reason>"}'
        )

    def verify(self, outcome, ctx) -> Verdict:  # noqa: ANN001
        try:
            resp = self.client.complete(
                self.build_prompt(outcome, ctx),
                system=self.system,
                schema=self.SCHEMA,
                max_tokens=self.max_tokens,
            )
            if getattr(ctx, "journal", None):
                ctx.journal.record("llm.critic", tokens=resp.tokens)
            data = extract_json(resp.text)
            return Verdict(accepted=bool(data.get("accepted", False)),
                           reason=str(data.get("reason", "")))
        except Exception as exc:  # fail-closed: a broken verifier rejects, never rubber-stamps
            return Verdict(accepted=False, reason=f"verifier error: {exc!r}")
