"""LLMTactic — a tactic whose work is a model's judgment.

Subclass it and implement :meth:`build_prompt` to describe the task from the
context. By default the model is asked to return JSON ``{success, reward, notes}``,
which becomes the :class:`Outcome` — so an LLM tactic competes and learns exactly
like a hand-written one. Token usage is reported as ``Outcome.cost`` so a Budget
can cap spend.

Override :meth:`interpret` if you want richer parsing (e.g. extract structured
findings into ``Outcome.metrics``).
"""

from __future__ import annotations


from ..core.outcome import Outcome
from ..core.tactic import Tactic
from .client import LLMClient, extract_json

_SYSTEM = (
    "You are a careful operator inside an autonomous agent. You do one job and "
    "report the result honestly. Never inflate the reward — it is a learning "
    "signal, and a fake high score poisons the system. Respond ONLY with the "
    "requested JSON object."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "reward": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["success", "reward", "notes"],
    "additionalProperties": False,
}


class LLMTactic(Tactic):
    SYSTEM = _SYSTEM
    SCHEMA: dict = _SCHEMA

    def __init__(
        self,
        name: str,
        client: LLMClient,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        cost_per_call: float | None = None,
    ) -> None:
        super().__init__(name=name)
        self.client = client
        self.system = system or self.SYSTEM
        self.max_tokens = max_tokens
        # If set, every call costs this (e.g. dollars). Else cost = tokens used.
        self.cost_per_call = cost_per_call

    def build_prompt(self, ctx) -> str:  # noqa: ANN001
        raise NotImplementedError("LLMTactic subclasses must implement build_prompt()")

    def _cost(self, resp) -> float:  # noqa: ANN001
        return self.cost_per_call if self.cost_per_call is not None else float(resp.tokens)

    def interpret(self, data: dict, resp, ctx) -> Outcome:  # noqa: ANN001
        return Outcome(
            success=bool(data.get("success", False)),
            reward=float(data.get("reward", 0.0)),
            cost=self._cost(resp),
            metrics={"tokens": resp.tokens},
            notes=str(data.get("notes", "")),
        )

    def execute(self, ctx) -> Outcome:  # noqa: ANN001
        prompt = self.build_prompt(ctx)
        resp = self.client.complete(
            prompt, system=self.system, schema=self.SCHEMA, max_tokens=self.max_tokens
        )
        if getattr(ctx, "journal", None):
            ctx.journal.record("llm.tactic", tactic=self.name, tokens=resp.tokens)
        try:
            data = extract_json(resp.text)
        except ValueError:
            # Unparseable judgment is a loss, not a crash — the colony moves on.
            return Outcome(success=False, reward=0.0, cost=self._cost(resp),
                           notes="LLM returned no parseable result")
        return self.interpret(data, resp, ctx)
