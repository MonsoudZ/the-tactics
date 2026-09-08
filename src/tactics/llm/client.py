"""The LLM client seam — one interface, swappable implementations.

Everything in this layer depends only on :class:`LLMClient.complete`. That keeps
the framework provider-agnostic and, crucially, **testable without a network or
API key**: tests and demos use :class:`ScriptedClient`; production uses
:class:`ClaudeClient`, which lazy-imports the official ``anthropic`` SDK so the
package installs and its tests run even when ``anthropic`` isn't present.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

# Default to the most capable model; callers can override per ClaudeClient.
DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(Protocol):
    """Minimal contract: turn a prompt into text, optionally schema-constrained."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = ...,
        schema: dict | None = ...,
        max_tokens: int | None = ...,
    ) -> LLMResponse: ...


def extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object out of a model response.

    Tolerates code fences and surrounding prose; raises ValueError if nothing
    JSON-shaped is present so callers can treat it as a failed judgment.
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError(f"no parseable JSON in response: {text!r}") from exc
        raise ValueError(f"no JSON object in response: {text!r}")


class ClaudeClient:
    """Talk to Claude through the official ``anthropic`` SDK.

    The SDK is imported lazily on first use, so importing this module (and running
    the test suite) never requires ``anthropic`` to be installed. Pass ``client``
    to inject a pre-built/fake SDK client (used in tests).

    Note: per the Claude API, Opus-4.x models reject ``temperature``/``top_p`` —
    this client never sends them. Structured output is requested via
    ``output_config.format`` when a ``schema`` is provided.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._client = client
        self._api_key = api_key

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415 - lazy by design
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError(
                    "ClaudeClient needs the 'anthropic' package. "
                    "Install it with: python3 -m pip install 'tactics[llm]'"
                ) from exc
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        msg = client.messages.create(**kwargs)
        text = "".join(
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"
        )
        usage = getattr(msg, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            raw=msg,
        )


class ScriptedClient:
    """Deterministic LLM stand-in for tests and offline demos.

    ``responses`` is either a list of strings (returned in order, last repeats) or
    a callable ``(prompt, system) -> str``. Records every call for assertions.
    """

    def __init__(self, responses: list[str] | Callable[[str, str | None], str]) -> None:
        self._responses = responses
        self._i = 0
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        if callable(self._responses):
            text = self._responses(prompt, system)
        else:
            text = self._responses[min(self._i, len(self._responses) - 1)]
            self._i += 1
        return LLMResponse(text=text, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)


class SdkClient:
    """Talk to Claude through the **Claude Agent SDK** — no API key required.

    :class:`ClaudeClient` needs ``ANTHROPIC_API_KEY`` and bills the API. The
    agents in the agent-sdk playbook need neither: they run through the Claude
    Code CLI, which carries its own authentication. That left the two halves of
    memory authenticated differently — the execution half worked while the
    verbal half (the Scribe) reported "no ANTHROPIC_API_KEY" in exactly the
    environments where agents ran fine. This closes that: same SDK, same auth,
    same subscription.

    Two deliberate differences from the API path:

    * **Every tool call is denied.** A completion has no business touching the
      filesystem, and the SDK's agent loop would otherwise be free to. The hook
      is the seam that cannot be shadowed, so the denial is real rather than a
      matter of not granting anything.
    * **``schema`` becomes an instruction, not a constraint.** The SDK has no
      ``output_config.format``, so the shape is asked for in words and parsed
      leniently by :func:`extract_json`. Callers already treat an unparseable
      answer as a failed judgment, which is the right posture for a request the
      model was merely asked to honour.

    ``max_tokens`` is accepted for interface compatibility and ignored: the SDK
    exposes no equivalent.
    """

    def __init__(self, model: str | None = None, *, runner: Callable[..., Any] | None = None) -> None:
        self.model = model
        self._runner = runner          # test seam; the real one drives the SDK

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if schema is not None:
            prompt = (
                f"{prompt}\n\nReply with a single JSON object matching this schema, "
                f"and nothing else:\n{json.dumps(schema)}"
            )
        run = (self._runner or _sdk_complete)(prompt, system, self.model)
        if run.get("error"):
            raise RuntimeError(f"SdkClient: {run['error']}")
        return LLMResponse(
            text=run.get("text", ""),
            input_tokens=int(run.get("input_tokens", 0)),
            output_tokens=int(run.get("output_tokens", 0)),
            raw=run,
        )


def _sdk_complete(prompt: str, system: str | None, model: str | None) -> dict:
    """One tool-free turn through the Agent SDK. Lazy-imported, never raises."""
    import asyncio  # noqa: PLC0415 - lazy by design
    from dataclasses import fields as dataclass_fields  # noqa: PLC0415

    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return {"error": f"claude-agent-sdk not installed: {exc}"}

    async def deny(hook_input, tool_use_id, context):  # noqa: ANN001 - SDK contract
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "SdkClient answers questions; it does not use tools",
        }}

    wanted = {
        "system_prompt": system,
        "model": model,
        "max_turns": 1,
        "hooks": {"PreToolUse": [HookMatcher(hooks=[deny])]},
    }
    known = {f.name for f in dataclass_fields(ClaudeAgentOptions)}
    options = ClaudeAgentOptions(**{k: v for k, v in wanted.items() if v is not None and k in known})

    out: dict[str, Any] = {"text": "", "input_tokens": 0, "output_tokens": 0}

    async def drive() -> None:
        chunks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            for block in getattr(message, "content", None) or []:
                if hasattr(block, "text") and not hasattr(block, "name"):
                    chunks.append(getattr(block, "text", ""))
            if hasattr(message, "total_cost_usd"):      # the ResultMessage
                usage = getattr(message, "model_usage", None) or {}
                out["input_tokens"] = sum(int(u.get("inputTokens", 0) or 0) for u in usage.values())
                out["output_tokens"] = sum(int(u.get("outputTokens", 0) or 0) for u in usage.values())
                out["cost_usd"] = float(getattr(message, "total_cost_usd", None) or 0.0)
        out["text"] = "\n".join(c for c in chunks if c).strip()

    try:
        asyncio.run(drive())
    except Exception as exc:  # noqa: BLE001 - a failed judgment is a loss, not a crash
        out["error"] = repr(exc)
    return out
