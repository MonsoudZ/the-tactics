"""The scribe's client, on the same auth the agents already use.

Offline: the SDK call is injected. What matters is that a completion needs no
API key, that a schema still produces parseable JSON, and that a client whose
job is answering questions cannot touch the filesystem.
"""

from __future__ import annotations

import json

import pytest

from tactics.llm.client import SdkClient, extract_json


def _runner(text="ok", **extra):
    seen = {}

    def run(prompt, system, model):
        seen.update(prompt=prompt, system=system, model=model)
        return {"text": text, "input_tokens": 11, "output_tokens": 3, **extra}

    run.seen = seen
    return run


def test_a_completion_needs_no_api_key(monkeypatch):
    # The whole point: agents run on the Claude Code CLI's own auth, so the
    # scribe demanding ANTHROPIC_API_KEY left the two halves of memory on
    # different credentials.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    reply = SdkClient(runner=_runner("pong")).complete("ping")
    assert reply.text == "pong"
    assert reply.input_tokens == 11 and reply.output_tokens == 3


def test_a_schema_is_asked_for_in_words_and_parses_back(monkeypatch):
    # The SDK has no output_config.format, so the shape is requested rather than
    # constrained — and the answer is parsed leniently.
    schema = {"type": "object", "properties": {"lessons": {"type": "array"}}}
    run = _runner('```json\n{"lessons": ["x"]}\n```')
    reply = SdkClient(runner=run).complete("distil this", schema=schema)
    assert "lessons" in run.seen["prompt"] and json.dumps(schema) in run.seen["prompt"]
    assert extract_json(reply.text) == {"lessons": ["x"]}


def test_the_system_prompt_and_model_reach_the_sdk():
    run = _runner()
    SdkClient(model="some-model", runner=run).complete("p", system="be terse")
    assert run.seen["system"] == "be terse" and run.seen["model"] == "some-model"


def test_a_failed_call_raises_rather_than_returning_empty_text():
    # An empty string would read as "the model had nothing to say", and the
    # scribe writes nothing on an unparseable answer — so the failure would be
    # indistinguishable from a considered silence.
    with pytest.raises(RuntimeError, match="boom"):
        SdkClient(runner=_runner(error="boom")).complete("p")


def test_the_client_refuses_every_tool_call():
    # A completion has no business touching the filesystem, and the SDK's agent
    # loop would otherwise be free to. Denial is enforced by the hook rather
    # than by declining to grant, because a grant elsewhere would shadow that.
    import asyncio
    import dataclasses
    import sys

    from tactics.llm import client as mod

    @dataclasses.dataclass
    class _Options:                      # a real dataclass: the code reads its fields
        system_prompt: object = None
        model: object = None
        max_turns: object = None
        hooks: object = None

    class _Matcher:
        def __init__(self, hooks):
            self.hooks = hooks

    seen = {}

    async def _query(prompt, options):
        seen["options"] = options
        return
        yield  # pragma: no cover - unreachable; makes this an async generator

    fake = type("m", (), {"ClaudeAgentOptions": _Options, "HookMatcher": _Matcher,
                          "query": staticmethod(_query)})
    monkey = pytest.MonkeyPatch()
    monkey.setitem(sys.modules, "claude_agent_sdk", fake)
    try:
        mod._sdk_complete("p", None, None)
    finally:
        monkey.undo()

    hook = seen["options"].hooks["PreToolUse"][0].hooks[0]
    decision = asyncio.run(hook({"tool_name": "Write"}, "id", None))
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
