"""Tests for the LLM layer — all offline via ScriptedClient / injected fakes."""

from __future__ import annotations

import pytest

from tactics import Context, Goal, Outcome, Target
from tactics.colony import Blackboard
from tactics.llm import ClaudeClient, LLMCritic, LLMTactic, ScriptedClient, extract_json


class _Tgt(Target):
    name = "t"

    def observe(self) -> dict:
        return {}


def _ctx(task=None):
    return Context(target=_Tgt(), goal=Goal(name="g"), task=task)


# --- extract_json ------------------------------------------------------------


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_code_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_finds_object_in_prose():
    assert extract_json('Sure! {"a": 1, "b": 2} hope that helps') == {"a": 1, "b": 2}


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no json here")


# --- LLMTactic ---------------------------------------------------------------


class Draft(LLMTactic):
    def build_prompt(self, ctx) -> str:
        return "draft something"


def test_llm_tactic_parses_outcome_and_charges_tokens():
    client = ScriptedClient(['{"success": true, "reward": 0.8, "notes": "good"}'])
    outcome = Draft("draft", client).execute(_ctx())
    assert outcome.success is True
    assert outcome.reward == 0.8
    assert outcome.notes == "good"
    assert outcome.cost > 0  # token-based cost flows into the Budget


def test_llm_tactic_unparseable_is_a_loss_not_a_crash():
    client = ScriptedClient(["the model rambled with no json"])
    outcome = Draft("draft", client).execute(_ctx())
    assert outcome.success is False
    assert outcome.reward == 0.0


def test_llm_tactic_fixed_cost_overrides_tokens():
    client = ScriptedClient(['{"success": true, "reward": 1.0, "notes": ""}'])
    outcome = Draft("draft", client, cost_per_call=0.03).execute(_ctx())
    assert outcome.cost == 0.03


def test_llm_tactic_passes_schema_to_client():
    client = ScriptedClient(['{"success": true, "reward": 1.0, "notes": ""}'])
    Draft("draft", client).execute(_ctx())
    assert client.calls[0]["schema"] is not None  # structured output requested


# --- LLMCritic ---------------------------------------------------------------


def test_llm_critic_accepts_and_rejects():
    accept = LLMCritic(ScriptedClient(['{"accepted": true, "reason": "ok"}']))
    reject = LLMCritic(ScriptedClient(['{"accepted": false, "reason": "nope"}']))
    out = Outcome.win(1.0)
    assert accept.verify(out, _ctx()).accepted is True
    assert reject.verify(out, _ctx()).accepted is False


def test_llm_critic_fails_closed_on_error():
    def boom(prompt, system):
        raise RuntimeError("api down")

    verdict = LLMCritic(ScriptedClient(boom)).verify(Outcome.win(1.0), _ctx())
    assert verdict.accepted is False  # broken verifier rejects, never rubber-stamps
    assert "error" in verdict.reason


def test_llm_critic_fails_closed_on_garbage():
    verdict = LLMCritic(ScriptedClient(["not json"])).verify(Outcome.win(1.0), _ctx())
    assert verdict.accepted is False


# --- ClaudeClient adapter (no network: inject a fake SDK client) -------------


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    input_tokens = 11
    output_tokens = 7


class _Msg:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.usage = _Usage()


class _FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Msg('{"ok": true}')


class _FakeSDK:
    def __init__(self):
        self.messages = _FakeMessages()


def test_claude_client_builds_request_and_reads_usage():
    fake = _FakeSDK()
    client = ClaudeClient(client=fake, max_tokens=512)
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    resp = client.complete("hi", system="sys", schema=schema)

    kw = fake.messages.last_kwargs
    assert kw["model"] == "claude-opus-4-8"
    assert kw["max_tokens"] == 512
    assert kw["system"] == "sys"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in kw  # Opus 4.x rejects temperature
    assert resp.text == '{"ok": true}'
    assert resp.tokens == 18


def test_claude_client_thinking_flag_adds_adaptive():
    fake = _FakeSDK()
    ClaudeClient(client=fake, thinking=True).complete("hi")
    assert fake.messages.last_kwargs["thinking"] == {"type": "adaptive"}
