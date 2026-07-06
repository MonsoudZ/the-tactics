"""Tests for verbal memory — the lesson store, the scribe, and prompt injection.

All offline: the scribe runs against ScriptedClient, stores against tmp_path.
"""

from __future__ import annotations

import json

from tactics import (
    Context,
    Goal,
    InMemoryLessons,
    Journal,
    JsonlLessons,
    Lesson,
    Target,
    render_lessons,
)
from tactics.core.engine import RunResult
from tactics.llm import LLMTactic, Scribe, ScriptedClient


class _Tgt(Target):
    name = "leads"

    def observe(self) -> dict:
        return {}


def _ctx():
    return Context(target=_Tgt(), goal=Goal(name="win-clients"))


# --- Lesson / render ----------------------------------------------------------


def test_render_lessons_empty_is_empty_string():
    assert render_lessons([]) == ""


def test_render_lessons_formats_scope():
    block = render_lessons([Lesson(text="verify websiteUri manually", playbook="leads")])
    assert "Lessons from past runs:" in block
    assert "[leads] verify websiteUri manually" in block


# --- InMemoryLessons.relevant ---------------------------------------------------


def test_relevant_prefers_matching_playbook_and_goal():
    store = InMemoryLessons()
    store.add(Lesson(text="general truth"))
    store.add(Lesson(text="leads truth", playbook="leads", goal="win-clients"))
    top = store.relevant(playbook="leads", goal="win-clients", limit=2)
    assert top[0].text == "leads truth"
    assert top[1].text == "general truth"  # general lessons stay eligible


def test_relevant_excludes_other_playbooks():
    store = InMemoryLessons()
    store.add(Lesson(text="trading truth", playbook="trading"))
    assert store.relevant(playbook="leads") == []


def test_relevant_uses_keyword_overlap_and_respects_limit():
    store = InMemoryLessons()
    store.add(Lesson(text="rubocop autocorrect breaks heredocs"))
    store.add(Lesson(text="places api omits websiteUri"))
    top = store.relevant(query="check the websiteUri field from the places api", limit=1)
    assert len(top) == 1
    assert "websiteUri" in top[0].text


def test_relevant_breaks_ties_by_recency():
    store = InMemoryLessons()
    store.add(Lesson(text="older", ts=1.0))
    store.add(Lesson(text="newer", ts=2.0))
    assert store.relevant(limit=2)[0].text == "newer"


# --- JsonlLessons persistence ---------------------------------------------------


def test_jsonl_lessons_roundtrip(tmp_path):
    path = str(tmp_path / "lessons.jsonl")
    store = JsonlLessons(path)
    store.add(Lesson(text="persist me", playbook="leads", tags=["io"]))

    reloaded = JsonlLessons(path)
    got = list(reloaded.entries())
    assert len(got) == 1
    assert got[0].text == "persist me"
    assert got[0].playbook == "leads"
    assert got[0].tags == ["io"]


def test_jsonl_lessons_is_append_only_one_line_per_lesson(tmp_path):
    path = str(tmp_path / "lessons.jsonl")
    store = JsonlLessons(path)
    store.add(Lesson(text="one"))
    store.add(Lesson(text="two"))
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "one"


def test_jsonl_lessons_skips_corrupt_lines(tmp_path):
    path = str(tmp_path / "lessons.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"text": "good"}\n')
        fh.write("not json at all\n")
        fh.write('{"text": "also good"}\n')
    store = JsonlLessons(path)
    assert [lesson.text for lesson in store.entries()] == ["good", "also good"]


# --- Scribe ---------------------------------------------------------------------


def _result() -> RunResult:
    return RunResult(goal=Goal(name="win-clients"), journal=Journal())


def test_scribe_distills_and_records_lessons():
    store = InMemoryLessons()
    client = ScriptedClient([json.dumps({
        "lessons": [
            {"text": "cold email needs SPF/DKIM", "tags": ["email"], "evidence": "0 replies"},
            {"text": "call local businesses instead"},
        ]
    })])
    result = _result()
    got = Scribe(client, store).distill(result, playbook="leads")

    assert [lesson.text for lesson in got] == [
        "cold email needs SPF/DKIM", "call local businesses instead",
    ]
    stored = list(store.entries())
    assert stored[0].playbook == "leads"
    assert stored[0].goal == "win-clients"
    assert stored[0].evidence == "0 replies"
    assert result.journal.of_kind("scribe.distilled")


def test_scribe_unparseable_response_writes_nothing():
    store = InMemoryLessons()
    result = _result()
    got = Scribe(ScriptedClient(["that was a great run!"]), store).distill(result)
    assert got == []
    assert list(store.entries()) == []
    assert result.journal.of_kind("scribe.failed")


def test_scribe_client_error_writes_nothing():
    def boom(prompt, system):
        raise RuntimeError("api down")

    store = InMemoryLessons()
    got = Scribe(ScriptedClient(boom), store).distill(_result())
    assert got == []
    assert list(store.entries()) == []


def test_scribe_caps_lessons_and_drops_empty_text():
    store = InMemoryLessons()
    client = ScriptedClient([json.dumps({
        "lessons": [{"text": ""}, {"text": "a"}, {"text": "b"}, {"text": "c"}],
    })])
    got = Scribe(client, store, max_lessons=2).distill(_result())
    assert [lesson.text for lesson in got] == ["a", "b"]


def test_scribe_empty_lessons_is_valid():
    store = InMemoryLessons()
    got = Scribe(ScriptedClient(['{"lessons": []}']), store).distill(_result())
    assert got == []


# --- LLMTactic lesson injection ---------------------------------------------------


class _Judge(LLMTactic):
    def build_prompt(self, ctx) -> str:  # noqa: ANN001
        return "Judge this outreach email about the places api."


def _ok_client() -> ScriptedClient:
    return ScriptedClient(['{"success": true, "reward": 1.0, "notes": "fine"}'])


def test_llmtactic_prepends_relevant_lessons():
    store = InMemoryLessons()
    store.add(Lesson(text="places api omits websiteUri", playbook="leads"))
    client = _ok_client()
    outcome = _Judge("judge", client, lessons=store).execute(_ctx())

    assert outcome.success
    prompt = client.calls[0]["prompt"]
    assert prompt.startswith("Lessons from past runs:")
    assert "places api omits websiteUri" in prompt
    assert "Judge this outreach email" in prompt


def test_llmtactic_excludes_foreign_playbook_lessons():
    store = InMemoryLessons()
    store.add(Lesson(text="trading-only wisdom", playbook="trading"))
    client = _ok_client()
    _Judge("judge", client, lessons=store).execute(_ctx())
    assert "trading-only wisdom" not in client.calls[0]["prompt"]


def test_llmtactic_without_store_is_unchanged():
    client = _ok_client()
    _Judge("judge", client).execute(_ctx())
    assert client.calls[0]["prompt"] == "Judge this outreach email about the places api."


def test_full_loop_scribe_feeds_next_tactic(tmp_path):
    """The compounding loop: run #1's scribe informs run #2's tactic."""
    store = JsonlLessons(str(tmp_path / "lessons.jsonl"))

    scribe_client = ScriptedClient([json.dumps({
        "lessons": [{"text": "verify websiteUri before outreach", "tags": ["places"]}],
    })])
    Scribe(scribe_client, store).distill(_result(), playbook="leads")

    tactic_client = _ok_client()
    _Judge("judge", tactic_client, lessons=JsonlLessons(store.path)).execute(_ctx())
    assert "verify websiteUri before outreach" in tactic_client.calls[0]["prompt"]
