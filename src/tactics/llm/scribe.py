"""Scribe — closes the learning loop by writing the run down.

After a run, the numeric result (rewards in Memory) is captured automatically —
but the *narrative* result (what actually happened, what to do differently)
evaporates with the journal. The Scribe fixes that: hand it a finished
:class:`~tactics.core.engine.RunResult` or
:class:`~tactics.colony.colony.ColonyResult` and it asks a model to distill the
journal and findings into a few durable :class:`~tactics.core.lessons.Lesson`\\ s,
which it records in a :class:`~tactics.core.lessons.LessonStore`.

Same safety posture as the rest of the LLM layer: offline-testable via
``ScriptedClient``; an unparseable or failed distillation writes *nothing* —
silence, not garbage, and never a crash. The Scribe is deliberately conservative:
its system prompt demands few, specific, evidence-backed lessons, because a bad
lesson pollutes every future prompt that includes it.
"""

from __future__ import annotations

from ..core.lessons import Lesson, LessonStore
from .client import LLMClient, extract_json

_SYSTEM = (
    "You are the scribe of an autonomous agent system. After a run, you distill "
    "what was durably learned into at most a few short lessons that will be fed "
    "to future runs. Be strict: record only insights that are specific, "
    "actionable, and supported by the evidence shown — never restate the goal, "
    "never pad, never speculate. If nothing durable was learned, return an empty "
    "list. Respond ONLY with the requested JSON object."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lessons"],
    "additionalProperties": False,
}


class Scribe:
    SYSTEM = _SYSTEM
    SCHEMA: dict = _SCHEMA

    def __init__(
        self,
        client: LLMClient,
        store: LessonStore,
        *,
        system: str | None = None,
        max_lessons: int = 5,
        journal_tail: int = 40,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.store = store
        self.system = system or self.SYSTEM
        self.max_lessons = max_lessons
        self.journal_tail = journal_tail
        self.max_tokens = max_tokens

    # --- evidence assembly ----------------------------------------------------

    def build_prompt(self, result, playbook: str | None) -> str:  # noqa: ANN001
        goal = getattr(result, "goal", None)
        goal_name = getattr(goal, "name", "(unknown)")
        summary = result.summary() if hasattr(result, "summary") else repr(result)

        journal = getattr(result, "journal", None)
        journal_text = journal.explain(limit=self.journal_tail) if journal else "(no journal)"

        findings = getattr(result, "findings", None) or []
        findings_text = "\n".join(
            f"- {f.source}: {f.kind} {f.detail}" for f in findings[-20:]
        ) or "(no findings)"

        return (
            "An autonomous run just finished. Distill what was durably learned.\n\n"
            f"Playbook: {playbook or '(general)'}\n"
            f"Goal: {goal_name}\n"
            f"Result: {summary}\n\n"
            f"Journal (last {self.journal_tail} events):\n{journal_text}\n\n"
            f"Findings:\n{findings_text}\n\n"
            f"Return at most {self.max_lessons} lessons as JSON: "
            '{"lessons": [{"text": "...", "tags": ["..."], "evidence": "..."}]}. '
            "An empty list is a valid and often correct answer."
        )

    # --- the one job ------------------------------------------------------------

    def distill(self, result, *, playbook: str | None = None) -> list[Lesson]:  # noqa: ANN001
        """Turn a finished run into recorded lessons. Failure records nothing."""
        goal = getattr(result, "goal", None)
        goal_name = getattr(goal, "name", None)
        journal = getattr(result, "journal", None)
        try:
            resp = self.client.complete(
                self.build_prompt(result, playbook),
                system=self.system,
                schema=self.SCHEMA,
                max_tokens=self.max_tokens,
            )
            data = extract_json(resp.text)
        except Exception:  # no lesson beats a fabricated lesson — fail silent
            if journal:
                journal.record("scribe.failed")
            return []

        lessons: list[Lesson] = []
        for row in data.get("lessons") or []:
            if len(lessons) >= self.max_lessons:
                break
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            lesson = Lesson(
                text=text,
                playbook=playbook,
                goal=goal_name,
                tags=[str(t) for t in row.get("tags", [])],
                evidence=str(row.get("evidence", "")),
            )
            self.store.add(lesson)
            lessons.append(lesson)

        if journal:
            journal.record("scribe.distilled", lessons=len(lessons), tokens=resp.tokens)
        return lessons
