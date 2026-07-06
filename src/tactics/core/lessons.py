"""Lessons — the framework's *verbal* memory.

:class:`~tactics.core.memory.MemoryStore` remembers numbers (which tactic earned
what reward, where). Lessons remember *words*: the distilled "what we learned"
from a run — "Places API often omits websiteUri; verify before outreach",
"RuboCop autocorrect breaks heredocs in this repo". Numeric memory picks the next
tactic; verbal memory makes model-backed tactics start informed instead of cold.

The flow: a run finishes → a scribe (see ``tactics.llm.scribe``) distills its
journal and findings into a few :class:`Lesson`\\ s → they land in a
:class:`LessonStore` → every future :class:`~tactics.llm.tactic.LLMTactic` wired
to that store gets the relevant ones prepended to its prompt. Learning compounds
across runs, playbooks, and models.

Stores mirror the memory layer: :class:`InMemoryLessons` for tests,
:class:`JsonlLessons` for persistence (append-only JSONL — each lesson is one
line, so writes are O(1) and the file is greppable/human-editable).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Protocol

# Cheap tokenizer for relevance scoring — good enough, dependency-free.
_WORD_MIN = 3


def _terms(text: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if len(w) >= _WORD_MIN}


@dataclass
class Lesson:
    """One distilled insight. ``text`` is the lesson; the rest is provenance."""

    text: str
    playbook: str | None = None  # which domain it came from (Target.name)
    goal: str | None = None      # which goal was being pursued
    tags: list[str] = field(default_factory=list)
    evidence: str = ""           # short pointer to why we believe it
    source: str = "scribe"       # who wrote it (scribe, human, tactic name...)
    ts: float = field(default_factory=time.time)

    def render(self) -> str:
        scope = self.playbook or "general"
        return f"- [{scope}] {self.text}"


def render_lessons(lessons: Iterable[Lesson], *, header: str = "Lessons from past runs:") -> str:
    """Format lessons as a prompt block. Empty input renders to an empty string."""
    lines = [lesson.render() for lesson in lessons]
    if not lines:
        return ""
    return header + "\n" + "\n".join(lines)


class LessonStore(Protocol):
    """How lessons are written and recalled. Swap the backend freely."""

    def add(self, lesson: Lesson) -> None: ...

    def relevant(
        self,
        *,
        playbook: str | None = ...,
        goal: str | None = ...,
        query: str | None = ...,
        limit: int = ...,
    ) -> list[Lesson]: ...

    def entries(self) -> Iterator[Lesson]: ...


class InMemoryLessons:
    """Non-persistent store. Great for tests and single-process runs."""

    def __init__(self) -> None:
        self._lessons: list[Lesson] = []

    def add(self, lesson: Lesson) -> None:
        self._lessons.append(lesson)

    def entries(self) -> Iterator[Lesson]:
        yield from self._lessons

    def relevant(
        self,
        *,
        playbook: str | None = None,
        goal: str | None = None,
        query: str | None = None,
        limit: int = 8,
    ) -> list[Lesson]:
        """Most useful lessons first.

        Scoring is deliberately simple: exact playbook/goal matches score high
        (general lessons — ``playbook=None`` — always remain eligible), keyword
        overlap with ``query``/``tags`` adds a little, and recency breaks ties.
        A lesson scoped to a *different* playbook is excluded — what's true for
        trading isn't presumed true for lead-gen.
        """
        q_terms = _terms(query) if query else set()
        scored: list[tuple[float, float, Lesson]] = []
        for lesson in self._lessons:
            if playbook and lesson.playbook and lesson.playbook != playbook:
                continue  # scoped to another domain — not our business
            score = 0.0
            if playbook and lesson.playbook == playbook:
                score += 2.0
            if goal and lesson.goal == goal:
                score += 1.0
            if q_terms:
                l_terms = _terms(lesson.text) | {t.lower() for t in lesson.tags}
                score += len(q_terms & l_terms) * 0.25
            scored.append((score, lesson.ts, lesson))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [lesson for _, _, lesson in scored[:limit]]


class JsonlLessons(InMemoryLessons):
    """Persistent store backed by an append-only JSONL file.

    One lesson per line, so ``add`` is a single O(1) append (contrast
    ``JsonStore``'s full rewrite) and the file doubles as a human-readable,
    greppable, hand-editable knowledge base — delete a line to retract a lesson.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a corrupt line loses one lesson, never the store
                self._lessons.append(Lesson(**{
                    k: v for k, v in raw.items()
                    if k in {"text", "playbook", "goal", "tags", "evidence", "source", "ts"}
                }))

    def add(self, lesson: Lesson) -> None:
        super().add(lesson)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(lesson)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
