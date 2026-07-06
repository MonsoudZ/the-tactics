"""Dogfood run — the-tactics audits *itself*, then writes down what it learned.

This is the whole framework eating its own cooking: point the ``repo_health``
playbook at this very repository, let the colony run its checks (tests, lint,
secret scan) as a verifying swarm, then hand the finished run to the
:class:`~tactics.llm.scribe.Scribe` so it distills a durable
:class:`~tactics.core.lessons.Lesson` about *this codebase* — lesson #9, appended
to the same ``.tactics/lessons.jsonl`` that seeds every future ``LLMTactic``.

Two axes you control:

* **Which brain.** If ``ANTHROPIC_API_KEY`` is set we use the live
  :class:`~tactics.llm.client.ClaudeClient` (``claude-opus-4-8``) and the lesson
  is genuinely model-distilled from the run's journal. Otherwise we fall back to
  a :class:`~tactics.llm.client.ScriptedClient` so the demo runs offline and
  deterministically — the audit is real, but the "distillation" is a canned
  (clearly-labelled) stand-in.

* **Review vs commit.** By default we distill into a throwaway in-memory store
  and *print* the candidate lesson — nothing touches disk. Pass ``--commit`` to
  append it to ``.tactics/lessons.jsonl`` for real. This mirrors the Scribe's own
  fail-silent ethos: a bad lesson pollutes every future prompt, so you get to
  look before it lands.

Run it:
    python3 examples/dogfood_repo_health.py            # audit + review lesson #9
    python3 examples/dogfood_repo_health.py --commit    # ...and append it for real
"""

from __future__ import annotations

import argparse
import os

from tactics import InMemoryLessons, JsonlLessons, LessonStore
from tactics.llm import ScriptedClient
from tactics.playbooks.repo_health import CodeRepo, audit_goal, build_audit_colony

# Where committed verbal memory lives — knowledge, not state (see .gitignore).
LESSONS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, ".tactics", "lessons.jsonl")

# The offline stand-in. This is a *scripted* distillation, not a live one — so it
# asserts only what THIS run demonstrates no matter how the individual checks land:
# the framework can audit its own repository and feed the colony's findings straight
# into the Scribe, closing the numeric+verbal learning loop in a single pass. A live
# ClaudeClient (with ANTHROPIC_API_KEY set) distills from the real journal instead.
_SCRIPTED_LESSON = (
    '{"lessons": [{'
    '"text": "the-tactics can dogfood repo_health on its own checkout: the audit '
    "colony's verified findings become the Scribe's evidence in the same run, so "
    'numeric memory (rewards) and verbal memory (lessons) are written from one pass '
    'over the same repo \\u2014 wire the Scribe to the committed lessons.jsonl to make '
    'it compound.", '
    '"tags": ["dogfood", "repo-health", "scribe", "self-audit"], '
    '"evidence": "this self-audit ran tests + lint + secret-scan on the-tactics and '
    'handed the resulting RunResult straight to the Scribe"}]}'
)


def _build_scribe(store: LessonStore):
    """Live Scribe if a key is present, else an offline scripted one."""
    from tactics.llm import Scribe

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        from tactics.llm import ClaudeClient

        return Scribe(ClaudeClient(), store), "live (claude-opus-4-8)"
    return Scribe(ScriptedClient([_SCRIPTED_LESSON]), store), "offline (scripted stand-in)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="append the distilled lesson to .tactics/lessons.jsonl (default: review only)",
    )
    parser.add_argument(
        "--path", default=os.path.join(os.path.dirname(__file__), os.pardir),
        help="repo to audit (default: the-tactics itself)",
    )
    args = parser.parse_args()

    # --- 1. audit this repo as a verifying swarm -----------------------------
    repo = CodeRepo(os.path.abspath(args.path))
    colony = build_audit_colony(repo, max_workers=1)  # max_workers=1 = deterministic
    result = colony.run(audit_goal())

    print("=== Audit ===")
    print(result.summary())
    for f in result.findings:
        print(f"  {f.kind:12s} {f.detail}")

    # --- 2. distill the run into a lesson ------------------------------------
    # Review mode writes to a throwaway store; --commit writes to the real file.
    existing = len(list(JsonlLessons(LESSONS_PATH).entries())) if os.path.exists(LESSONS_PATH) else 0
    store: LessonStore = JsonlLessons(LESSONS_PATH) if args.commit else InMemoryLessons()
    scribe, mode = _build_scribe(store)

    print(f"\n=== Scribe ({mode}) ===")
    lessons = scribe.distill(result, playbook=repo.name)

    if not lessons:
        print("Nothing durable distilled — the Scribe wrote nothing (that's a valid answer).")
        return

    for i, lesson in enumerate(lessons, start=existing + 1):
        print(f"  lesson #{i}: {lesson.text}")
        if lesson.evidence:
            print(f"             evidence: {lesson.evidence}")

    if args.commit:
        print(f"\nAppended {len(lessons)} lesson(s) to {os.path.normpath(LESSONS_PATH)}.")
    else:
        print("\nReview only — nothing written. Re-run with --commit to append for real.")


if __name__ == "__main__":
    main()
