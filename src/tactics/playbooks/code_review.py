"""Code-review playbook — the brain reads real code and finds real problems.

This is the judgment capability: a tactic that feeds a file to the model and gets
back structured findings — bugs, security issues, bad patterns, dead/unused code,
risks — each with a severity and a line. The colony fans this out over many files
in parallel; an (optional) LLM critic can verify each finding before it's trusted.

Offline-testable with ScriptedClient; live with ClaudeClient (uses your key,
spends tokens — the Budget caps it).

    from tactics.llm import ClaudeClient
    from tactics.playbooks.repo_health import CodeRepo
    from tactics.playbooks.code_review import build_review_colony, review_goal
    colony = build_review_colony(CodeRepo("."), ClaudeClient(), files=["app/models/user.rb"])
    result = colony.run(review_goal())
"""

from __future__ import annotations

from ..colony import AcceptCritic, Colony, Critic, FunctionPlanner
from ..core.goal import Goal
from ..core.memory import InMemoryStore, MemoryStore
from ..core.outcome import Outcome
from ..llm.client import LLMClient
from ..llm.tactic import LLMTactic
from .repo_health import CodeRepo

_REVIEW_SYSTEM = (
    "You are a rigorous senior engineer doing a code review. Find REAL problems: "
    "bugs, security issues, bad patterns, dead or unused code, and risky shortcuts. "
    "Cite the line number. Do not invent issues — if the file is clean, return an "
    "empty findings list. Respond ONLY with the requested JSON."
)

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "kind": {"type": "string"},
                    "line": {"type": "integer"},
                    "message": {"type": "string"},
                },
                "required": ["severity", "kind", "line", "message"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
    "additionalProperties": False,
}

_SEVERITY_COST = {"high": 0.34, "medium": 0.15, "low": 0.05}


class CodeReview(LLMTactic):
    SYSTEM = _REVIEW_SYSTEM
    SCHEMA = _REVIEW_SCHEMA

    def is_applicable(self, ctx) -> bool:  # noqa: ANN001
        return ctx.task is not None and "file" in (ctx.task.payload or {})

    def build_prompt(self, ctx) -> str:  # noqa: ANN001
        path = ctx.task.payload["file"]
        code = ctx.target.read(path)
        numbered = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(code.splitlines()))
        return (
            f"Review this file: {path}\n\n{numbered}\n\n"
            'Return JSON {"findings": [{"severity","kind","line","message"}], "summary": "..."}.'
        )

    def interpret(self, data: dict, resp, ctx) -> Outcome:  # noqa: ANN001
        findings = data.get("findings") or []
        penalty = sum(_SEVERITY_COST.get(f.get("severity", "low"), 0.05) for f in findings)
        reward = round(max(0.0, 1.0 - penalty), 3)  # clean file scores 1.0
        highs = sum(1 for f in findings if f.get("severity") == "high")
        return Outcome(
            success=highs == 0,
            reward=reward,
            cost=self._cost(resp),
            metrics={
                "file": ctx.task.payload.get("file"),
                "findings": findings,
                "summary": data.get("summary", ""),
            },
            notes=data.get("summary", ""),
        )


def review_goal() -> Goal:
    return Goal(name="code_review", description="Review the codebase for bugs and bad patterns")


_SRC_EXT = (".py", ".rb", ".swift", ".js", ".ts", ".tsx", ".go", ".java", ".rs")


def build_review_colony(
    target: CodeRepo,
    client: LLMClient,
    *,
    files: list[str] | None = None,
    critic: Critic | None = None,
    memory: MemoryStore | None = None,
    max_workers: int = 4,
    max_rounds: int | None = None,
) -> Colony:
    """Review ``files`` (default: tracked source files) one task each. Pass an
    ``LLMCritic`` to adversarially verify findings before they're trusted."""
    if files is None:
        files = [f for f in target.tracked_files() if f.endswith(_SRC_EXT)]

    def plan(goal, board, t):  # noqa: ANN001
        if board.tasks:
            return []
        return [board.post_task(f"review:{f}", payload={"file": f}, signal="review") for f in files]

    return Colony(
        target,
        [CodeReview("code_review", client)],
        FunctionPlanner(plan),
        memory=memory or InMemoryStore(),
        critic=critic or AcceptCritic(),
        max_workers=max_workers,
        max_rounds=max_rounds if max_rounds is not None else len(files) + 2,
    )
