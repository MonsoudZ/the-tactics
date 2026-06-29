"""Set a goal, watch the brain achieve it — with REAL tools on a REAL file.

We plant a Python file with unused imports (a real lint error), set the goal
"lint is clean", and hand the brain a fix tactic. It loops: see it's dirty ->
fix it -> check again -> done. Then we show the same goal under a DryRun gate,
where the fix is held and the goal is NOT reached — the safety story.

Run it:  python examples/goal_achievement_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tactics import Agent, AutoApprove, DryRun
from tactics.playbooks.repo_health import CodeRepo, FixCommand, command_passes_goal

MESSY = "import os\nimport sys\n\nVALUE = 1\n"  # os and sys are unused -> ruff F401


def run(gate, label: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "messy.py"
        f.write_text(MESSY, encoding="utf-8")

        repo = CodeRepo(str(d))
        goal = command_passes_goal("lint_clean", ["ruff", "check", "."])
        fix = FixCommand("fix_lint", ["ruff", "check", "--fix", "."])

        before = repo.run(["ruff", "check", "."])[0]
        result = Agent(repo, [fix], gate=gate, max_steps=4).pursue(goal)
        after = repo.run(["ruff", "check", "."])[0]

        print(f"\n=== {label} ===")
        print(f"lint before: {'CLEAN' if before == 0 else 'DIRTY'}")
        print(result.summary())
        print(f"goal achieved: {result.satisfied}   stop reason: {result.stop_reason}")
        print(f"lint after:  {'CLEAN' if after == 0 else 'DIRTY'}")
        print(f"file now: {f.read_text().strip()!r}")


def main() -> None:
    # Autonomous: the gate approves the (reversible) fix, so the brain reaches the goal.
    run(AutoApprove(), "Autonomous (AutoApprove) — it fixes until clean")
    # Safe: DryRun holds the fix, so the goal is NOT reached and nothing changed.
    run(DryRun(), "Dry-run (review-only) — fix held, goal not reached, file untouched")


if __name__ == "__main__":
    main()
