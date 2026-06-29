"""The core promise: set a goal, the brain works until it's achieved.

These use a stateful fake runner (no real tools) so they're fast and hermetic.
The 'lint' starts dirty; the fix tactic makes it clean; the Agent loops until the
goal's done-test passes.
"""

from __future__ import annotations

from tactics import Agent, AutoApprove, DryRun, Goal
from tactics.playbooks.repo_health import CodeRepo, FixCommand, command_passes_goal


def _repo_that_can_be_fixed():
    """A repo whose lint is dirty until `ruff check --fix` is run."""
    state = {"clean": False}

    def runner(cmd):
        if cmd[:3] == ["ruff", "check", "--fix"]:
            state["clean"] = True
            return 0, "fixed 1 issue"
        if cmd[:2] == ["ruff", "check"]:  # the goal's done-test
            return (0, "All checks passed!") if state["clean"] else (1, "F401 unused import")
        return 0, ""

    return CodeRepo(runner=runner), state


def test_agent_achieves_goal_by_acting():
    repo, state = _repo_that_can_be_fixed()
    goal = command_passes_goal("lint_clean", ["ruff", "check", "."])
    fix = FixCommand("fix_lint", ["ruff", "check", "--fix", "."])

    result = Agent(repo, [fix], gate=AutoApprove(), max_steps=5).pursue(goal)

    assert result.satisfied is True            # the goal was reached
    assert result.stop_reason == "goal satisfied"
    assert state["clean"] is True              # it actually fixed the thing
    assert len(result.steps) == 1              # one fix, then the goal test passed


def test_already_satisfied_goal_does_nothing():
    repo, state = _repo_that_can_be_fixed()
    state["clean"] = True  # already clean
    goal = command_passes_goal("lint_clean", ["ruff", "check", "."])
    result = Agent(repo, [FixCommand("fix_lint", ["ruff", "check", "--fix", "."])],
                   gate=AutoApprove(), max_steps=5).pursue(goal)
    assert result.satisfied is True
    assert len(result.steps) == 0  # nothing to do — recognized "done" immediately


def test_dry_run_holds_the_fix_so_goal_is_not_met():
    repo, state = _repo_that_can_be_fixed()
    goal = command_passes_goal("lint_clean", ["ruff", "check", "."])
    fix = FixCommand("fix_lint", ["ruff", "check", "--fix", "."])

    result = Agent(repo, [fix], gate=DryRun(), max_steps=3).pursue(goal)

    assert result.satisfied is False           # nothing was committed...
    assert state["clean"] is False             # ...the fix never ran
    assert all(s.outcome.success is False for s in result.steps)  # each held for approval


def test_fix_only_runs_when_applicable():
    ran = {"n": 0}

    def runner(cmd):
        ran["n"] += 1
        return 0, ""

    repo = CodeRepo(runner=runner)
    fix = FixCommand("noop", ["true"], applies_when=lambda ctx: False)
    # never applicable -> agent has nothing to do for an unsatisfiable goal
    goal = Goal(name="never", is_satisfied=lambda ctx: False)
    result = Agent(repo, [fix], gate=AutoApprove(), max_steps=3).pursue(goal)
    assert result.stop_reason == "no applicable tactic"
    assert ran["n"] == 0
