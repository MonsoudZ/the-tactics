"""Tests for gated fix tactics (acting on findings)."""

from __future__ import annotations


from tactics import Agent, AutoApprove, DryRun, Goal
from tactics.playbooks.repo_health import CodeRepo, UntrackFile


def _repo_with_committed_key(tmp_path):
    calls = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[:2] == ["git", "ls-files"]:
            return 0, "config/master.key\napp.rb"
        if cmd[:3] == ["git", "rm", "--cached"]:
            return 0, "rm 'config/master.key'"
        return 0, ""

    return CodeRepo(str(tmp_path), runner=runner), calls


def test_untrack_file_fixes_committed_secret(tmp_path):
    repo, calls = _repo_with_committed_key(tmp_path)
    fix = UntrackFile("config/master.key")
    goal = Goal(name="x", is_satisfied=lambda ctx: False)  # run once via budget

    result = Agent(repo, [fix], gate=AutoApprove(), max_steps=1).pursue(goal)

    step = result.steps[0]
    assert step.outcome.success is True
    assert ["git", "rm", "--cached", "--", "config/master.key"] in calls
    # and it was added to .gitignore
    assert "config/master.key" in (tmp_path / ".gitignore").read_text()


def test_untrack_file_is_held_under_dry_run(tmp_path):
    repo, calls = _repo_with_committed_key(tmp_path)
    fix = UntrackFile("config/master.key")
    goal = Goal(name="x", is_satisfied=lambda ctx: False)

    result = Agent(repo, [fix], gate=DryRun(), max_steps=1).pursue(goal)

    assert result.steps[0].outcome.success is False
    assert not any(c[:3] == ["git", "rm", "--cached"] for c in calls)  # nothing changed
    assert not (tmp_path / ".gitignore").exists()


def test_untrack_not_applicable_when_file_untracked(tmp_path):
    repo = CodeRepo(str(tmp_path), runner=lambda cmd: (0, "app.rb\nGemfile"))
    # the secret isn't tracked, so the fix shouldn't apply
    from tactics import Context
    from tactics.colony.blackboard import Task

    ctx = Context(target=repo, goal=Goal(name="g"), task=Task(id="t", description="", payload={}))
    assert UntrackFile("config/master.key").is_applicable(ctx) is False
