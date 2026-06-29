"""Tests for the focumate Rails/Swift playbook (fake runners — no real tooling)."""

from __future__ import annotations

from tactics import Context, Goal
from tactics.colony.blackboard import Task
from tactics.playbooks.focumate import (
    build_rails_audit,
    build_swift_audit,
    prod_ready_goal,
    rails_checks,
    swift_checks,
)
from tactics.playbooks.repo_health import CodeRepo, CommandCheck, ForbiddenFileCheck


def _ctx(target, check):
    return Context(target=target, goal=Goal(name="g"),
                   task=Task(id="t", description="", payload={"check": check}))


# --- generic building blocks -------------------------------------------------


def test_command_check_pass_fail_and_missing_tool():
    ok = CommandCheck("x", ["tool"]).execute(_ctx(CodeRepo(runner=lambda c: (0, "")), "x"))
    assert ok.success and ok.reward == 1.0
    bad = CommandCheck("x", ["tool"]).execute(_ctx(CodeRepo(runner=lambda c: (1, "boom")), "x"))
    assert bad.success is False and bad.reward == 0.0
    missing = CommandCheck("x", ["tool"]).execute(_ctx(CodeRepo(runner=lambda c: (127, "")), "x"))
    assert missing.metrics.get("skipped") is True  # flagged, not passed


def test_forbidden_file_check_catches_committed_secret():
    repo = CodeRepo(runner=lambda c: (0, "app.rb\nconfig/master.key\n"))
    out = ForbiddenFileCheck("committed_secrets", ["config/master.key", ".env"]).execute(
        _ctx(repo, "committed_secrets")
    )
    assert out.success is False
    assert out.metrics["tracked"] == ["config/master.key"]


def test_forbidden_file_check_clean():
    repo = CodeRepo(runner=lambda c: (0, "app.rb\nGemfile\n"))
    out = ForbiddenFileCheck("committed_secrets", ["config/master.key", ".env"]).execute(
        _ctx(repo, "committed_secrets")
    )
    assert out.success is True


# --- composed audits ---------------------------------------------------------


def test_rails_audit_all_green_when_tools_pass(tmp_path):
    # git ls-files returns nothing -> secret scan + forbidden-file checks pass;
    # every command returns 0.
    target = CodeRepo(str(tmp_path), runner=lambda c: (0, "") if c[:1] == ["git"] else (0, "ok"))
    result = build_rails_audit(target, max_workers=1).run(prod_ready_goal("rails"))
    done = [f for f in result.findings if f.kind == "task_done"]
    assert len(done) == len(rails_checks())  # all 7 checks ran and were accepted


def test_rails_audit_flags_security_and_committed_key(tmp_path):
    def runner(cmd):
        if cmd[:1] == ["git"]:
            return 0, "config/master.key\n"     # committed key -> forbidden hit
        if "brakeman" in cmd:
            return 1, "1 security warning"        # security finding -> fail
        return 0, "ok"

    target = CodeRepo(str(tmp_path), runner=runner)
    result = build_rails_audit(target, max_workers=1).run(prod_ready_goal("rails"))
    by_check = {t.payload["check"]: t.result for t in result.board.tasks if t.result}
    assert by_check["security"].success is False
    assert by_check["committed_secrets"].success is False
    assert by_check["tests"].success is True


def test_swift_audit_runs_expected_checks(tmp_path):
    target = CodeRepo(str(tmp_path), runner=lambda c: (0, "") if c[:1] == ["git"] else (0, "ok"))
    result = build_swift_audit(target, max_workers=1).run(prod_ready_goal("swift"))
    done = [f for f in result.findings if f.kind == "task_done"]
    assert len(done) == len(swift_checks())  # tests, lint, secrets
