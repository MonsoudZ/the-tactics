"""Tests for the repo-health playbook (fake runner — no real subprocess)."""

from __future__ import annotations

from pathlib import Path

from tactics.playbooks.repo_health import (
    CodeRepo,
    RunLint,
    RunTests,
    ScanSecrets,
    audit_goal,
    build_audit_colony,
)


def _junit_runner(tests, failures=0, errors=0, skipped=0):
    """Fake runner that writes a JUnit XML to the --junitxml path RunTests passes."""

    def runner(cmd):
        path = cmd[-1].split("=", 1)[1]
        Path(path).write_text(
            f'<testsuite tests="{tests}" failures="{failures}" '
            f'errors="{errors}" skipped="{skipped}"></testsuite>',
            encoding="utf-8",
        )
        return (0 if failures == 0 and errors == 0 else 1, "ran")

    return runner


def _ctx_for(target, check):
    # Build a Context the way the colony would, with a task naming the check.
    from tactics.colony.blackboard import Task

    from tactics import Context, Goal

    return Context(target=target, goal=Goal(name="g"), task=Task(id="t", description="", payload={"check": check}))


def test_run_tests_parses_counts():
    repo = CodeRepo(runner=_junit_runner(tests=4))
    out = RunTests().execute(_ctx_for(repo, "tests"))
    assert out.success and out.reward == 1.0
    assert out.metrics["tests"] == 4


def test_run_tests_partial_failure_scores_fraction():
    repo = CodeRepo(runner=_junit_runner(tests=4, failures=1))
    out = RunTests().execute(_ctx_for(repo, "tests"))
    assert out.success is False
    assert out.reward == 0.75  # 3 passed / 4 run


def test_run_lint_missing_tool_is_flagged_not_passed():
    repo = CodeRepo(runner=lambda cmd: (127, "command not found: ruff"))
    out = RunLint().execute(_ctx_for(repo, "lint"))
    assert out.metrics.get("skipped") is True
    assert out.reward == 0.5  # neither pass nor fail — a tooling gap


def test_scan_secrets_flags_a_planted_key(tmp_path):
    # Assemble the fake key by concatenation so THIS test file doesn't trip the
    # scanner when it later audits the tactics repo itself.
    fake = "AKIA" + "ABCDEFGHIJKLMNOP"  # AKIA + 16 chars
    (tmp_path / "config.py").write_text(f'KEY = "{fake}"\n', encoding="utf-8")
    repo = CodeRepo(str(tmp_path), runner=lambda cmd: (0, "config.py"))  # git ls-files -> one file
    out = ScanSecrets().execute(_ctx_for(repo, "secrets"))
    assert out.success is False
    assert out.metrics["count"] == 1
    assert out.metrics["hits"][0]["kind"] == "aws_access_key"
    assert out.metrics["hits"][0]["file"] == "config.py"


def test_scan_secrets_clean_repo_passes(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    repo = CodeRepo(str(tmp_path), runner=lambda cmd: (0, "ok.py"))
    out = ScanSecrets().execute(_ctx_for(repo, "secrets"))
    assert out.success is True
    assert out.metrics["count"] == 0


def test_audit_colony_runs_all_checks(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")

    def runner(cmd):
        if cmd[:1] == ["git"]:
            return 0, "ok.py"
        if "pytest" in cmd:
            Path(cmd[-1].split("=", 1)[1]).write_text(
                '<testsuite tests="5" failures="0" errors="0" skipped="0"></testsuite>',
                encoding="utf-8",
            )
            return 0, "ran"
        if cmd[:1] == ["ruff"]:
            return 0, ""
        return 127, "?"

    target = CodeRepo(str(tmp_path), runner=runner)
    result = build_audit_colony(target, max_workers=1).run(audit_goal())
    kinds = {f.detail.get("task") for f in result.findings if f.kind == "task_done"}
    assert len(kinds) == 3  # tests, lint, secrets all ran and were accepted
