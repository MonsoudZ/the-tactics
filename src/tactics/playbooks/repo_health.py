"""Repo-health playbook — audit a code repository, safely.

The brain points at a repo and runs checks as a verifying swarm: tests, lint, and
a secret scan (the one that would have caught a hardcoded API key). Each check is
a Tactic that reaches the repo through the Target — by shelling out — which is
exactly how this same playbook will later drive a Rails app (`rspec`), a Swift
app (`xcodebuild`), or anything else: only the commands change.

v1 is read-only (audit + report). Fix tactics that *change* code come next; when
they do, they propose through the approval gate so nothing is written without a
pass — the safety model is already in place.

    from tactics.playbooks.repo_health import CodeRepo, build_audit_colony, audit_goal
    colony = build_audit_colony(CodeRepo("."))
    result = colony.run(audit_goal())
    print(result.summary())
    for f in result.findings: print(f.kind, f.detail)
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Callable

from ..colony import AcceptCritic, Colony, FunctionPlanner
from ..core.goal import Goal
from ..core.memory import InMemoryStore, MemoryStore
from ..core.outcome import Outcome
from ..core.tactic import Tactic
from ..core.target import Target

# Specific patterns — tuned to avoid false positives on test fixtures.
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("slack_token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}")),
]

_SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".ico", ".lock")


class CodeRepo(Target):
    """A code repository the brain can inspect. ``run`` is the seam to the shell
    (injectable for tests). Add domain methods as you add checks."""

    name = "code_repo"

    def __init__(self, path: str = ".", *, runner: Callable[[list[str]], tuple[int, str]] | None = None):
        self.path = path
        self._run = runner or self._subprocess

    def _subprocess(self, cmd: list[str]) -> tuple[int, str]:
        try:
            p = subprocess.run(
                cmd, cwd=self.path, capture_output=True, text=True, timeout=600
            )
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except FileNotFoundError:
            return 127, f"command not found: {cmd[0]}"
        except Exception as exc:  # noqa: BLE001 - report, never raise into the loop
            return 1, f"error running {cmd}: {exc!r}"

    def run(self, cmd: list[str]) -> tuple[int, str]:
        return self._run(cmd)

    def observe(self) -> dict:
        return {"path": self.path}

    def tracked_files(self) -> list[str]:
        code, out = self.run(["git", "ls-files"])
        return [ln.strip() for ln in out.splitlines() if ln.strip()] if code == 0 else []

    def read(self, relpath: str) -> str:
        try:
            with open(os.path.join(self.path, relpath), encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""


class _Check(Tactic):
    check = ""

    def is_applicable(self, ctx) -> bool:  # noqa: ANN001
        return ctx.task is not None and ctx.task.payload.get("check") == self.check


def _parse_junit(path: str) -> dict | None:
    """Read pytest's JUnit XML into aggregate counts. None if unreadable.

    Machine-readable, unlike the human summary line — which some environments
    suppress under output capture, so we never depend on it.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    agg = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for s in suites:
        for k in agg:
            agg[k] += int(s.get(k, 0) or 0)
    return agg


class RunTests(_Check):
    check = "tests"

    def __init__(self, cmd: list[str] | None = None):
        super().__init__(name="run_tests")
        self.cmd = cmd or ["python3", "-m", "pytest", "-q"]

    def execute(self, ctx) -> Outcome:  # noqa: ANN001
        fd, xml_path = tempfile.mkstemp(suffix=".xml", prefix="tactics-junit-")
        os.close(fd)
        try:
            code, _out = ctx.target.run(self.cmd + [f"--junitxml={xml_path}"])
            stats = _parse_junit(xml_path)
        finally:
            try:
                os.remove(xml_path)
            except OSError:
                pass

        if code == 127:
            return Outcome(success=False, reward=0.5, metrics={"skipped": True},
                           notes="pytest not installed (skipped)")
        if stats is None:
            ok = code == 0
            return Outcome(success=ok, reward=1.0 if ok else 0.0, metrics={},
                           notes="tests ran (counts unavailable)")

        run = stats["tests"] - stats["skipped"]
        bad = stats["failures"] + stats["errors"]
        passed = run - bad
        reward = 1.0 if (bad == 0 and run > 0) else (passed / run if run else 0.0)
        if ctx.journal:
            ctx.journal.record("check.tests", **stats)
        return Outcome(
            success=(code == 0 and bad == 0),
            reward=round(reward, 3),
            metrics=stats,
            notes=(
                f"{passed} passed, {stats['failures']} failed, "
                f"{stats['errors']} errors, {stats['skipped']} skipped"
            ),
        )


class RunLint(_Check):
    check = "lint"

    def __init__(self, cmd: list[str] | None = None):
        super().__init__(name="run_lint")
        self.cmd = cmd or ["ruff", "check", "."]

    def execute(self, ctx) -> Outcome:  # noqa: ANN001
        code, out = ctx.target.run(self.cmd)
        if code == 127:  # tool not installed — a tooling gap, not a pass
            return Outcome(success=False, reward=0.5, metrics={"skipped": True},
                           notes="linter not installed (skipped)")
        clean = code == 0
        return Outcome(success=clean, reward=1.0 if clean else 0.0,
                       metrics={"clean": clean}, notes="clean" if clean else out.strip()[-300:])


class ScanSecrets(_Check):
    check = "secrets"

    def __init__(self):
        super().__init__(name="scan_secrets")

    def execute(self, ctx) -> Outcome:  # noqa: ANN001
        hits = []
        for rel in ctx.target.tracked_files():
            if rel.endswith(_SKIP_EXT):
                continue
            text = ctx.target.read(rel)
            for kind, rx in _SECRET_PATTERNS:
                for m in rx.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append({"file": rel, "line": line, "kind": kind})
        ok = not hits
        if ctx.journal:
            ctx.journal.record("check.secrets", hits=len(hits))
        return Outcome(
            success=ok,
            reward=1.0 if ok else 0.0,
            metrics={"hits": hits, "count": len(hits)},
            notes="no secrets found" if ok else f"{len(hits)} potential secret(s) committed!",
        )


class CommandCheck(_Check):
    """Generic check: run a command, pass = exit 0. Flags a missing tool rather
    than passing it. The reusable building block for Rails/Swift/etc. checks."""

    def __init__(self, check: str, cmd: list[str], *, name: str | None = None):
        super().__init__(name=name or f"run_{check}")
        self.check = check
        self.cmd = list(cmd)

    def execute(self, ctx) -> Outcome:  # noqa: ANN001
        code, out = ctx.target.run(self.cmd)
        if code == 127:
            return Outcome(success=False, reward=0.5,
                           metrics={"skipped": True, "tool": self.cmd[0]},
                           notes=f"{self.cmd[0]} not installed (skipped)")
        ok = code == 0
        return Outcome(success=ok, reward=1.0 if ok else 0.0, metrics={"exit": code},
                       notes="passed" if ok else (out.strip()[-400:] or f"exit {code}"))


class ForbiddenFileCheck(_Check):
    """Fail if a sensitive file is tracked in git (e.g. Rails config/master.key,
    a committed .env). Catches the 'secret committed' class before it ships."""

    def __init__(self, check: str, patterns: list[str], *, name: str | None = None):
        super().__init__(name=name or f"check_{check}")
        self.check = check
        self.patterns = list(patterns)

    def execute(self, ctx) -> Outcome:  # noqa: ANN001
        import fnmatch

        tracked = ctx.target.tracked_files()
        hits = sorted({f for f in tracked for p in self.patterns if fnmatch.fnmatch(f, p)})
        ok = not hits
        return Outcome(success=ok, reward=1.0 if ok else 0.0, metrics={"tracked": hits},
                       notes="none tracked" if ok else f"sensitive file(s) committed: {hits}")


def audit_goal() -> Goal:
    return Goal(name="repo_health", description="Audit the repository for health and safety")


def build_audit_colony(
    target: CodeRepo,
    *,
    tactics: list[Tactic] | None = None,
    memory: MemoryStore | None = None,
    max_workers: int = 4,
    max_rounds: int | None = None,
) -> Colony:
    tactics = tactics or [RunTests(), RunLint(), ScanSecrets()]
    checks = [t.check for t in tactics]  # type: ignore[attr-defined]
    # Enough rounds to run every check even at max_workers=1 (checks are one-shot).
    if max_rounds is None:
        max_rounds = len(tactics) + 2

    def plan(goal, board, target):  # noqa: ANN001
        if board.tasks:
            return []
        return [board.post_task(f"check:{c}", payload={"check": c}) for c in checks]

    return Colony(
        target,
        tactics,
        FunctionPlanner(plan),
        memory=memory or InMemoryStore(),
        critic=AcceptCritic(),
        max_workers=max_workers,
        max_rounds=max_rounds,
    )
