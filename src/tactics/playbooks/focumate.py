"""focumate playbook — production-readiness for a Rails backend + a Swift app.

This is the repo-health pattern pointed at a real product. The same CodeRepo
Target shells out; only the commands change. Two audits:

  * build_rails_audit  — tests, RuboCop, Brakeman (security), bundler-audit
    (vulnerable gems), pending migrations, a secret scan, and a check that
    config/master.key / .env aren't committed.
  * build_swift_audit  — build/test, SwiftLint, and a secret scan.

Every check flags a missing tool instead of silently passing, so a partial
toolchain shows up as gaps to close rather than false green. All commands are
overridable to match focumate's actual setup (rspec vs minitest, SPM vs xcode).

Usage (in a session scoped to the focumate repo):

    from tactics.playbooks.focumate import build_rails_audit, prod_ready_goal
    from tactics.playbooks.repo_health import CodeRepo
    result = build_rails_audit(CodeRepo("/path/to/focumate-rails")).run(prod_ready_goal("rails"))
    print(result.summary())
    for t in result.board.tasks:
        if t.result: print(t.payload["check"], t.result.notes)
"""

from __future__ import annotations

from ..core.goal import Goal
from ..core.tactic import Tactic
from .repo_health import (
    CommandCheck,
    ForbiddenFileCheck,
    ScanSecrets,
    build_audit_colony,
)


def prod_ready_goal(which: str = "app") -> Goal:
    return Goal(name=f"{which}_prod_ready", description=f"{which} production-readiness audit")


# --- Rails -------------------------------------------------------------------


def rails_checks(*, test_cmd: list[str] | None = None) -> list[Tactic]:
    """Standard Rails prod-readiness checks. Override test_cmd for minitest, etc."""
    return [
        CommandCheck("tests", test_cmd or ["bundle", "exec", "rspec"]),
        CommandCheck("lint", ["bundle", "exec", "rubocop", "-f", "quiet"]),
        CommandCheck("security", ["bundle", "exec", "brakeman", "-q", "-w2"]),
        CommandCheck("deps", ["bundle", "exec", "bundler-audit", "check", "--update"]),
        CommandCheck("migrations", ["bundle", "exec", "rails", "db:abort_if_pending_migrations"]),
        ScanSecrets(),
        ForbiddenFileCheck("committed_secrets", ["config/master.key", "config/credentials/*.key", ".env"]),
    ]


def build_rails_audit(target, *, checks: list[Tactic] | None = None, **kw):  # noqa: ANN001
    return build_audit_colony(target, tactics=checks or rails_checks(), **kw)


# --- Swift -------------------------------------------------------------------


def swift_checks(*, test_cmd: list[str] | None = None, lint: bool = True) -> list[Tactic]:
    """Standard Swift prod-readiness checks. Override test_cmd for xcodebuild."""
    checks: list[Tactic] = [CommandCheck("tests", test_cmd or ["swift", "test"])]
    if lint:
        checks.append(CommandCheck("lint", ["swiftlint", "--strict"]))
    checks.append(ScanSecrets())
    return checks


def build_swift_audit(target, *, checks: list[Tactic] | None = None, **kw):  # noqa: ANN001
    return build_audit_colony(target, tactics=checks or swift_checks(), **kw)
