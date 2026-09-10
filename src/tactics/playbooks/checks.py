"""Work out how to test a repository — and prove it before anything is spent.

The check command *is* the reward. Everything else in this framework —
worktrees, the gate, the critic, patch re-verification — is downstream of it
being right, and none of it can tell a red test caused by your code from a red
test caused by the machine. That is not theoretical: a live run against a real
Rails app failed one spec because this container's tzdata lacked the legacy
timezone links, and had it been handed to agents they would have "fixed"
production code to accommodate a missing symlink, and the check would have gone
green, and every safeguard would have agreed.

So two jobs here, and the second matters more than the first.

**Propose.** Read the repo and offer the commands its own ecosystem implies, in
the order a person would try them: what the project declares (`npm test`, a
Makefile target) before what its ecosystem defaults to.

**Prove.** Actually run them. A candidate is only usable once it has been seen
to execute, and *executing is not the same as passing*: a suite reporting three
failures is a perfectly good check (that is a repo needing repair), while
`command not found` is not a check at all. Distinguishing those two is the
whole point — the first is a measurement, the second is silence wearing a
measurement's clothes.

    from tactics.playbooks.checks import choose
    check = choose("/path/to/repo")
    print(check.argv, check.why)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

#: Output that says the command itself could not start. These are never checks.
_BROKEN = re.compile(
    r"command not found|No such file or directory|cannot find|"
    r"is not recognized|ModuleNotFoundError|LoadError|"
    r"Could not find .* in locally installed gems|"
    r"missing script|npm ERR! Missing script",
    re.IGNORECASE)


#: Output that says the runner started, found nothing, and measured nothing.
#: Distinct from a failing suite: an agent can satisfy an empty check by adding
#: any passing test at all, which is never the task it was given.
_EMPTY = re.compile(
    r"no tests ran|collected 0 items|no tests were found|no examples found|"
    r"^0 examples, 0 failures|no test files to run",
    re.IGNORECASE | re.MULTILINE)


@dataclass
class Check:
    """One way of testing this repository, and the evidence that suggested it."""

    argv: list[str]
    why: str
    kind: str
    ran: bool | None = None         # None until proven
    detail: str = ""
    seconds: float = 0.0

    @property
    def command(self) -> str:
        return " ".join(self.argv)


@dataclass
class Attempt:
    """What happened when a candidate was tried. Kept so a refusal can explain."""

    check: Check
    code: int
    output: str = field(repr=False, default="")

    @property
    def usable(self) -> bool:
        """Did the command run? Not "did it pass" — a red suite is a good check.

        The burden is on *failure to start*, not on proof of running, and that
        way round matters: the first version of this demanded test-shaped output
        and rejected `test -f built.txt`, which is a perfectly good check that
        fails silently with exit 1 and nothing to say. Plenty of real checks do.

        The OS answers most of it — 127 is "command not found" and 126 is "found
        but not executable", both conventions no test runner uses for a failing
        suite. The patterns then catch the cases that start a process and still
        cannot measure anything: `npm ERR! Missing script`, a Ruby `LoadError`,
        a Python `ModuleNotFoundError`. Everything else ran.
        """
        if _EMPTY.search(self.output) or (self.code == 5 and "pytest" in self.check.command):
            return False        # pytest's 5 is "collected nothing"
        if self.code == 0:
            return True
        if self.code in (126, 127):
            return False
        return not _BROKEN.search(self.output)


# --- proposing ----------------------------------------------------------------


def _has(root: str, *names: str) -> bool:
    return any(os.path.exists(os.path.join(root, n)) for n in names)


def _read(root: str, name: str) -> str:
    try:
        with open(os.path.join(root, name), encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _pythonpath_configured(root: str) -> bool:
    return any("pythonpath" in _read(root, name).lower()
               for name in ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg"))


def propose(root: str) -> list[Check]:
    """Every command this repository plausibly tests with, best guess first.

    Ordered by how much the *project* said, rather than how much we inferred: a
    declared `npm test` or a Makefile `test:` target is the author telling you,
    and beats any default this module knows.
    """
    out: list[Check] = []

    # What the project itself declares.
    package = _read(root, "package.json")
    if package:
        try:
            scripts = json.loads(package).get("scripts") or {}
        except json.JSONDecodeError:
            scripts = {}
        if "test" in scripts:
            out.append(Check(["npm", "test", "--silent"],
                             f"package.json declares a test script: {scripts['test']!r}",
                             "npm"))
    makefile = _read(root, "Makefile") or _read(root, "makefile")
    if re.search(r"^test\s*:", makefile, re.MULTILINE):
        out.append(Check(["make", "test"], "the Makefile has a `test` target", "make"))

    # Ruby. `bundle exec rspec` needs a binstub the gem home may not have; the
    # runner can always be invoked through Ruby itself, which needs only the gem
    # — that fallback is what makes this work on a container where `bundle exec
    # rspec` reports "command not found" while rspec-core is plainly installed.
    if _has(root, "Gemfile"):
        if os.path.isdir(os.path.join(root, "spec")):
            out.append(Check(["bin/rspec"], "bin/rspec binstub with a spec/ directory", "rspec")
                       if _has(root, "bin/rspec") else
                       Check(["bundle", "exec", "rspec"], "Gemfile with a spec/ directory", "rspec"))
            out.append(Check(
                ["bundle", "exec", "ruby", "-e",
                 'require "rspec/core"; exit RSpec::Core::Runner.run(["spec"])'],
                "rspec-core through Ruby, for a gem home with no binstubs", "rspec"))
            # The path is explicit on purpose. `Runner.run([])` exits 0 having
            # run nothing — the binary defaults to spec/, the API does not — and
            # a check that measures zero examples while reporting success is the
            # most dangerous thing this module could hand anyone. It was
            # declared "proven" once before the empty-check rule caught it.
        if os.path.isdir(os.path.join(root, "test")):
            out.append(Check(["bin/rails", "test"], "Gemfile with a test/ directory", "rails"))

    # Python. `python3 -m pytest` rather than `pytest` for the same reason as
    # the Ruby fallback: a module needs no executable on PATH.
    if _has(root, "pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini") or \
            os.path.isdir(os.path.join(root, "tests")):
        if os.path.isdir(os.path.join(root, "src")) and not _pythonpath_configured(root):
            # The trap: inside a worktree an editable install resolves to the
            # *main* tree, so the check silently measures code the agent never
            # touched. `env` because the check is argv, run without a shell.
            out.append(Check(["env", "PYTHONPATH=src", "python3", "-m", "pytest", "-q"],
                             "src/ layout with no pythonpath configured — without this "
                             "the check measures the main tree, not the worktree",
                             "pytest"))
        out.append(Check(["python3", "-m", "pytest", "-q"], "a Python project with tests",
                         "pytest"))

    if _has(root, "go.mod"):
        out.append(Check(["go", "test", "./..."], "go.mod", "go"))
    if _has(root, "Cargo.toml"):
        out.append(Check(["cargo", "test"], "Cargo.toml", "cargo"))
    if _has(root, "pom.xml"):
        out.append(Check(["mvn", "-q", "test"], "pom.xml", "maven"))
    if _has(root, "build.gradle", "build.gradle.kts"):
        out.append(Check(["gradle", "test"], "a Gradle build file", "gradle"))

    return [c for c in out if shutil.which(c.argv[0]) or c.argv[0].startswith("bin/")]


# --- proving ------------------------------------------------------------------


def attempt(root: str, check: Check, *, timeout: int = 900) -> Attempt:
    """Run a candidate once and see whether it is a check at all."""
    import time

    started = time.time()
    try:
        done = subprocess.run(check.argv, cwd=root, capture_output=True,
                              text=True, timeout=timeout)
        code, output = done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired:
        code, output = 1, f"timed out after {timeout}s"
    except OSError as exc:
        code, output = 127, f"command not found: {exc}"
    check.seconds = round(time.time() - started, 1)
    result = Attempt(check=check, code=code, output=output)
    check.ran = result.usable
    check.detail = ("passed" if code == 0 else
                    "ran, and the suite is red" if result.usable else
                    "ran, but collected no tests — nothing here to measure"
                    if _EMPTY.search(output) or code == 5 else
                    output.strip().splitlines()[-1][:160] if output.strip() else "no output")
    return result


def choose(root: str, *, timeout: int = 900) -> tuple[Check | None, list[Attempt]]:
    """The first proposal that actually runs, and everything tried on the way.

    Returns ``(None, attempts)`` when nothing works, which is a *result* and not
    an error: refusing to start beats running a swarm against a check that
    cannot measure anything. The attempts carry why each candidate failed, so
    the refusal can say what to fix.
    """
    attempts: list[Attempt] = []
    for candidate in propose(root):
        tried = attempt(root, candidate, timeout=timeout)
        attempts.append(tried)
        if tried.usable:
            return candidate, attempts
    return None, attempts
