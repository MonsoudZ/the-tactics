"""`python -m tactics <repo> "<task>"` — point the brain at any repository.

Everything this does was already possible; it was just thirty lines of wiring you
had to get right, and the easiest one to get wrong (the check command) is the one
that decides the reward. So this is a front door, not a new capability.

    python -m tactics ~/code/api "add retry with backoff to the HTTP client" \\
        --check "pytest -q" --agents 3 --budget 5.00

What it does: cuts a git worktree per agent, sends each a different brief, runs
them in parallel, verifies each one *in its own tree* by running your check,
prints the scoreboard and the candidate patches, and leaves your repository
untouched. Landing a patch is a separate, deliberate ``--apply``.

Three postures, and the default is the careful one:

  * default — agents edit and run commands freely inside their own worktrees;
    anything irreversible (a push, an ``rm -rf``, a tool nobody classified) is
    escalated, which means a prompt if you are at a terminal and a refusal if
    you are not;
  * ``--yes`` — approve everything, still inside the worktrees;
  * ``--dry-run`` — the agents may read and reason but cannot write a byte, and
    the journal records the change they *wanted* to make.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import tempfile
import time
from typing import Any

from . import __version__
from .core.approval import AutoApprove, DryRun, PolicyGate
from .core.budget import Budget
from .playbooks.agent_sdk import (
    AgentWorkspace,
    build_delivery_colony,
    land_best_patch,
    run_and_learn,
    work_queue_goal,
)

# Edits and ordinary commands flow; a push, a wipe or an unclassified tool stops.
_ROUTINE = ("low", "medium")


def _git(repo: str, *args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return 1, repr(exc)


def _check_repo(repo: str) -> str:
    """Why this repo cannot be worked on, or "" if it can."""
    if not os.path.isdir(repo):
        return f"{repo} is not a directory"
    if _git(repo, "rev-parse", "--git-dir")[0] != 0:
        return f"{repo} is not a git repository — worktree isolation needs one"
    if _git(repo, "rev-parse", "HEAD")[0] != 0:
        return f"{repo} has no commits yet; agents start from a checkout of HEAD"
    return ""


def _warnings(repo: str, check: list[str]) -> list[str]:
    """Things that quietly measure the wrong thing. Cheap to check, costly to miss."""
    out = []
    dirty = _git(repo, "status", "--porcelain")[1].strip()
    if dirty:
        out.append(f"{len(dirty.splitlines())} uncommitted change(s) — worktrees are cut "
                   "from HEAD, so the agents will not see them")
    if not os.path.exists(os.path.join(repo, ".gitignore")):
        out.append("no .gitignore — anything the agent or your check command builds "
                   "(__pycache__, coverage data, compiled output) lands in the patch "
                   "alongside the real change, and can stop it applying")
    if os.path.isdir(os.path.join(repo, "src")) and check and "pytest" in " ".join(check):
        configured = any(
            "pythonpath" in _read(os.path.join(repo, name)).lower()
            for name in ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg")
        )
        editable = "PYTHONPATH" in " ".join(check)
        if not configured and not editable:
            out.append("this looks like a src/ layout with no pythonpath configured — the "
                       "check runs *inside* each worktree, so an editable install would "
                       "silently measure your main tree instead. Consider "
                       "--check 'env PYTHONPATH=src python3 -m pytest -q'")
    return out


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


_PROMPT = threading.Lock()


def _describe(proposal) -> str:  # noqa: ANN001
    """What is actually being asked for — "tool call: Bash" is not a question."""
    detail = getattr(proposal, "detail", None) or {}
    what = detail.get("command") or detail.get("file_path") or ""
    return f"{proposal.action}{f'  {what}' if what else ''}"


def _ask(proposal, ctx) -> bool:  # noqa: ANN001
    """Escalation for the default posture: prompt at a terminal, refuse otherwise.

    Serialized, because this is reached from each ant's own thread. With
    ``--agents`` above one and no lock, several agents read the same stdin at
    once: the prompts interleave into one unreadable line and a "y" meant for
    one of them is delivered to whichever thread happens to be reading — an
    approval given for the wrong action, which is the one outcome a gate must
    never produce. One question at a time, and the others wait.
    """
    if not sys.stdin.isatty():
        print(f"  refused (no terminal to ask): {_describe(proposal)}", file=sys.stderr)
        return False
    with _PROMPT:
        try:
            answer = input(f"  approve? {_describe(proposal)}  [y/N] ")
        except EOFError:        # stdin closed under us — refuse, do not guess
            print("  refused (stdin closed)", file=sys.stderr)
            return False
    return answer.strip().lower() in ("y", "yes")


def _scribe_client() -> tuple[Any, str]:
    """The client the scribe distils with, or None and the reason there isn't one.

    The SDK path first, and only then the API. The agents this CLI runs need no
    API key — they authenticate through the Claude Code CLI — so demanding one
    for the scribe left the two halves of memory on different credentials, and
    the verbal half unavailable in exactly the environments where the execution
    half worked. Same SDK, same auth, no key, nothing billed to the API.
    """
    try:
        from .llm.client import ClaudeClient, SdkClient
    except ImportError as exc:
        return None, f"tactics[llm] not installed ({exc})"

    try:
        import claude_agent_sdk  # noqa: F401, PLC0415 - presence check only
    except ImportError:
        pass
    else:
        return SdkClient(), ""

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None, ("no claude-agent-sdk and no ANTHROPIC_API_KEY — "
                      "install with: pip install 'tactics[agent-sdk]'")
    try:
        return ClaudeClient(), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"could not build a client ({exc!r})"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tactics",
        description="Point the brain at a repository: competing briefs, verified reward, "
                    "gated writes, and memory that compounds across runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Your repository is never written to unless you pass --apply.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("repo", help="path to the git repository to work on")
    p.add_argument("task", nargs="?", default=None,
                   help="what you want done, in a sentence (omit with --report or --fix)")
    p.add_argument("--check", default="pytest -q",
                   help="the command that decides success. This *is* the reward — a wrong "
                        "one measures the wrong thing (default: %(default)s)")
    p.add_argument("--agents", type=int, default=1, metavar="N",
                   help="how many agents to run in parallel, each in its own worktree")
    p.add_argument("--rounds", type=int, default=1, help="attempts per agent (default: 1)")
    p.add_argument("--budget", type=float, default=5.00, metavar="USD",
                   help="hard ceiling on spend (default: $%(default).2f)")
    p.add_argument("--max-turns", type=int, default=None, metavar="N",
                   help="cap each agent's turns; fewer turns means less exploring")
    p.add_argument("--model", default=None, help="model for the agents (default: the SDK's)")
    posture = p.add_mutually_exclusive_group()
    posture.add_argument("--yes", action="store_true",
                         help="approve every tool call (still inside the worktrees)")
    posture.add_argument("--dry-run", action="store_true",
                         help="propose only — the agents cannot write a byte")
    p.add_argument("--show-diff", action="store_true",
                   help="print each candidate patch in full — worktrees are destroyed "
                        "when the run ends, so this is the only copy")
    p.add_argument("--apply", action="store_true",
                   help="land the best verified patch on your repository")
    p.add_argument("--no-persist", action="store_true",
                   help="do not read or write <repo>/.tactics (memory and lessons)")
    p.add_argument("--no-learn", action="store_true",
                   help="skip the scribe; run without distilling lessons afterwards")
    p.add_argument("--patch-dir", default=None, metavar="DIR",
                   help="where to archive each patch as it is produced "
                        "(default: <repo>/.tactics/patches/<run>); a directory "
                        "you name here is never pruned")
    p.add_argument("--keep-runs", type=int, default=20, metavar="N",
                   help="how many runs of patch archive to keep (default: 20)")
    p.add_argument("--report", action="store_true",
                   help="read the repo and print what it does, what is missing, "
                        "and what could go — then stop")
    p.add_argument("--fix", type=int, default=None, metavar="N",
                   help="do item N from the report's actionable list, checked against "
                        "the report's own measurement rather than the agent's word")
    return p


def _prepare_tactics_dir(repo: str) -> None:
    """Make `.tactics/` ignore the part of itself nobody should ever commit.

    Two different things live in there. Patches are run artifacts — one per
    agent, per run, forever — and committing them is never right. Memory and
    lessons are what past runs on this repository learned, and a team may very
    well want those in git so everyone's agents start informed. So the archive
    is ignored from inside the directory, which needs no change to a .gitignore
    the user maintains, and the interesting choice is left to them.
    """
    marker = os.path.join(repo, ".tactics", ".gitignore")
    if os.path.exists(marker):
        return
    try:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("# Written by tactics.\n"
                     "# Patches are run artifacts and are never worth committing.\n"
                     "# The memory and lessons beside them may well be: they are what\n"
                     "# past runs on this repository learned.\n"
                     "patches/\n")
    except OSError:         # a read-only repo is the user's business, not a crash
        pass


def _tactics_dir_status(repo: str) -> str:
    """Whether the user has decided about `.tactics/` yet: ignored, tracked, loose."""
    def git(*args: str):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)

    try:
        if git("check-ignore", "-q", ".tactics").returncode == 0:
            return "ignored"
        if git("ls-files", "--", ".tactics").stdout.strip():
            return "tracked"
    except OSError:         # pragma: no cover - no git on PATH is caught earlier
        return "ignored"
    return "loose"


def _diagnose(errors: list[str], args) -> str:  # noqa: ANN001
    """Turn the agent's error into the thing to actually do about it, if we can."""
    joined = " ".join(errors).lower()
    if "not installed" in joined or "no module named" in joined:
        return "pip install 'tactics[agent-sdk]'"
    if ("api key" in joined) or "unauthorized" in joined or "401" in joined:
        return "authenticate the Claude CLI, or set ANTHROPIC_API_KEY"
    if "maximum number of turns" in joined:
        limit = f" (currently {args.max_turns})" if args.max_turns else ""
        return f"raise --max-turns{limit}, or give the agents a narrower task"
    return ""


def _from_report(repo: str, args) -> tuple[int | None, Any]:  # noqa: ANN001
    """Print the report, or pick the job the user asked for out of it.

    Returns (exit code, job). An exit code means stop; a job means carry on and
    let the colony work it with the report's own measurement as half the reward.
    """
    from .playbooks.report import inventory, render, work

    print(f"reading {repo} …", file=sys.stderr)
    report = inventory(repo)
    jobs = work(report)

    if args.report:
        print(render(report))
        print("\n## What can be done about it\n")
        if not jobs:
            print("Nothing here has a mechanical definition of done.")
        else:
            print("Each of these is checkable, so `--fix N` will do it and measure "
                  "the result rather than take the agent's word.\n")
            for i, job in enumerate(jobs):
                print(f"{i:>3}. [{job.kind}] {job.metric} — {job.before:g} → {job.target:g}")
        return 0, None

    if not 0 <= args.fix < len(jobs):
        print(f"tactics: --fix {args.fix} is out of range; the report has "
              f"{len(jobs)} actionable item(s). Run --report to see them.", file=sys.stderr)
        return 2, None

    job = jobs[args.fix]
    args.task = job.description
    print(f"from the report: [{job.kind}] {job.metric}  {job.before:g} → {job.target:g}\n")
    return None, job


def _catch_terminate():
    """Make a SIGTERM raise, so the cleanup below it still runs.

    Ctrl-C already unwinds (KeyboardInterrupt is an exception), but the default
    SIGTERM disposition is to die on the spot — which is how `timeout`, a CI
    cancellation, or a plain `kill` leave a worktree per agent behind. Turning
    it into SystemExit costs nothing and covers everything except SIGKILL,
    which nothing can catch and the reaper handles on the next run instead.
    """
    def terminate(_signum, _frame):
        raise SystemExit(143)        # 128 + SIGTERM

    try:
        return signal.signal(signal.SIGTERM, terminate)
    except ValueError:               # not the main thread; nothing to install
        return None


def _restore_terminate(previous) -> None:  # noqa: ANN001
    if previous is not None:
        try:
            signal.signal(signal.SIGTERM, previous)
        except ValueError:           # pragma: no cover - not the main thread
            pass


def _patch_dir(args, repo: str) -> str:  # noqa: ANN001
    """A directory of its own per run, so one run never overwrites another's.

    Each run's candidates are separate answers, sometimes to a different task
    entirely; flattening them into one folder would silently lose the earlier
    set. Under ``--no-persist`` the repository stays untouched, so the archive
    goes to a temporary directory instead of being thrown away — the flag is
    about not writing your repo, not about discarding the work.
    """
    if args.patch_dir:
        return os.path.abspath(os.path.expanduser(args.patch_dir))
    stamp = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    if args.no_persist:
        return os.path.join(tempfile.gettempdir(), "tactics-patches", stamp)
    return os.path.join(repo, ".tactics", "patches", stamp)


def _prune_patch_archives(root: str, keep: int) -> list[str]:
    """Drop all but the ``keep`` most recent run directories under ``root``.

    One directory per run, forever, is a slow leak into the repository. Only the
    archive the CLI manages is pruned, and the run names are UTC stamps so the
    order is the name order. "Keep everything" is spelled by naming your own
    ``--patch-dir``: a directory you chose is never deleted from.
    """
    try:
        runs = sorted(entry.path for entry in os.scandir(root) if entry.is_dir())
    except OSError:                 # no archive yet, or not ours to read
        return []
    doomed = runs[:-keep]
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True)
    return [p for p in doomed if not os.path.exists(p)]


def main(argv: list[str] | None = None, *, runner=None) -> int:  # noqa: ANN001
    args = build_parser().parse_args(argv)
    repo = os.path.abspath(os.path.expanduser(args.repo))

    problem = _check_repo(repo)
    if problem:
        print(f"tactics: {problem}", file=sys.stderr)
        return 2
    if args.keep_runs < 1:
        print("tactics: --keep-runs must be at least 1 — the run about to start "
              "needs somewhere to put its patches", file=sys.stderr)
        return 2

    if args.report or args.fix is not None:
        code, job = _from_report(repo, args)
        if code is not None:
            return code
    elif not args.task:
        print("tactics: a task is required (or use --report / --fix N)", file=sys.stderr)
        return 2
    else:
        job = None

    check = shlex.split(args.check)
    if args.agents > 1 and args.dry_run:
        print("tactics: note — a dry run writes nothing, so parallel agents only "
              "produce parallel proposals", file=sys.stderr)

    gate = (DryRun() if args.dry_run
            else AutoApprove() if args.yes
            else PolicyGate(escalate=_ask, allow_risk=_ROUTINE))

    patch_dir = _patch_dir(args, repo)
    pruned = ([] if args.patch_dir
              else _prune_patch_archives(os.path.dirname(patch_dir), args.keep_runs))

    workspace = AgentWorkspace(repo, check=check, isolate=True, runner=runner,
                               model=args.model, patch_dir=patch_dir,
                               gauge=(lambda tree, j=job: j.ok(tree)) if job else None)
    reaped = workspace.reap_abandoned_worktrees()
    if not args.no_persist:
        _prepare_tactics_dir(repo)
    colony = build_delivery_colony(
        workspace,
        gate=gate,
        budget=Budget(max_cost=args.budget),
        max_workers=args.agents,
        max_rounds=args.rounds,
        persist=not args.no_persist,
    )
    if args.max_turns:
        import dataclasses
        for tactic in colony.tactics:
            tactic.spec = dataclasses.replace(tactic.spec, max_turns=args.max_turns)

    # What does a passing check prove? On a repo whose check is already green,
    # only that nothing broke — the agent then writes the test that grades its
    # own work. Worth one run of the check to know which situation this is,
    # rather than reporting a green check as though it always meant the same
    # thing.
    baseline_green = workspace.verify()[0] if not args.dry_run else None

    print(f"repo    {repo}")
    print(f"task    {args.task}")
    print(f"check   {' '.join(check)}   <- this decides the reward")
    print(f"posture {type(gate).__name__}   agents {args.agents}   budget ${args.budget:.2f}")
    if baseline_green is True:
        print("        your check already passes, so a pass after the run only means "
              "nothing broke —\n        each candidate's own tests are re-run without "
              "its code to see if they prove anything")
    elif baseline_green is False:
        print("        your check currently fails, so a pass after the run is real "
              "evidence of repair")
    for warning in _warnings(repo, check):
        print(f"warning {warning}", file=sys.stderr)
    if reaped:
        print(f"cleaned up {len(reaped)} worktree(s) abandoned by an earlier run")
    if pruned:
        # Said out loud rather than done quietly: these are files the last report
        # named, and the retention is only discoverable if it announces itself.
        print(f"pruned {len(pruned)} patch archive(s), keeping the last {args.keep_runs}")
    print()

    client, why = (None, "disabled") if args.no_learn else _scribe_client()
    previous = _catch_terminate()
    try:
        result, lessons = run_and_learn(colony, work_queue_goal(args.task), client=client)
    finally:
        workspace.cleanup()
        _restore_terminate(previous)

    return _report(args, workspace, result, lessons, client, why, gate,
                   baseline_green=baseline_green)


def _report(args, workspace, result, lessons, client, why, gate,
            *, baseline_green=None) -> int:  # noqa: ANN001
    runs = [e for e in result.journal.events if e.kind == "agent.run"]
    spent = sum(e.data.get("cost_usd", 0.0) for e in runs)
    held = sum(e.kind in ("gate.hold", "brief.deny") for e in result.journal.events)

    print(f"{len(runs)} run(s), ${spent:.2f} spent, {held} tool call(s) held by the gate")
    for e in runs:
        print(f"  {e.data['tactic']:<18} ${e.data.get('cost_usd', 0):<7} "
              f"{e.data.get('changed_files', 0)} file(s)")
        # A run that never started looked exactly like a run that found nothing
        # to do: same shape, $0.00, no patch. The error was in the journal all
        # along and simply never printed.
        if e.data.get("error"):
            print(f"      failed: {e.data['error']}")

    patches = workspace.patches
    failures = [e.data["error"] for e in runs if e.data.get("error")]
    all_failed = bool(runs) and len(failures) == len(runs)
    # Spend is the line between the two. A run that cost money reached the
    # model, so whatever went wrong is an answer about this task — running out
    # of turns is the ordinary case. A run that cost nothing never got that far,
    # and calling that a result would be reporting a verdict nobody reached.
    never_started = all_failed and spent == 0.0
    if all_failed:
        # Found dogfooding: all three agents hit the turn limit and all three
        # had already written a patch the check then verified. Calling that a
        # failed run contradicts the verdicts printed directly underneath it —
        # an error is how the run *ended*, not a verdict on what it produced.
        if never_started:
            print("\nevery agent failed before doing any work — this is a setup "
                  "problem, not a result.")
        elif patches:
            print("\nevery agent errored before finishing, but the work they had "
                  "already done was captured and verified below.")
        else:
            print("\nevery agent failed part-way through its run.")
        hint = _diagnose(failures, args)
        if hint:
            print(f"  → {hint}")

    # Whether the check passed is the entire point, and listing candidates
    # without it reads as though every patch works. The critic re-ran the check
    # for each ant; say what it found.
    verdicts = [e for e in result.journal.events if e.kind == "verify"]
    if verdicts:
        print("\nverification (your check, re-run against each agent's own tree):")
        for e in verdicts:
            print(f"  {e.data.get('tactic', '?'):<18} {e.data.get('reason', '')}")

    print(f"\n{len(patches)} candidate patch(es); your repository is untouched")
    for patch in patches:
        print(f"  from {patch.tactic or patch.task:<18} "
              f"{len(patch.text.splitlines())} diff lines  {patch.files}")

    # Where they went. Each was written before its worktree was destroyed, so
    # this holds even for the run that never got as far as printing a report.
    saved = [p for p in patches if p.saved_to]
    if saved:
        print(f"\nsaved to {os.path.dirname(saved[0].saved_to)}")
        for patch in saved:
            print(f"  {os.path.basename(patch.saved_to)}"
                  f"   git apply --3way {shlex.quote(patch.saved_to)}")
    elif patches:
        print(f"\nwarning: could not write to {workspace.patch_dir}; "
              "the patches below are the only copy", file=sys.stderr)

    if baseline_green and patches:
        # The check was green before, so it cannot distinguish a real change
        # from a no-op. Ask each patch's tests to prove themselves instead.
        print("\ndo these patches' own tests prove anything? (their tests, applied "
              "without their code)")
        for patch in patches:
            proof = workspace.proves_itself(patch)
            name = patch.tactic or patch.task
            if proof.ok:
                print(f"  {name:<18} yes — its tests fail without its code")
            else:
                print(f"  {name:<18} NO — {proof.detail or 'its tests pass regardless'}")

    if args.show_diff and patches:
        # Printed in full and not truncated: the worktree that produced this is
        # already gone, so what is on screen is the only copy there is.
        for patch in patches:
            title = f" patch from {patch.tactic or patch.task} "
            print(f"\n{title:-^72}")
            print(patch.text.rstrip() or "(empty)")
        print("-" * 72)

    if client is None and not args.no_learn:
        print(f"\nno lessons distilled: {why}")
    elif lessons:
        print(f"\n{len(lessons)} lesson(s) written to .tactics/ for next time:")
        for lesson in lessons:
            print(f"  • {lesson.text}")

    if never_started:
        return 2                      # nothing ran; distinct from "ran, found nothing"

    if not args.no_persist and _tactics_dir_status(workspace.path) == "loose":
        print("\nnote: .tactics/ holds this repo's memory and lessons (its patch "
              "archive is\n      already ignored). Commit it to share what runs "
              "learn, or add\n      .tactics/ to .gitignore to keep it to yourself.")

    if not patches:
        print("\nnothing to apply.")
        return 1

    if args.apply:
        outcome = land_best_patch(workspace, gate=AutoApprove())
        print(f"\napply: {outcome.notes}")
        return 0 if outcome.success else 1

    print("\nre-run with --apply to land the best verified patch.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
