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
import subprocess
import sys
from typing import Any

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
                       "--check 'PYTHONPATH=$PWD/src pytest -q'")
    return out


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _ask(proposal, ctx) -> bool:  # noqa: ANN001
    """Escalation for the default posture: prompt at a terminal, refuse otherwise."""
    if not sys.stdin.isatty():
        print(f"  refused (no terminal to ask): {proposal.action}", file=sys.stderr)
        return False
    answer = input(f"  approve? {proposal.action}  [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _scribe_client() -> tuple[Any, str]:
    """The client the scribe distils with, or None and the reason there isn't one."""
    try:
        from .llm.client import ClaudeClient
    except ImportError as exc:
        return None, f"tactics[llm] not installed ({exc})"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None, "no ANTHROPIC_API_KEY in the environment"
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
    p.add_argument("repo", help="path to the git repository to work on")
    p.add_argument("task", help="what you want done, in a sentence")
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
    return p


def main(argv: list[str] | None = None, *, runner=None) -> int:  # noqa: ANN001
    args = build_parser().parse_args(argv)
    repo = os.path.abspath(os.path.expanduser(args.repo))

    problem = _check_repo(repo)
    if problem:
        print(f"tactics: {problem}", file=sys.stderr)
        return 2

    check = shlex.split(args.check)
    if args.agents > 1 and args.dry_run:
        print("tactics: note — a dry run writes nothing, so parallel agents only "
              "produce parallel proposals", file=sys.stderr)

    gate = (DryRun() if args.dry_run
            else AutoApprove() if args.yes
            else PolicyGate(escalate=_ask, allow_risk=_ROUTINE))

    workspace = AgentWorkspace(repo, check=check, isolate=True,
                               runner=runner, model=args.model)
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

    print(f"repo    {repo}")
    print(f"task    {args.task}")
    print(f"check   {' '.join(check)}   <- this decides the reward")
    print(f"posture {type(gate).__name__}   agents {args.agents}   budget ${args.budget:.2f}")
    for warning in _warnings(repo, check):
        print(f"warning {warning}", file=sys.stderr)
    print()

    client, why = (None, "disabled") if args.no_learn else _scribe_client()
    try:
        result, lessons = run_and_learn(colony, work_queue_goal(args.task), client=client)
    finally:
        workspace.cleanup()

    return _report(args, workspace, result, lessons, client, why, gate)


def _report(args, workspace, result, lessons, client, why, gate) -> int:  # noqa: ANN001
    runs = [e for e in result.journal.events if e.kind == "agent.run"]
    spent = sum(e.data.get("cost_usd", 0.0) for e in runs)
    held = sum(e.kind in ("gate.hold", "brief.deny") for e in result.journal.events)

    print(f"{len(runs)} run(s), ${spent:.2f} spent, {held} tool call(s) held by the gate")
    for e in runs:
        print(f"  {e.data['tactic']:<18} ${e.data.get('cost_usd', 0):<7} "
              f"{e.data.get('changed_files', 0)} file(s)")

    # Whether the check passed is the entire point, and listing candidates
    # without it reads as though every patch works. The critic re-ran the check
    # for each ant; say what it found.
    verdicts = [e for e in result.journal.events if e.kind == "verify"]
    if verdicts:
        print("\nverification (your check, re-run against each agent's own tree):")
        for e in verdicts:
            print(f"  {e.data.get('tactic', '?'):<18} {e.data.get('reason', '')}")

    patches = workspace.patches
    print(f"\n{len(patches)} candidate patch(es); your repository is untouched")
    for patch in patches:
        print(f"  from {patch.tactic or patch.task:<18} "
              f"{len(patch.text.splitlines())} diff lines  {patch.files}")

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
