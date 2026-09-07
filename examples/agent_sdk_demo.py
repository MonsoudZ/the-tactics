"""Agent-SDK demo: the framework governing a Claude Code agent — offline.

The split this playbook exists for:

  * the **Claude Agent SDK** is the execution layer (edits files, runs tests,
    spawns subagents) — years ahead of anything worth rebuilding here;
  * **the-tactics** is the governance layer around it — which brief to send,
    whether to believe the result, what it cost, and what it learned.

This demo runs a *scripted* agent (no SDK, no API key, no network) so you can
watch the machinery. Swap the ``runner`` out and the same colony drives a real
Claude Code agent against a real repo — verified against SDK 0.2.152, where the
three postures below produced 0, 2 and 2 written files respectively.

Run it:  python3 examples/agent_sdk_demo.py
"""

from __future__ import annotations

from tactics import AutoApprove, Budget, DryRun, PolicyGate
from tactics.playbooks.agent_sdk import (
    AgentRun,
    AgentWorkspace,
    build_delivery_colony,
    delivery_goal,
)


class ScriptedAgent:
    """Stands in for a real SDK run: asks the gate, then 'edits' if allowed."""

    def __init__(self, cost: float = 0.30) -> None:
        self.cost = cost
        self.edited = False

    def runner(self, brief, spec, bridge, ws) -> AgentRun:
        run = AgentRun(cost_usd=self.cost, input_tokens=1200, output_tokens=300)
        for tool, payload in (
            ("Read", {"file_path": "parser.py"}),
            ("Write", {"file_path": "parser.py"}),
            ("Bash", {"command": "python -m pytest -q"}),
            ("Bash", {"command": "git push origin main"}),  # the one a human should see
        ):
            allowed, _ = bridge.decide(tool, payload)
            run.tools_used.append(tool)
            if allowed and tool == "Write":
                self.edited = True
        # A delegated call, as a real swarm brief produces. The gate sees the
        # subagent's calls individually — delegation is not a way around it.
        bridge.decide("Bash", {"command": "pytest -q"}, agent="code-reviewer")
        run.denied = list(bridge.denied)
        return run

    def shell(self, cmd):
        if cmd[:2] == ["git", "status"]:
            return 0, " M parser.py\n" if self.edited else ""
        if cmd[:2] == ["git", "rev-parse"]:
            return 0, "main\n"
        if cmd[:2] == ["git", "diff"]:
            return 0, ""
        return (0, "3 passed") if self.edited else (1, "1 failed")


def run(label: str, gate) -> None:
    agent = ScriptedAgent()
    ws = AgentWorkspace(".", check=["pytest"], runner=agent.runner, shell=agent.shell)
    colony = build_delivery_colony(
        ws, gate=gate, budget=Budget(max_cost=5.00), max_rounds=2
    )
    result = colony.run(delivery_goal("make the parser handle empty input"))

    print(f"\n=== {label} ===")
    print(f"  files changed : {len(ws.changed_files())}")
    print(f"  check passes  : {ws.verify()[0]}")
    for event in result.journal.events:
        if event.kind.startswith("gate.") or event.kind == "agent.run":
            print(f"  {event.kind:<12} {event.data}")


def fan_out() -> None:
    """Three ants, three git worktrees, one repo — and the repo stays clean."""
    import pathlib
    import subprocess
    import tempfile

    from tactics.playbooks.agent_sdk import work_queue_goal

    root = tempfile.mkdtemp(prefix="tactics-demo-repo-")
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, check=True)
    git("init", "-q"); git("config", "user.email", "d@d"); git("config", "user.name", "d")
    pathlib.Path(root, "seed.txt").write_text("seed\n")
    git("add", "-A"); git("commit", "-qm", "seed")

    def runner(brief, spec, bridge, ws):
        """Each ant writes a file naming its own tree, if the gate allows it."""
        run = AgentRun(cost_usd=0.05)
        allowed, _ = bridge.decide("Write", {"file_path": "work.txt"})
        run.tools_used.append("Write")
        if allowed:
            pathlib.Path(ws.path, "work.txt").write_text(f"done in {pathlib.Path(ws.path).name}\n")
        run.denied = list(bridge.denied)
        return run

    ws = AgentWorkspace(root, check=["test", "-f", "work.txt"], runner=runner, isolate=True)
    try:
        colony = build_delivery_colony(ws, gate=AutoApprove(), max_workers=3, max_rounds=1)
        result = colony.run(work_queue_goal("do the work"))
        briefs = [e.data["tactic"] for e in result.journal.events if e.kind == "agent.run"]
        print("\n=== Fan-out — 3 ants, 3 worktrees, 1 repo ===")
        print(f"  briefs tried     : {briefs}   <- three different ones, one round")
        print(f"  patches captured : {len(ws.patches)} ({len({p.text for p in ws.patches})} distinct)")
        print(f"  main repo dirty  : {ws.changed_files()}   <- never written to")
        ok, _ = ws.apply_patch(ws.patches[0])
        print(f"  landed one patch : ok={ok}, repo now {ws.changed_files()}")
    finally:
        ws.cleanup()


def compounding() -> None:
    """Run 1 teaches run 2 — through disk, not through this process."""
    import json
    import pathlib
    import subprocess
    import tempfile

    from tactics.llm import ScriptedClient
    from tactics.playbooks.agent_sdk import brief_lessons, run_and_learn, work_queue_goal

    root = tempfile.mkdtemp(prefix="tactics-demo-compound-")
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, check=True)
    git("init", "-q"); git("config", "user.email", "d@d"); git("config", "user.name", "d")
    pathlib.Path(root, "seed.txt").write_text("seed\n")
    git("add", "-A"); git("commit", "-qm", "seed")

    sent: list[str] = []

    def runner(brief, spec, bridge, ws):
        sent.append(brief)
        run = AgentRun(cost_usd=0.05)
        if bridge.decide("Write", {"file_path": "work.txt"})[0]:
            pathlib.Path(ws.path, "work.txt").write_text("done\n")
        return run

    lesson = "this suite needs PYTHONPATH=$PWD/src or it measures the wrong tree"
    scribe_says = ScriptedClient([json.dumps({"lessons": [{"text": lesson, "evidence": "round 1"}]})])

    ws = AgentWorkspace(root, check=["test", "-f", "work.txt"], runner=runner, isolate=True)
    colony = build_delivery_colony(ws, persist=True, gate=AutoApprove(), max_rounds=1)
    _result, written = run_and_learn(colony, work_queue_goal("do the work"), client=scribe_says)
    ws.cleanup()

    print("\n=== Compounding — run 1 writes down what it learned ===")
    print(f"  brief run 1 was given : {sent[0]!r}")
    print(f"  lessons the scribe wrote: {[lesson.text for lesson in written]}")
    print(f"  on disk: {sorted(p.name for p in pathlib.Path(root, '.tactics').iterdir())}")

    # A second process would see exactly this: nothing in memory, everything on disk.
    ws2 = AgentWorkspace(root, check=["test", "-f", "work.txt"], runner=runner, isolate=True)
    colony2 = build_delivery_colony(ws2, persist=True, gate=AutoApprove(), max_rounds=1)
    colony2.run(work_queue_goal("do the next thing"))
    ws2.cleanup()
    print(f"\n  brief run 2 was given : {sent[-1].splitlines()[0]!r}")
    print(f"  ...carrying the lesson: {lesson in sent[-1]}")
    print(f"  what each brief has earned: "
          f"{ {e.tactic: round(e.stats.mean_reward, 2) for e in colony2.memory.entries()} }")


if __name__ == "__main__":
    # 1. Everything allowed — the agent works, the repo is measured, the colony learns.
    run("AutoApprove — the agent works", AutoApprove())

    # 2. Nothing allowed — the agent may read and reason, but cannot write a byte.
    #    The journal holds the change it wanted to make. Review, then re-run for real.
    run("DryRun — proposal only, zero writes", DryRun())

    # 3. The realistic posture: edits auto-approve, `git push` goes to a human.
    run("PolicyGate — edits fine, push escalates", PolicyGate(escalate=lambda p, c: False))

    # 4. Parallel ants, each in its own git worktree — max_workers > 1 is refused
    #    without isolation, because interleaved diffs cannot be attributed.
    fan_out()

    # 5. And what one run learns, the next one starts with.
    compounding()

    print(
        "\nReward came from re-running the check, never from the agent's summary.\n"
        "Cost is dollars, so Budget(max_cost=...) is a real ceiling.\n"
        "Point `runner` at claude_agent_sdk and none of the above changes."
    )
