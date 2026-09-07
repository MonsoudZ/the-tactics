"""Agent-SDK playbook — the Claude Agent SDK as a Target the framework governs.

The execution layer (edit files, run tests, spawn subagents, compact context) is
a solved problem: the **Claude Agent SDK** is Claude Code packaged as a library.
What it deliberately has no opinion about is everything this framework is for —
*which* brief to send, whether to believe the result, what the whole thing cost
across two hundred runs, and what last week taught it.

So we don't compete with the harness; we govern it. The SDK becomes a Target:

    Goal → pick a Brief → run an SDK agent → measure the repo → learn → repeat

Four seams do the joining, and each one is a contract that already existed:

  * **Target** — :class:`AgentWorkspace` runs the agent and, separately, measures
    the repo. ``runner`` is injectable, so every test here is offline.
  * **Tactic** — a :class:`BriefTactic` *is* a brief: a system prompt, a tool
    allowlist, and a subagent roster. Different brief shapes compete on results,
    and the policy learns which shape wins on which kind of task. This is the
    thing a static roster of markdown prompts cannot do.
  * **ApprovalGate** — :class:`GateBridge` adapts ``ctx.gate`` to the SDK's
    ``can_use_tool`` callback, so every write, every Bash command, every unknown
    tool is classified and decided by *our* gate, and lands in *our* journal.
    ``DryRun`` turns the whole colony into a proposal engine that cannot write.
  * **Budget** — ``Outcome.cost`` carries the run's dollar estimate, so
    ``Budget(max_cost=5.00)`` is a real ceiling on an autonomous swarm.

**Reward is measured, never reported.** The agent's own account of its work is
exactly what we don't trust: reward comes from re-running the repo's check
command and looking at the diff. A run that says "done!" and changed nothing, or
left the tests red, scores 0. Efficiency lives on ``cost``, not ``reward`` —
they're separate axes on purpose (a cheap run that fails is not a good run).

    from tactics.playbooks.agent_sdk import AgentWorkspace, build_delivery_colony, delivery_goal
    ws = AgentWorkspace(".", check=["python3", "-m", "pytest", "-q"])
    colony = build_delivery_colony(ws, gate=DryRun())          # propose-only
    result = colony.run(delivery_goal("add retry to the client"))
    print(result.journal.explain())

Requires ``pip install claude-agent-sdk`` to run for real; the SDK is imported
lazily (like ``ClaudeClient``), so this module imports and its tests pass
without it.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from ..colony import Colony, FunctionCritic, FunctionPlanner, Verdict
from ..core.approval import Proposal
from ..core.goal import Goal
from ..core.memory import InMemoryStore, MemoryStore
from ..core.outcome import Outcome
from ..core.tactic import Tactic
from ..core.target import Target

# Tools that only read. They never reach the gate — gating them would flood the
# journal with noise and teach reviewers to skim it, which is how a real approval
# gets waved through.
READ_ONLY_TOOLS = frozenset(
    {"Read", "Grep", "Glob", "NotebookRead", "WebSearch", "WebFetch", "TodoWrite", "Task"}
)

# Tools that change the working tree. Reversible because the workspace is a git
# repo: the diff is inspectable and revertible before anything is committed.
EDIT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})

# Bash commands that leave the workspace — push somewhere, destroy history, or
# install something. These are the ones a human should see.
_IRREVERSIBLE_BASH = [
    re.compile(p)
    for p in (
        r"\brm\s+(-\w+\s+)*-\w*[rf]",
        r"\bgit\s+push\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\b",
        r"\bgit\s+commit\b",
        r"\bsudo\b",
        r"\bcurl\b[^|]*\|\s*(ba)?sh",
        r"\bdd\s+if=",
        r"\bmkfs\b",
        r"\bchmod\s+777\b",
        r"\b(npm|pip|gem|cargo)\s+publish\b",
        r"\b(docker|kubectl|terraform|aws|gcloud)\b",
        r"\bgh\s+(pr|release)\b",
        r">\s*/dev/(sd|nvme)",
    )
]


def classify_tool_call(tool_name: str, input_data: dict[str, Any]) -> tuple[bool, str]:
    """Judge one tool call: ``(reversible, risk)`` for the :class:`Proposal`.

    Fail-closed on anything unrecognized — an unknown tool is treated as
    irreversible and high risk, so a ``PolicyGate`` escalates it instead of
    waving it through. A new SDK tool should have to earn its way onto a list.
    """
    if tool_name in READ_ONLY_TOOLS:
        return True, "low"
    if tool_name in EDIT_TOOLS:
        return True, "low"
    if tool_name == "Bash":
        cmd = str(input_data.get("command", ""))
        if any(p.search(cmd) for p in _IRREVERSIBLE_BASH):
            return False, "high"
        return True, "medium"
    return False, "high"


class GateBridge:
    """Adapts this framework's :class:`ApprovalGate` to the SDK's ``can_use_tool``.

    This is the load-bearing join. The SDK asks "may I run this tool?"; the
    answer comes from ``ctx.gate`` — the same gate that governs every other
    tactic — and the decision is written to ``ctx.journal``. Consequences that
    fall out for free:

      * ``DryRun`` → the agent may read and reason but cannot write a byte, and
        the journal ends up holding the change it *wanted* to make.
      * ``PolicyGate`` → edits auto-approve, ``git push`` and unknown tools go to
        a human.
      * ``CallbackGate`` → your own hook sees every call.

    Denials are returned with ``interrupt=False`` so the agent learns the
    boundary and routes around it rather than dying mid-task.
    """

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.allowed: list[str] = []
        self.denied: list[str] = []

    def decide(self, tool_name: str, input_data: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Sync decision core — the whole permission model, testable without the SDK."""
        input_data = input_data or {}
        if tool_name in READ_ONLY_TOOLS:
            self.allowed.append(tool_name)
            return True, "read-only"

        gate = getattr(self.ctx, "gate", None)
        if gate is None:
            # Fail closed. A missing gate is a wiring bug, and the safe reading of
            # a wiring bug is "no", not "yes".
            self.denied.append(tool_name)
            return False, "no approval gate wired — denying"

        reversible, risk = classify_tool_call(tool_name, input_data)
        detail = {"tool": tool_name}
        if tool_name == "Bash":
            detail["command"] = str(input_data.get("command", ""))[:200]
        elif "file_path" in input_data:
            detail["file_path"] = str(input_data["file_path"])

        # commit=None on purpose: the gate is deciding *permission*, and the SDK
        # performs the action. Submitting still journals the decision.
        result = gate.submit(
            Proposal(
                action=f"agent tool call: {tool_name}",
                commit=None,
                reversible=reversible,
                risk=risk,
                detail=detail,
            ),
            self.ctx,
        )
        (self.allowed if result.approved else self.denied).append(tool_name)
        return result.approved, result.reason

    def as_callback(self) -> Callable[..., Any]:
        """Return the async ``can_use_tool`` callback the SDK expects."""

        async def can_use_tool(tool_name, input_data, context):  # noqa: ANN001 - SDK contract
            from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

            approved, reason = self.decide(tool_name, input_data)
            if approved:
                return PermissionResultAllow(updated_input=input_data)
            return PermissionResultDeny(message=reason, interrupt=False)

        return can_use_tool


@dataclass
class BriefSpec:
    """Everything that varies between one brief shape and another.

    ``agents`` stays a plain dict (``{name: {description, prompt, tools, model}}``)
    so this module never imports the SDK; it's converted to ``AgentDefinition``
    inside the runner.
    """

    system_prompt: str = ""
    allowed_tools: tuple[str, ...] = ("Read", "Grep", "Glob", "Edit", "Write", "Bash")
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    permission_mode: str = "default"
    max_turns: int | None = None
    model: str | None = None


@dataclass
class AgentRun:
    """What one SDK run produced. ``cost_usd`` is what the Budget spends."""

    text: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tools_used: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    error: str = ""


def _sdk_runner(brief: str, spec: BriefSpec, bridge: GateBridge | None, ws: "AgentWorkspace") -> AgentRun:
    """Run a real SDK agent. Lazy-imports ``claude_agent_sdk``.

    Cost comes from ``ResultMessage.total_cost_usd``, not from summing assistant
    usage: with subagents the ``usage`` field counts only the top-level loop, so
    a swarm brief would look artificially cheap and the Budget would under-count
    exactly the tactic most able to run away with the bill. (That field is a
    client-side estimate, good for budgeting, not for billing anyone.)

    Options are filtered to the fields the installed ``ClaudeAgentOptions``
    actually declares, so a version skew drops an option instead of raising.
    """
    import asyncio
    from dataclasses import fields as dataclass_fields

    try:
        from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query
    except ImportError as exc:
        return AgentRun(error=f"claude-agent-sdk not installed: {exc}")

    wanted: dict[str, Any] = {
        "system_prompt": spec.system_prompt or None,
        "allowed_tools": list(spec.allowed_tools),
        "permission_mode": spec.permission_mode,
        "cwd": ws.path,
        "max_turns": spec.max_turns,
        "model": spec.model or ws.model,
    }
    if spec.agents:
        wanted["agents"] = {
            name: AgentDefinition(**definition) for name, definition in spec.agents.items()
        }
    if bridge is not None:
        wanted["can_use_tool"] = bridge.as_callback()

    known = {f.name for f in dataclass_fields(ClaudeAgentOptions)}
    options = ClaudeAgentOptions(
        **{k: v for k, v in wanted.items() if v is not None and k in known}
    )

    run = AgentRun()

    async def drive() -> None:
        async for message in query(prompt=brief, options=options):
            for block in getattr(message, "content", None) or []:
                if getattr(block, "type", None) == "tool_use":
                    run.tools_used.append(getattr(block, "name", "?"))
                elif getattr(block, "type", None) == "text":
                    run.text = getattr(block, "text", "")
            # ResultMessage carries the cumulative totals; both are optional.
            if hasattr(message, "total_cost_usd"):
                run.cost_usd = float(getattr(message, "total_cost_usd", None) or 0.0)
                usage = getattr(message, "usage", None) or {}
                if isinstance(usage, dict):
                    run.input_tokens = int(usage.get("input_tokens", 0) or 0)
                    run.output_tokens = int(usage.get("output_tokens", 0) or 0)

    try:
        asyncio.run(drive())
    except Exception as exc:  # noqa: BLE001 - a failed run is a loss, never a crash
        # query() raises after yielding an error result, so whatever cost and
        # tool history arrived before the failure is still on `run`.
        run.error = repr(exc)

    if bridge is not None:
        run.denied = list(bridge.denied)
    return run


class AgentWorkspace(Target):
    """A git working tree an SDK agent acts on, and that we measure independently.

    Two seams, deliberately separate: ``run_agent`` is what the agent does,
    ``verify`` is what actually happened. Both are injectable, so the tests in
    this repo run with no SDK, no API key, and no network.
    """

    name = "agent_sdk"

    def __init__(
        self,
        path: str = ".",
        *,
        check: list[str] | None = None,
        runner: Callable[[str, BriefSpec, "GateBridge | None", "AgentWorkspace"], AgentRun] | None = None,
        shell: Callable[[list[str]], tuple[int, str]] | None = None,
        model: str | None = None,
    ) -> None:
        self.path = path
        self.check = list(check) if check else ["python3", "-m", "pytest", "-q"]
        self.model = model
        self._runner = runner or _sdk_runner
        self._shell = shell or self._subprocess

    # --- shell seam ----------------------------------------------------------

    def _subprocess(self, cmd: list[str]) -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, cwd=self.path, capture_output=True, text=True, timeout=1800)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except FileNotFoundError:
            return 127, f"command not found: {cmd[0]}"
        except Exception as exc:  # noqa: BLE001 - report, never raise into the loop
            return 1, f"error running {cmd}: {exc!r}"

    def run(self, cmd: list[str]) -> tuple[int, str]:
        return self._shell(cmd)

    # --- observation ---------------------------------------------------------

    def observe(self) -> dict[str, Any]:
        """Total and non-throwing, per the Target contract."""
        changed = self.changed_files()
        code, branch = self.run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return {
            "path": self.path,
            "branch": branch.strip() if code == 0 else "",
            "changed_files": changed,
            "check": " ".join(self.check),
        }

    def features(self, data: dict[str, Any]) -> dict[str, Any]:
        """Bucket learning by situation: a dirty tree and a clean one are
        different problems, and a brief that wins on one may lose on the other."""
        return {"dirty": bool(data.get("changed_files"))}

    def changed_files(self) -> list[str]:
        code, out = self.run(["git", "status", "--porcelain"])
        if code != 0:
            return []
        return [ln[3:].strip() for ln in out.splitlines() if ln.strip()]

    def diff_text(self) -> str:
        code, out = self.run(["git", "diff"])
        return out if code == 0 else ""

    def verify(self) -> tuple[bool, str]:
        """Run the check command. This — not the agent's summary — is the reward."""
        code, out = self.run(self.check)
        return code == 0, out

    # --- the agent seam ------------------------------------------------------

    def run_agent(self, brief: str, spec: BriefSpec, bridge: GateBridge | None = None) -> AgentRun:
        return self._runner(brief, spec, bridge, self)


class BriefTactic(Tactic):
    """A brief that competes.

    Subclass and override :meth:`build_brief` (and pass a :class:`BriefSpec`) to
    add a new strategy. Tactics never reference each other; they meet only in
    memory, as numbers. That is what lets you add a fifth brief tomorrow and have
    it start earning its place without touching the other four.
    """

    spec: BriefSpec = BriefSpec()

    def build_brief(self, ctx: Any) -> str:
        return str(ctx.task.description if ctx.task is not None else ctx.goal.description)

    def execute(self, ctx: Any) -> Outcome:
        bridge = GateBridge(ctx)
        run = ctx.target.run_agent(self.build_brief(ctx), self.spec, bridge)
        cost = run.cost_usd
        metrics = {
            "tools": len(run.tools_used),
            "denied": len(run.denied),
            "cost_usd": round(cost, 4),
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
        }

        journal = getattr(ctx, "journal", None)
        if journal is not None:
            journal.record("agent.run", tactic=self.name, **metrics, error=run.error)

        if run.error:
            return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                           notes=f"agent run failed: {run.error[:200]}")

        changed = ctx.target.changed_files()
        metrics["changed_files"] = len(changed)
        if not changed:
            held = f" ({len(run.denied)} tool call(s) held by the gate)" if run.denied else ""
            return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                           notes=f"agent made no changes{held}")

        passed, output = ctx.target.verify()
        return Outcome(
            success=passed,
            reward=1.0 if passed else 0.0,
            cost=cost,
            metrics=metrics,
            notes=f"check passed, {len(changed)} file(s) changed"
            if passed
            else f"check failed: {output.strip()[-300:]}",
        )


_BASE_RULES = (
    "Work only inside this repository. Make the smallest change that does the job. "
    "Do not commit, push, or install anything. Match the surrounding code's style."
)


class SingleAgentNarrow(BriefTactic):
    """One agent, minimal tools, tight scope. The cheap default — often enough,
    and the baseline every richer brief has to beat on results."""

    spec = BriefSpec(
        system_prompt=(
            f"You are a focused engineer. {_BASE_RULES} Change as few files as possible "
            "and stop as soon as the task is done."
        ),
        allowed_tools=("Read", "Grep", "Glob", "Edit", "Write", "Bash"),
    )


class WriteTestFirst(BriefTactic):
    """Write the failing test first, then make it pass. Costs more turns; tends to
    win where the bar is 'prove it', and to lose on trivial edits."""

    spec = BriefSpec(
        system_prompt=(
            f"You are a test-driven engineer. {_BASE_RULES} First write a test that fails "
            "for the stated reason and run it to confirm it fails. Only then write the fix, "
            "and re-run to confirm it passes."
        ),
        allowed_tools=("Read", "Grep", "Glob", "Edit", "Write", "Bash"),
    )

    def build_brief(self, ctx: Any) -> str:
        return f"{super().build_brief(ctx)}\n\nStart by writing a test that fails."


class PlanThenPatch(BriefTactic):
    """Read widely before touching anything, then patch narrowly. Earns its extra
    reading on unfamiliar code and wastes it on a one-liner."""

    spec = BriefSpec(
        system_prompt=(
            f"You are a careful engineer working in unfamiliar code. {_BASE_RULES} "
            "First read enough of the codebase to state, in two or three sentences, what "
            "you will change and why. Then make exactly that change and nothing more."
        ),
        allowed_tools=("Read", "Grep", "Glob", "Edit", "Write", "Bash"),
    )


class ReviewedSwarm(BriefTactic):
    """Main agent plus a reviewer subagent — the SDK's own parallelism, entered as
    one competitor among several rather than assumed to be the best shape."""

    spec = BriefSpec(
        system_prompt=(
            f"You are a senior engineer. {_BASE_RULES} When your change is written, "
            "delegate to the code-reviewer subagent and address what it finds before "
            "you finish."
        ),
        allowed_tools=("Read", "Grep", "Glob", "Edit", "Write", "Bash", "Task"),
        agents={
            "code-reviewer": {
                "description": "Reviews a working-tree diff for bugs and missed cases.",
                "prompt": (
                    "You review a diff adversarially. Report concrete defects — wrong "
                    "behavior, missed edge cases, untested paths — with file and line. "
                    "Say plainly when the diff is fine; do not invent findings."
                ),
                "tools": ["Read", "Grep", "Glob", "Bash"],
            }
        },
    )


def delivery_goal(description: str, name: str = "delivery") -> Goal:
    """A goal that is satisfied when the workspace's own check command passes."""
    return Goal(
        name=name,
        description=description,
        is_satisfied=lambda ctx: ctx.target.verify()[0],
    )


def verification_critic() -> FunctionCritic:
    """Re-measure the repo before the colony learns anything from an outcome.

    The failure this exists to catch is a tactic that claims a win it didn't earn:
    accept only when the claim and the repo agree. A genuine, verified loss *is*
    accepted — learning that a brief fails here is the point.
    """

    def verify(outcome: Outcome, ctx: Any) -> Verdict:
        passed, output = ctx.target.verify()
        if outcome.success and not passed:
            return Verdict(accepted=False, reward=0.0,
                           reason=f"claimed success but check fails: {output.strip()[-200:]}")
        if not outcome.success and passed and outcome.metrics.get("changed_files"):
            return Verdict(accepted=False, reason="claimed failure but check passes")
        return Verdict(accepted=True, reason="check re-run agrees")

    return FunctionCritic(verify)


def build_delivery_colony(
    target: AgentWorkspace,
    *,
    tactics: list[Tactic] | None = None,
    memory: MemoryStore | None = None,
    gate: Any = None,
    budget: Any = None,
    max_rounds: int = 4,
    max_workers: int = 1,
) -> Colony:
    """Wire the loop: one task, several competing briefs, verified after each try.

    ``max_workers`` defaults to 1 for a real reason, not caution: parallel ants
    would be parallel agents editing one working tree, and the diff we measure
    could not be attributed to any of them. Fan out by giving each worker its own
    checkout (a git worktree) before raising this.

    Persist ``memory`` (a ``JsonStore``) to keep what the briefs earn across runs
    — an in-memory store makes every session start cold.
    """
    tactics = tactics or [SingleAgentNarrow(), WriteTestFirst(), PlanThenPatch(), ReviewedSwarm()]

    def plan(goal, board, target):  # noqa: ANN001
        if board.tasks:
            return []
        return [board.post_task(goal.description or f"pursue:{goal.name}")]

    return Colony(
        target,
        tactics,
        FunctionPlanner(plan),
        memory=memory or InMemoryStore(),
        critic=verification_critic(),
        gate=gate,
        budget=budget,
        max_workers=max_workers,
        max_rounds=max_rounds,
    )
