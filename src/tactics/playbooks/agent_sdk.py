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
    PreToolUse hook, so every write, every Bash command, every unknown tool is
    classified and decided by *our* gate, and lands in *our* journal. ``DryRun``
    turns the whole colony into a proposal engine that cannot write. (The hook
    rather than ``can_use_tool`` for a reason the first live run taught us the
    hard way — see :class:`GateBridge`.)
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

import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
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
# "Agent" is delegation (spawning a subagent); "Task" is the older name for the
# same tool — carry both, since the live tool name varies by CLI version and a
# roster that misses it silently degrades a swarm brief into a solo one.
# Delegation itself changes nothing: the subagent's own tool calls fire this same
# hook, individually, and are gated on their own merits.
DELEGATION_TOOLS = frozenset({"Agent", "Task"})

READ_ONLY_TOOLS = frozenset(
    {"Read", "Grep", "Glob", "NotebookRead", "WebSearch", "WebFetch", "TodoWrite"}
) | DELEGATION_TOOLS

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
    """Adapts this framework's :class:`ApprovalGate` to the SDK's permission system.

    This is the load-bearing join. Before any tool runs, the SDK asks "may I?";
    the answer comes from ``ctx.gate`` — the same gate that governs every other
    tactic — and the decision is written to ``ctx.journal``. Consequences that
    fall out for free:

      * ``DryRun`` → the agent may read and reason but cannot write a byte, and
        the journal ends up holding the change it *wanted* to make.
      * ``PolicyGate`` → edits auto-approve, ``git push`` and unknown tools go to
        a human.
      * ``CallbackGate`` → your own hook sees every call.

    **It asks through a PreToolUse hook, and that choice is load-bearing.** The
    SDK's other permission seam, ``can_use_tool``, is *shadowed*: an
    ``allowed_tools`` entry naming a whole tool — or ``permission_mode
    "bypassPermissions"``, or an allow rule in a settings file — auto-approves
    the call before the callback is ever consulted. A gate with a silent bypass
    is not a gate. A PreToolUse hook sees every call regardless.

    ``brief_tools`` is the roster the brief declared. It is enforced *here*, not
    by the SDK's ``allowed_tools`` (which grants permission rather than
    withholding it), so a brief that says "Read and Edit only" means it.
    """

    def __init__(self, ctx: Any, brief_tools: tuple[str, ...] | None = None) -> None:
        self.ctx = ctx
        self.brief_tools = frozenset(brief_tools) if brief_tools else None
        self.allowed: list[str] = []
        self.denied: list[str] = []

    def decide(
        self,
        tool_name: str,
        input_data: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> tuple[bool, str]:
        """Sync decision core — the whole permission model, testable without the SDK.

        ``agent`` names the subagent that made the call (``None`` on the main
        loop), so a swarm's journal says which ant asked for what.
        """
        input_data = input_data or {}
        who = f"{agent} subagent" if agent else "agent"
        if self.brief_tools is not None and tool_name not in self.brief_tools:
            self.denied.append(tool_name)
            reason = f"{tool_name} is outside this brief's tool allowlist"
            journal = getattr(self.ctx, "journal", None)
            if journal is not None:  # a refusal nobody can see is a refusal nobody trusts
                journal.record("brief.deny", tool=tool_name, agent=agent, reason=reason)
            return False, reason
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
        detail: dict[str, Any] = {"tool": tool_name}
        if agent:
            detail["agent"] = agent
        if tool_name == "Bash":
            detail["command"] = str(input_data.get("command", ""))[:200]
        elif "file_path" in input_data:
            detail["file_path"] = str(input_data["file_path"])

        # commit=None on purpose: the gate is deciding *permission*, and the SDK
        # performs the action. Submitting still journals the decision.
        result = gate.submit(
            Proposal(
                action=f"{who} tool call: {tool_name}",
                commit=None,
                reversible=reversible,
                risk=risk,
                detail=detail,
            ),
            self.ctx,
        )
        (self.allowed if result.approved else self.denied).append(tool_name)
        return result.approved, result.reason

    def as_hook(self) -> Callable[..., Any]:
        """Return the async PreToolUse hook callback — the un-shadowable seam."""

        async def pre_tool_use(hook_input, tool_use_id, context):  # noqa: ANN001 - SDK contract
            approved, reason = self.decide(
                hook_input.get("tool_name", ""),
                hook_input.get("tool_input") or {},
                # Present only when the call came from inside a Task-spawned
                # subagent. Verified live: subagent calls do fire this hook, so a
                # brief cannot delegate its way around DryRun.
                agent=hook_input.get("agent_type"),
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow" if approved else "deny",
                    "permissionDecisionReason": reason,
                }
            }

        return pre_tool_use


def _roster_error(spec: "BriefSpec") -> str:
    """Why this brief cannot do what it says, or ``""`` if it can.

    Found live: `ReviewedSwarm` listed "Task" as its delegation tool, but the
    installed CLI names it "Agent", so the gate denied the one call the brief
    existed to make. It still scored 1.0 — as a solo run wearing a swarm's name,
    which is worse than a failure, because the policy learns the wrong thing.
    """
    if spec.agents and not (set(spec.allowed_tools) & DELEGATION_TOOLS):
        return (
            f"brief declares subagents {sorted(spec.agents)} but its tool allowlist "
            f"has no delegation tool (one of {sorted(DELEGATION_TOOLS)}), so it can "
            "never reach them"
        )
    return ""


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
class Patch:
    """One ant's work, lifted out of its worktree before the worktree is destroyed.

    Under isolation the main repository is never written to, so this is the whole
    product of a parallel run. Landing it is a separate, deliberate act — see
    :meth:`AgentWorkspace.apply_patch`.
    """

    task: str
    text: str
    files: list[str] = field(default_factory=list)
    tactic: str = ""

    def __bool__(self) -> bool:
        return bool(self.text.strip())


@dataclass
class AgentRun:
    """What one SDK run produced. ``cost_usd`` is what the Budget spends."""

    text: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tools_used: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    # What the SDK itself counted as denied — an independent check on our own
    # tally, so a gate that silently stopped firing shows up as a mismatch.
    sdk_denials: int = 0
    models: list[str] = field(default_factory=list)
    error: str = ""


def _sdk_runner(brief: str, spec: BriefSpec, bridge: GateBridge | None, ws: "AgentWorkspace") -> AgentRun:
    """Run a real SDK agent. Lazy-imports ``claude_agent_sdk``.

    Three things here are the difference between a gate and the appearance of one,
    and all three were learned from the first live run:

    * **The brief's tool roster is never sent as ``allowed_tools``.** That option
      *grants* permission — every tool named in it is auto-approved before any
      callback runs. Sending the roster there silently disabled the gate and a
      ``DryRun`` colony wrote two files. The roster is enforced by the bridge.
    * **Permission goes through a PreToolUse hook**, which sees every call.
    * **``bypassPermissions`` is refused outright**, because it shadows the hook.

    Cost comes from ``ResultMessage.total_cost_usd``, not from summing assistant
    usage: with subagents the ``usage`` field counts only the top-level loop, so
    a swarm brief would look artificially cheap and the Budget would under-count
    exactly the tactic most able to run away with the bill. (That field is a
    client-side estimate — good for budgeting, not for billing anyone.)

    Options are filtered to the fields the installed ``ClaudeAgentOptions``
    actually declares, so a version skew drops an option instead of raising.
    """
    import asyncio
    from dataclasses import fields as dataclass_fields

    if spec.permission_mode == "bypassPermissions":
        return AgentRun(error="refused: permission_mode 'bypassPermissions' bypasses the gate")

    try:
        from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, HookMatcher, query
    except ImportError as exc:
        return AgentRun(error=f"claude-agent-sdk not installed: {exc}")

    wanted: dict[str, Any] = {
        "system_prompt": spec.system_prompt or None,
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
        # matcher=None matches every tool. This is the only permission seam that
        # an allowed_tools entry or a settings-file allow rule cannot shadow.
        wanted["hooks"] = {"PreToolUse": [HookMatcher(hooks=[bridge.as_hook()])]}

    known = {f.name for f in dataclass_fields(ClaudeAgentOptions)}
    options = ClaudeAgentOptions(
        **{k: v for k, v in wanted.items() if v is not None and k in known}
    )

    run = AgentRun()

    async def drive() -> None:
        # Content blocks are matched structurally: the installed dataclasses carry
        # no discriminating ``type`` field, so a ``block.type == "tool_use"`` read
        # matches nothing and silently reports zero tool calls.
        async for message in query(prompt=brief, options=options):
            for block in getattr(message, "content", None) or []:
                if hasattr(block, "name") and hasattr(block, "input"):
                    run.tools_used.append(getattr(block, "name", "?"))
                elif hasattr(block, "text"):
                    run.text = getattr(block, "text", "")
            if hasattr(message, "total_cost_usd"):  # the ResultMessage
                # A call can emit more than one result; the last carries the
                # running total for the whole call, so overwrite, never sum.
                run.cost_usd = float(getattr(message, "total_cost_usd", None) or 0.0)
                # Tokens come from model_usage for the same reason cost does:
                # `usage` covers only the top-level loop. A live ReviewedSwarm run
                # reported usage out:512 against model_usage out:5388.
                model_usage = getattr(message, "model_usage", None) or {}
                if model_usage:
                    run.models = sorted(model_usage)
                    run.input_tokens = sum(int(u.get("inputTokens", 0) or 0) for u in model_usage.values())
                    run.output_tokens = sum(int(u.get("outputTokens", 0) or 0) for u in model_usage.values())
                else:
                    usage = getattr(message, "usage", None) or {}
                    if isinstance(usage, dict):
                        run.input_tokens = int(usage.get("input_tokens", 0) or 0)
                        run.output_tokens = int(usage.get("output_tokens", 0) or 0)
                run.sdk_denials = len(getattr(message, "permission_denials", None) or [])

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
        isolate: bool = False,
    ) -> None:
        self.path = path
        self.check = list(check) if check else ["python3", "-m", "pytest", "-q"]
        self.model = model
        self.isolate = isolate
        self._runner = runner or _sdk_runner
        self._custom_shell = shell is not None
        self._shell = shell or self._subprocess
        self._lock = threading.Lock()
        self._worktree_root: str | None = None
        self.patches: list[Patch] = []
        self.task_id: str = ""  # set on a session, for patch attribution

    # --- per-ant isolation ---------------------------------------------------

    def session(self, task: Any = None) -> "AgentWorkspace":
        """Hand this ant its own git worktree, so parallel ants can't collide.

        Without this, ``max_workers > 1`` means several agents editing one working
        tree: the diffs interleave, and no result can be attributed to the ant
        that produced it — which corrupts the learning signal rather than merely
        losing work. Each session is a detached checkout of HEAD, so it starts
        from a known commit and never sees another ant's half-finished edit.

        Note the two consequences worth knowing before you turn this on: the
        main repository is never written to (work comes back as a :class:`Patch`),
        and the check command runs *inside* the worktree — so make it
        self-contained, or it will measure the wrong tree.
        """
        if not self.isolate:
            return self
        with self._lock:
            if self._worktree_root is None:
                # Outside the repo, or git would treat it as untracked content.
                self._worktree_root = tempfile.mkdtemp(prefix="tactics-worktrees-")
            path = os.path.join(self._worktree_root, f"ant-{uuid.uuid4().hex[:8]}")
            code, out = self.run(["git", "worktree", "add", "--detach", path, "HEAD"])
        if code != 0:
            # Fail loudly. Silently sharing the main tree is the exact corruption
            # this method exists to prevent.
            raise RuntimeError(f"could not create worktree at {path}: {out.strip()[-300:]}")
        child = AgentWorkspace(
            path,
            check=self.check,
            runner=self._runner,
            # A custom shell is a test seam and passes through; the default one is
            # bound to its owner's path, so the child must build its own.
            shell=self._shell if self._custom_shell else None,
            model=self.model,
        )
        child.task_id = str(getattr(task, "id", "") or "")
        return child

    def release(self, session: "Target") -> None:
        """Lift the work out as a patch, then remove the worktree."""
        if session is self or not isinstance(session, AgentWorkspace):
            return
        patch = session.capture_patch()
        with self._lock:
            if patch:
                self.patches.append(patch)
        self.run(["git", "worktree", "remove", "--force", session.path])

    def capture_patch(self) -> Patch:
        """The full diff of this workspace, including files the agent created."""
        self.run(["git", "add", "-A", "-N"])  # intent-to-add: untracked files show up
        _code, text = self.run(["git", "diff"])
        return Patch(task=self.task_id, text=text, files=self.changed_files())

    def apply_patch(self, patch: Patch) -> tuple[bool, str]:
        """Land a patch on this repository. Irreversible enough to gate.

        Concurrent ants can produce patches that touch the same lines; ``--3way``
        resolves what it can and fails loudly on the rest rather than mangling
        the tree.
        """
        handle, tmp = tempfile.mkstemp(suffix=".patch")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(patch.text if patch.text.endswith("\n") else patch.text + "\n")
            code, out = self.run(["git", "apply", "--3way", tmp])
            return code == 0, out
        finally:
            os.unlink(tmp)

    def cleanup(self) -> None:
        """Remove every worktree this workspace created. Safe to call twice."""
        with self._lock:
            root, self._worktree_root = self._worktree_root, None
        if not root:
            return
        for name in sorted(os.listdir(root)):
            self.run(["git", "worktree", "remove", "--force", os.path.join(root, name)])
        self.run(["git", "worktree", "prune"])
        shutil.rmtree(root, ignore_errors=True)

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

    def snapshot(self) -> str:
        """A cheap fingerprint of the working tree, for before/after comparison.

        Absolute dirtiness is not evidence: a tree that was already dirty makes
        every run look like it changed something, and a tactic would bank a win
        it did not earn. Covers tracked content (``git diff``) and the set of
        untracked paths (``git status``), so an edit to an already-dirty file
        still registers.
        """
        return f"{self.run(['git', 'status', '--porcelain'])[1]}\x00{self.diff_text()}"

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
        broken = _roster_error(self.spec)
        if broken:  # never spend money on a brief that cannot do its job
            return Outcome(success=False, reward=0.0, notes=f"misconfigured brief: {broken}")
        bridge = GateBridge(ctx, self.spec.allowed_tools)
        before_snapshot = ctx.target.snapshot()
        before_files = set(ctx.target.changed_files())
        run = ctx.target.run_agent(self.build_brief(ctx), self.spec, bridge)
        cost = run.cost_usd
        metrics = {
            "tools": len(run.tools_used),
            "denied": len(run.denied),
            "cost_usd": round(cost, 4),
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "models": len(run.models),
        }

        journal = getattr(ctx, "journal", None)

        if run.error:
            if journal is not None:
                journal.record("agent.run", tactic=self.name, **metrics)
            return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                           notes=f"agent run failed: {run.error[:200]}")

        # Measured against the baseline, not against absolute dirtiness.
        changed = sorted(set(ctx.target.changed_files()) - before_files)
        metrics["changed_files"] = len(changed)
        # Journalled after the measurement, so the audit trail carries what the
        # run actually did rather than only what it was asked to do.
        if journal is not None:
            journal.record("agent.run", tactic=self.name, **metrics)
        if ctx.target.snapshot() == before_snapshot:
            held = f" ({len(run.denied)} tool call(s) held by the gate)" if run.denied else ""
            return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                           notes=f"agent left the working tree untouched{held}")

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
        allowed_tools=("Read", "Grep", "Glob", "Edit", "Write", "Bash", *sorted(DELEGATION_TOOLS)),
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
    """A goal satisfied when the workspace's check command passes.

    Right for **repair**: the suite is red, and green is the finish line. Wrong
    for new behavior — if the check already passes, this goal is satisfied before
    a single ant runs, and the colony stops having done nothing. Use
    :func:`work_queue_goal` for that; the live run that found this reported
    "satisfied after 0 rounds" on a perfectly healthy repo.
    """
    return Goal(
        name=name,
        description=description,
        is_satisfied=lambda ctx: ctx.target.verify()[0],
    )


def work_queue_goal(description: str, name: str = "delivery") -> Goal:
    """A goal with no repo-level finish line: run until the work queue is empty.

    For **new behavior**, where "the check passes" was already true before you
    started and so says nothing about whether the job got done. The colony stops
    on "no open work" (or its round/budget caps); each ant's own reward still
    comes from the check re-run in its own worktree.
    """
    return Goal(name=name, description=description)


def verification_critic() -> FunctionCritic:
    """Re-measure the repo before the colony learns anything from an outcome.

    The failure this exists to catch is a tactic that claims a win it didn't earn:
    accept only when the claim and the repo agree. A genuine, verified loss *is*
    accepted — learning that a brief fails here is the point.
    """

    def verify(outcome: Outcome, ctx: Any) -> Verdict:
        # A run the gate held is a verdict about the *gate*, not about the brief.
        # Learning "SingleAgentNarrow scores 0" from a DryRun would teach the
        # policy to avoid a brief that was never allowed to try, and completing
        # the task would mark undone work as done.
        if not outcome.success and outcome.metrics.get("denied") and not outcome.metrics.get("changed_files"):
            return Verdict(accepted=False, reason="held by the gate — no evidence about this brief")
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

    ``max_workers > 1`` requires ``AgentWorkspace(isolate=True)`` and is refused
    without it. That is not caution: parallel agents editing one working tree
    produce interleaved diffs that cannot be attributed to the ant that made
    them, so the swarm would learn from noise — a quieter, worse failure than a
    crash. With isolation each ant gets its own git worktree, the main repo is
    never written to, and the work comes back as ``target.patches``.

    Persist ``memory`` (a ``JsonStore``) to keep what the briefs earn across runs
    — an in-memory store makes every session start cold.
    """
    if max_workers > 1 and not getattr(target, "isolate", False):
        raise ValueError(
            "max_workers > 1 needs AgentWorkspace(isolate=True): parallel agents "
            "sharing one working tree make every result unattributable"
        )
    tactics = tactics or [SingleAgentNarrow(), WriteTestFirst(), PlanThenPatch(), ReviewedSwarm()]

    def plan(goal, board, target):  # noqa: ANN001
        if board.tasks:
            return []
        # One task per worker: with isolation these run as independent attempts at
        # the same goal, and the policy picks a brief for each. Serial runs get a
        # single task and retry it across rounds instead.
        return [
            board.post_task(goal.description or f"pursue:{goal.name}")
            for _ in range(max_workers)
        ]

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
