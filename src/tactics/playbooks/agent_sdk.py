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

try:                        # advisory locks: POSIX only, and optional here
    import fcntl
except ImportError:         # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from ..colony import Colony, FunctionCritic, FunctionPlanner, Verdict
from ..core.approval import Proposal
from ..core.context import Context
from ..core.goal import Goal
from ..core.lessons import JsonlLessons, LessonStore, render_lessons
from ..core.memory import InMemoryStore, JsonStore, MemoryStore
from ..core.outcome import Outcome
from ..core.policy import Policy, UCBPolicy, WithoutReplacement
from ..core.tactic import Tactic
from ..core.target import Target
from ..llm.client import extract_json
from ..llm.scribe import Scribe

# Where a repo's accumulated experience lives. Numeric memory (which brief wins
# where) and verbal memory (what past runs learned) sit side by side, inside the
# repo they describe — so cloning the repo carries its experience with it, and a
# brief's record from a Rails app never leaks into a Swift one.
STORE_DIR = ".tactics"
MEMORY_FILE = "agent_sdk_memory.json"
LESSONS_FILE = "agent_sdk_lessons.jsonl"

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


WORKTREE_ROOT_PREFIX = "tactics-worktrees-"
OWNER_LOCK = ".owner.lock"


def _worktree_root_of(path: str) -> str | None:
    """The ``tactics-worktrees-*`` directory a worktree sits in, if any.

    Used to tell our own leavings from a worktree the user added themselves,
    which must never be touched.
    """
    current = os.path.abspath(path)
    while True:
        parent, name = os.path.split(current)
        if not name or parent == current:
            return None
        if name.startswith(WORKTREE_ROOT_PREFIX):
            return current
        current = parent


def _is_abandoned(root: str) -> bool:
    """True when no live process is still working in this worktree root.

    The question a reaper has to answer is "did the run that made this die, or
    is it still going?", and an advisory lock answers it with no bookkeeping to
    get wrong: the kernel drops it when the process ends, however it ends — a
    clean exit, a crash, or ``kill -9``. Where locks are unavailable the answer
    is no, because reaping a *live* run's worktrees is far worse than leaving a
    dead one's behind.
    """
    if fcntl is None:       # pragma: no cover - Windows
        return False
    lock = os.path.join(root, OWNER_LOCK)
    if not os.path.exists(lock):
        return True         # nobody ever claimed it
    try:
        with open(lock, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        return False        # somebody is still in there
    return True


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
    saved_to: str = ""          # where it was archived, if a patch_dir was given

    def __bool__(self) -> bool:
        return bool(self.text.strip())


@dataclass
class PatchTrial:
    """What happened when a candidate patch met a clean checkout of HEAD."""

    applies: bool
    passes: bool
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.applies and self.passes


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
        patch_dir: str | None = None,
    ) -> None:
        self.path = path
        self.check = list(check) if check else ["python3", "-m", "pytest", "-q"]
        self.model = model
        self.isolate = isolate
        # Where to archive each patch as it is lifted out. Without this the only
        # copy of an ant's work lives in memory until the run returns, so a run
        # that is killed, crashes, or dies on its budget takes the work with it.
        self.patch_dir = patch_dir
        self._runner = runner or _sdk_runner
        self._custom_shell = shell is not None
        self._shell = shell or self._subprocess
        self._lock = threading.Lock()
        self._worktree_root: str | None = None
        self.patches: list[Patch] = []
        self.landed: list[Patch] = []
        self.discarded: list[Patch] = []
        self.task_id: str = ""      # set on a session, for patch attribution
        # The agent's work, captured before the check ran. Running the check
        # creates files of its own — .pyc, coverage data, build output — and
        # those are the *check's* side effects, not the agent's change. A patch
        # carrying them is bigger, unattributable, and often will not apply.
        self.pending_patch: Patch | None = None
        # Held open for as long as this run lives, and never closed by hand:
        # closing it is what tells a later run the root is abandoned.
        self._owner_lock: Any = None
        self.produced_by: str = ""  # which brief is working in this session
        # A session points back at the repository it was cut from; the main
        # workspace points at itself. A tactic that must act on the *real* tree
        # (landing a patch) reaches it through here.
        self.root: AgentWorkspace = self

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
        child = self._add_worktree(prefix="ant")
        child.task_id = str(getattr(task, "id", "") or "")
        return child

    def _add_worktree(self, *, prefix: str = "wt") -> "AgentWorkspace":
        """A detached checkout of HEAD, as a workspace in its own right."""
        with self._lock:
            if self._worktree_root is None:
                # Outside the repo, or git would treat it as untracked content.
                self._worktree_root = tempfile.mkdtemp(prefix=WORKTREE_ROOT_PREFIX)
                self._claim(self._worktree_root)
            path = os.path.join(self._worktree_root, f"{prefix}-{uuid.uuid4().hex[:8]}")
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
        child.root = self
        return child

    def trial(self, patch: Patch) -> PatchTrial:
        """Apply a candidate patch to a scratch checkout of HEAD and measure it.

        This is what makes choosing between patches an act of measurement rather
        than an opinion. A patch that passed in the tree it was born in may still
        fail here — HEAD has moved, or another patch landed first — and that is
        precisely what a selector needs to know. Nothing touches the real tree.
        """
        try:
            scratch = self._add_worktree(prefix="trial")
        except RuntimeError as exc:
            return PatchTrial(applies=False, passes=False, detail=str(exc))
        try:
            applied, out = scratch.apply_patch(patch)
            if not applied:
                return PatchTrial(applies=False, passes=False, detail=out.strip()[-300:])
            passes, output = scratch.verify()
            return PatchTrial(applies=True, passes=passes,
                              detail="" if passes else output.strip()[-300:])
        finally:
            self.run(["git", "worktree", "remove", "--force", scratch.path])

    def land(self, patch: Patch) -> None:
        """Record a patch as landed and retire the alternatives.

        The patches of a fan-out are competing answers to the *same* goal, so once
        one is in, applying another would stack a second implementation on top of
        the first. The rest are moved to ``discarded`` rather than deleted — a
        losing candidate is still evidence.
        """
        with self._lock:
            self.landed.append(patch)
            self.discarded.extend(p for p in self.patches if p is not patch)
            self.patches = []

    def release(self, session: "Target") -> None:
        """Lift the work out as a patch, archive it, then remove the worktree.

        The order is the point. Until this runs, the only copy of an ant's work
        is the worktree that is about to be destroyed; after it, the only copy
        is a list in memory that a killed run never returns. So the patch is
        written to ``patch_dir`` *before* the worktree goes, and the run has to
        survive nothing in particular for the work to be recoverable.
        """
        if session is self or not isinstance(session, AgentWorkspace):
            return
        patch = session.pending_patch or session.capture_patch()
        if patch:
            with self._lock:
                self.patches.append(patch)
                index = len(self.patches)
            self._save_patch(patch, index)
        self.run(["git", "worktree", "remove", "--force", session.path])

    def _save_patch(self, patch: Patch, index: int) -> None:
        """Write one patch to ``patch_dir``. Never costs the caller its work.

        Fail-soft on purpose: an unwritable directory is a lost archive copy,
        and turning that into an exception here would lose the patch itself —
        the opposite of the point. ``saved_to`` stays empty so a caller can say
        so rather than implying a file exists.
        """
        if not self.patch_dir:
            return
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", patch.tactic or patch.task or "patch").strip("-")
        try:
            os.makedirs(self.patch_dir, exist_ok=True)
            dest = os.path.join(self.patch_dir, f"{index:03d}-{stem or 'patch'}.patch")
            # Never clobber: a second run pointed at the same directory is a
            # second set of answers, not a correction of the first.
            if os.path.exists(dest):
                dest = f"{dest[:-6]}-{uuid.uuid4().hex[:6]}.patch"
            text = patch.text if patch.text.endswith("\n") else patch.text + "\n"
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            return
        patch.saved_to = dest

    def capture_patch(self) -> Patch:
        """The full diff of this workspace, including files the agent created."""
        self.run(["git", "add", "-A", "-N"])  # intent-to-add: untracked files show up
        return Patch(task=self.task_id, text=self.diff_text(), files=self.changed_files(),
                     tactic=self.produced_by)

    def apply_patch(self, patch: Patch) -> tuple[bool, str]:
        """Land a patch on this repository. Irreversible enough to gate.

        Concurrent ants can produce patches that touch the same lines; ``--3way``
        resolves what it can and fails loudly on the rest rather than mangling
        the tree. Note that ``--3way`` *stages* what it applies, so a landed
        patch shows up under ``git diff --cached`` rather than ``git diff`` — the
        change is ready to review and commit, not silently in the working tree.
        """
        handle, tmp = tempfile.mkstemp(suffix=".patch")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(patch.text if patch.text.endswith("\n") else patch.text + "\n")
            code, out = self.run(["git", "apply", "--3way", tmp])
            return code == 0, out
        finally:
            os.unlink(tmp)

    def _claim(self, root: str) -> None:
        """Take an advisory lock on this run's worktree root.

        Not for mutual exclusion — each run makes its own root — but so that a
        later run can tell an abandoned root from one still in use. The handle
        is deliberately kept open; the lock lasts exactly as long as the process.
        """
        if fcntl is None:   # pragma: no cover - Windows
            return
        try:
            handle = open(os.path.join(root, OWNER_LOCK), "w", encoding="utf-8")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:     # pragma: no cover - an unlockable temp dir
            return
        self._owner_lock = handle

    def reap_abandoned_worktrees(self) -> list[str]:
        """Remove this repo's worktrees left behind by a run that was killed.

        A run that dies without unwinding — ``kill -9``, a lost machine, an OOM
        — leaves its worktrees both registered and on disk, and ``git worktree
        prune`` will not touch them precisely because the directories are still
        there. So they accumulate, one full checkout at a time, until somebody
        notices.

        Two things are never reaped: a worktree the user added themselves (only
        paths under a ``tactics-worktrees-*`` root are considered), and a root
        another live run still holds the lock on — so running two agents against
        one repo at the same time stays safe.
        """
        self.run(["git", "worktree", "prune"])      # entries whose dirs are gone
        roots: dict[str, list[str]] = {}
        for path in self._registered_worktrees():
            root = _worktree_root_of(path)
            if root:
                roots.setdefault(root, []).append(path)

        reaped: list[str] = []
        for root in sorted(roots):
            if root == self._worktree_root or not _is_abandoned(root):
                continue
            for path in roots[root]:
                code, _out = self.run(["git", "worktree", "remove", "--force", path])
                if code == 0:
                    reaped.append(path)
            shutil.rmtree(root, ignore_errors=True)
        if reaped:
            self.run(["git", "worktree", "prune"])
        return reaped

    def _registered_worktrees(self) -> list[str]:
        """Every worktree git has recorded for this repository, bar the main one."""
        code, out = self.run(["git", "worktree", "list", "--porcelain"])
        if code != 0:
            return []
        paths = [line[len("worktree "):].strip()
                 for line in out.splitlines() if line.startswith("worktree ")]
        return [p for p in paths if os.path.abspath(p) != os.path.abspath(self.path)]

    def cleanup(self) -> None:
        """Remove every worktree this workspace created. Safe to call twice."""
        with self._lock:
            root, self._worktree_root = self._worktree_root, None
            handle, self._owner_lock = self._owner_lock, None
        if not root:
            return
        for name in sorted(os.listdir(root)):
            child = os.path.join(root, name)
            if os.path.isdir(child):    # the lock file is not a worktree
                self.run(["git", "worktree", "remove", "--force", child])
        self.run(["git", "worktree", "prune"])
        if handle is not None:
            handle.close()
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
        # `git diff HEAD`, not `git diff`: an agent is free to stage its own work
        # (a `git add` through Bash is an ordinary thing to do), and plain
        # `git diff` shows only what is *un*staged — so the work would come back
        # as an empty diff and be silently dropped.
        code, out = self.run(["git", "diff", "HEAD"])
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

    def __init__(
        self,
        *,
        name: str | None = None,
        lessons: LessonStore | None = None,
        lesson_limit: int = 6,
        lesson_budget: int = 1500,
    ) -> None:
        super().__init__(name=name)
        # Verbal memory: what past runs learned, prepended to every brief. This is
        # the half a fresh agent process cannot have — its context starts empty
        # every time, however good the harness is.
        self.lessons = lessons
        self.lesson_limit = lesson_limit
        # A ceiling on how much recalled text may precede the task. The store is
        # append-only, so without one it grows forever and every brief pays. This
        # is a guard against unbounded growth, *not* a fix for the length effect
        # measured in docs/experiments/ — that one is about how prescriptive a
        # lesson is, and the lesson that caused it would fit inside this budget.
        self.lesson_budget = lesson_budget

    def build_brief(self, ctx: Any) -> str:
        """The task itself. Override to shape *how* it is asked."""
        return str(ctx.task.description if ctx.task is not None else ctx.goal.description)

    def _with_lessons(self, brief: str, ctx: Any) -> str:
        """Prepend relevant past lessons — the same seam as ``LLMTactic``.

        Lessons go in the brief, not the system prompt: the system prompt *is*
        the brief shape being measured, and quietly varying it would make two
        runs of the same tactic incomparable.
        """
        if self.lessons is None:
            return brief
        relevant = self.lessons.relevant(
            playbook=getattr(ctx.target, "name", None),
            goal=getattr(ctx.goal, "name", None) if ctx.goal else None,
            query=brief,
            limit=self.lesson_limit,
        )
        kept, spent = [], 0
        for lesson in relevant:  # most relevant first, so the tail is what drops
            spent += len(lesson.text)
            if kept and spent > self.lesson_budget:
                break
            kept.append(lesson)
        journal = getattr(ctx, "journal", None)
        if journal is not None:
            # How many lessons actually reached the brief. A store that is wired
            # but empty — a wrong path, a wiped directory, a store that was never
            # populated — is otherwise indistinguishable from having nothing to
            # say, and produces a run that looks like evidence and is not.
            journal.record("lessons.recalled", tactic=self.name,
                           count=len(kept), dropped=len(relevant) - len(kept))
        block = render_lessons(kept, header="What past runs on this repository learned:")
        return f"{block}\n\n{brief}" if block else brief

    def execute(self, ctx: Any) -> Outcome:
        broken = _roster_error(self.spec)
        if broken:  # never spend money on a brief that cannot do its job
            return Outcome(success=False, reward=0.0, notes=f"misconfigured brief: {broken}")
        bridge = GateBridge(ctx, self.spec.allowed_tools)
        # Stamp the session so the patch it yields says which brief wrote it.
        setattr(ctx.target, "produced_by", self.name)
        before_snapshot = ctx.target.snapshot()
        before_files = set(ctx.target.changed_files())
        brief = self._with_lessons(self.build_brief(ctx), ctx)
        run = ctx.target.run_agent(brief, self.spec, bridge)
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

        # Measure first, and unconditionally. A run that errored — a turn limit,
        # a dropped connection — may still have done the work, and scoring it 0
        # without looking is exactly the self-report this playbook refuses to
        # trust, only inverted. The error is context for the notes; the tree and
        # the check decide the reward.
        changed = sorted(set(ctx.target.changed_files()) - before_files)
        metrics["changed_files"] = len(changed)
        if journal is not None:
            journal.record("agent.run", tactic=self.name, **metrics, error=run.error[:120])
        aside = f" (the run also errored: {run.error[:120]})" if run.error else ""

        if ctx.target.snapshot() == before_snapshot:
            if run.error:
                return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                               notes=f"agent run failed with no work done: {run.error[:200]}")
            held = f" ({len(run.denied)} tool call(s) held by the gate)" if run.denied else ""
            return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                           notes=f"agent left the working tree untouched{held}")

        # Capture before verifying: after this line the check will litter the
        # worktree with its own artifacts, and they are not the agent's work.
        ctx.target.pending_patch = ctx.target.capture_patch()

        passed, output = ctx.target.verify()
        if journal is not None and not passed:
            # The *reason* a check failed, in the audit trail rather than only in
            # this Outcome's notes. Without it the journal says a run failed but
            # never what broke, so a failure that recurs every round looks like
            # three anonymous zeroes — and nothing downstream, the scribe least of
            # all, can tell a repeating cause from a run of bad luck.
            journal.record("check.failed", tactic=self.name, detail=output.strip()[-400:])
        return Outcome(
            success=passed,
            reward=1.0 if passed else 0.0,
            cost=cost,
            metrics=metrics,
            notes=(f"check passed, {len(changed)} file(s) changed{aside}"
                   if passed else f"check failed: {output.strip()[-300:]}{aside}"),
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


_JUDGE_SYSTEM = (
    "You choose between candidate patches that have ALL already been verified: "
    "each one applies cleanly and passes the repository's own check. Your job is "
    "the remaining question — which is the better change to keep. Prefer the "
    "smallest change that fully does the job, code that matches the surrounding "
    "style, and tests that would catch a real regression. Respond ONLY with the "
    "requested JSON object."
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer"},
        "why": {"type": "string"},
    },
    "required": ["choice", "why"],
    "additionalProperties": False,
}


class ApplyBestPatch(Tactic):
    """Pick among the patches a fan-out produced, and land one through the gate.

    A parallel round leaves several competing answers to the same goal in
    ``target.patches``, each verified in the worktree it was born in. Choosing
    between them by hand is the last manual step in the loop; this closes it.

    **The choice is measured before it is judged.** Every candidate is re-tried
    against a scratch checkout of the *current* HEAD — a patch that passed where
    it was written can still fail here, because HEAD moved or because an earlier
    patch landed first, and that is exactly what a selector needs to know.
    Candidates that fail are out, with a reason. Only among the survivors does
    anything softer apply: the deterministic tie-break is the smallest verified
    diff, and an optional ``judge`` (any ``LLMClient``) may re-order *those* — it
    is shown them already in that ranked order, so its answer is a considered
    override of a defensible default rather than a choice made from nothing.

    The judge can never promote a failing patch, and it fails closed: an answer
    that is unparseable, or that names a candidate outside the verified set, is
    discarded in favour of the deterministic pick. A model's opinion is allowed
    to break a tie between proven options; it is never allowed to be the proof.

    Landing writes the real repository, so it goes through ``ctx.gate``. Under
    ``DryRun`` the whole selection still runs and the journal records which patch
    *would* have landed and why — a review artifact rather than a write.
    """

    def __init__(self, *, judge: Any = None, name: str | None = None) -> None:
        super().__init__(name=name)
        self.judge = judge

    def _repo(self, ctx: Any) -> "AgentWorkspace":
        # Reach past an isolated session: landing is a change to the real tree.
        return getattr(ctx.target, "root", ctx.target)

    def is_applicable(self, ctx: Any) -> bool:
        return bool(getattr(self._repo(ctx), "patches", None))

    # --- choosing -------------------------------------------------------------

    @staticmethod
    def _rank(candidate: tuple[Patch, PatchTrial]) -> tuple[int, int, str]:
        """Smallest verified change first; ties broken deterministically."""
        patch, _trial = candidate
        return (len(patch.text.splitlines()), len(patch.files), patch.task)

    def _ask_judge(self, survivors: list[tuple[Patch, PatchTrial]], ctx: Any) -> tuple[int | None, float, str]:
        listing = "\n\n".join(
            f"### Candidate {i}\nwritten by: {patch.tactic or '(unknown brief)'}\n"
            f"files: {patch.files}\n```diff\n{patch.text[:4000]}\n```"
            for i, (patch, _t) in enumerate(survivors)
        )
        prompt = (
            f"{len(survivors)} candidate patches all apply cleanly and pass the check. "
            f"Choose the one to keep.\n\n{listing}\n\n"
            'Reply as JSON: {"choice": <candidate number>, "why": "<one sentence>"}.'
        )
        try:
            resp = self.judge.complete(prompt, system=_JUDGE_SYSTEM, schema=_JUDGE_SCHEMA,
                                       max_tokens=1024)
            data = extract_json(resp.text)
            choice = int(data.get("choice", -1))
        except Exception as exc:  # noqa: BLE001 - a bad judgment is not a crash
            return None, 0.0, f"judge unusable ({exc!r})"
        cost = float(getattr(resp, "tokens", 0))
        if not 0 <= choice < len(survivors):
            return None, cost, f"judge named candidate {choice}, which is not on the list"
        return choice, cost, str(data.get("why", ""))[:200]

    def execute(self, ctx: Any) -> Outcome:
        repo = self._repo(ctx)
        journal = getattr(ctx, "journal", None)
        candidates = list(repo.patches)

        trials = [(patch, repo.trial(patch)) for patch in candidates]
        survivors = [(patch, trial) for patch, trial in trials if trial.ok]
        metrics = {"candidates": len(candidates), "verified": len(survivors)}
        if journal is not None:
            for patch, trial in trials:
                journal.record("patch.trial", tactic=patch.tactic, files=len(patch.files),
                               applies=trial.applies, passes=trial.passes,
                               detail=trial.detail[:120])

        if not survivors:
            why = "; ".join(
                f"{p.tactic or p.task}: {'does not apply' if not t.applies else 'check fails'}"
                for p, t in trials
            )
            return Outcome(success=False, reward=0.0, metrics=metrics,
                           notes=f"no candidate survived re-verification — {why}")

        survivors.sort(key=self._rank)
        chosen, _trial = survivors[0]
        reason = "smallest verified diff"
        cost = 0.0
        if self.judge is not None and len(survivors) > 1:
            pick, cost, why = self._ask_judge(survivors, ctx)
            if pick is None:
                reason = f"deterministic — {why}"  # fail closed onto the measured order
            else:
                chosen, _trial = survivors[pick]
                reason = f"judge: {why}"
        metrics["chose"] = chosen.tactic or chosen.task

        result = ctx.gate.submit(
            Proposal(
                action=f"land patch from {chosen.tactic or chosen.task} "
                       f"({len(chosen.files)} file(s), {len(chosen.text.splitlines())} diff lines)",
                commit=lambda: repo.apply_patch(chosen),
                reversible=True,  # a working-tree change, revertible with git
                risk="medium",    # but it is the real repository, not a worktree
                detail={"reason": reason, "files": chosen.files,
                        "rejected": len(candidates) - 1},
            ),
            ctx,
        )
        if not result.committed:
            return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                           notes=f"selected {metrics['chose']} ({reason}) but the gate held it")

        applied, out = result.result
        if not applied:  # verified in a scratch tree yet refused by the real one
            return Outcome(success=False, reward=0.0, cost=cost, metrics=metrics,
                           notes=f"apply failed after a clean trial: {out.strip()[-200:]}")
        repo.land(chosen)
        return Outcome(success=True, reward=1.0, cost=cost, metrics=metrics,
                       notes=f"landed {metrics['chose']} ({reason}); "
                             f"{len(candidates) - 1} alternative(s) discarded")


def land_best_patch(target: "AgentWorkspace", *, gate: Any = None, judge: Any = None,
                    journal: Any = None, goal: Goal | None = None) -> Outcome:
    """Run :class:`ApplyBestPatch` once, outside a colony. Returns its Outcome."""
    from ..core.approval import AutoApprove
    from ..core.journal import Journal

    ctx = Context(
        target=target, goal=goal or Goal(name="land_patch"), data={}, features={},
        gate=gate or AutoApprove(), journal=journal or Journal(),
    )
    tactic = ApplyBestPatch(judge=judge)
    if not tactic.is_applicable(ctx):
        return Outcome(success=False, reward=0.0, notes="no candidate patches to choose from")
    return tactic.execute(ctx)


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
        # The task here is "make this work", not "have a go at it": a verified
        # failure is trusted and learned from, and goes back on the board.
        return Verdict(
            accepted=True,
            done=outcome.success,
            reason="check re-run confirms the fix" if passed
                   else "check re-run confirms it is still failing",
        )

    return FunctionCritic(verify)


def brief_memory(repo: str, *, subdir: str = STORE_DIR) -> JsonStore:
    """Numeric memory that survives the process: which brief wins, and where.

    Without this every run starts cold and the policy re-learns from scratch —
    which is exactly the gap a static roster of prompts has, and the reason to
    have a policy at all.
    """
    return JsonStore(os.path.join(repo, subdir, MEMORY_FILE))


def brief_lessons(repo: str, *, subdir: str = STORE_DIR) -> JsonlLessons:
    """Verbal memory: append-only JSONL, greppable, and editable by hand —
    delete a line to retract a lesson that turned out to be wrong."""
    return JsonlLessons(os.path.join(repo, subdir, LESSONS_FILE))


class BriefScribe(Scribe):
    """A scribe that sees the brief scoreboard, not just the journal.

    The generic Scribe reads the journal and findings, which for this playbook
    describe *what happened* but not *which brief it happened to*. The durable
    lesson is almost always comparative — "PlanThenPatch wins on unfamiliar code
    and wastes turns on one-liners" — so the standings go in the evidence.
    """

    def __init__(self, client, store, *, memory: MemoryStore | None = None, **kw) -> None:  # noqa: ANN001
        super().__init__(client, store, **kw)
        self.memory = memory

    def scoreboard(self) -> str:
        if self.memory is None:
            return "(no recorded standings)"
        rows = sorted(
            self.memory.entries(),
            key=lambda e: (e.stats.mean_reward, e.stats.trials),
            reverse=True,
        )
        if not rows:
            return "(no recorded standings)"
        return "\n".join(
            f"- {e.tactic}: {e.stats.trials} run(s), mean reward "
            f"{e.stats.mean_reward:.2f}, situation {e.features}"
            for e in rows[:12]
        )

    def build_prompt(self, result, playbook: str | None) -> str:  # noqa: ANN001
        return (
            f"{super().build_prompt(result, playbook)}\n\n"
            f"Brief standings so far (numeric memory):\n{self.scoreboard()}\n\n"
            "Prefer lessons that would change which brief a future run picks, or how "
            "a brief is written. A lesson that merely restates a result is not durable.\n"
            "Name the cause; do not prescribe a procedure. A future run reads these on a "
            "budget, and a lesson that sends it investigating spends the budget it needed "
            "to write the fix — measured in docs/experiments/: a prescriptive lesson cut "
            "wrong fixes hardest and still lost, because it pushed 45 of 150 runs into "
            "turn exhaustion with nothing written."
        )


def run_and_learn(colony: Colony, goal: Goal, *, client: Any = None, lessons: LessonStore | None = None):
    """Run the colony, then write down what it learned. The compounding loop.

    Numeric memory records itself as the colony runs; the *words* — why a brief
    won, what the gate kept stopping, which repo quirk cost three rounds —
    evaporate with the journal unless something distills them. That is this.

    Without a ``client`` the run still happens and numeric memory still persists;
    only the verbal half is skipped. A failed distillation writes nothing at all,
    because one fabricated lesson pollutes every future brief that recalls it.
    """
    result = colony.run(goal)
    store = lessons or next(
        (t.lessons for t in colony.tactics if getattr(t, "lessons", None) is not None), None
    )
    if client is None or store is None:
        # Say so in the trail rather than returning an empty list that looks
        # like "the scribe considered it and declined".
        journal = getattr(result, "journal", None)
        if journal is not None:
            journal.record("scribe.skipped",
                           reason="no client" if client is None else "no lesson store")
        return result, []
    written = BriefScribe(client, store, memory=colony.memory).distill(
        result, playbook=getattr(colony.target, "name", None)
    )
    return result, written


def build_delivery_colony(
    target: AgentWorkspace,
    *,
    tactics: list[Tactic] | None = None,
    memory: MemoryStore | None = None,
    lessons: LessonStore | None = None,
    persist: bool | str = False,
    policy: Policy | None = None,
    spread: bool = True,
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

    ``spread`` (on by default) makes the ants of a parallel round try *different*
    briefs. Without it the policy is consulted once per ant against one memory
    snapshot, so all three reach the same conclusion and a three-agent round buys
    three samples of one brief instead of one each of three — throughput without
    information, exactly when the briefs are what you're comparing. Turn it off
    to spend a parallel round reducing variance on the current best instead.

    ``persist=True`` (or a directory path) keeps both halves of memory on disk
    under ``<repo>/.tactics/``: the numeric record of which brief wins where, and
    the lessons past runs wrote. Both are then wired in automatically — the store
    into the policy, the lessons into every brief. The default is off, because
    writing into someone's repository should be asked for, not assumed.
    """
    if max_workers > 1 and not getattr(target, "isolate", False):
        raise ValueError(
            "max_workers > 1 needs AgentWorkspace(isolate=True): parallel agents "
            "sharing one working tree make every result unattributable"
        )
    if persist:
        root = persist if isinstance(persist, str) else getattr(target, "path", ".")
        memory = memory or brief_memory(root)
        lessons = lessons or brief_lessons(root)

    policy = policy or UCBPolicy()
    if spread and max_workers > 1:
        policy = WithoutReplacement(policy)

    tactics = tactics or [
        SingleAgentNarrow(lessons=lessons),
        WriteTestFirst(lessons=lessons),
        PlanThenPatch(lessons=lessons),
        ReviewedSwarm(lessons=lessons),
    ]
    if lessons is not None:
        # Caller-supplied tactics get the store too. Otherwise `persist=True`
        # quietly means "persist numbers only" the moment you pass your own
        # roster, and the verbal half goes missing without a word.
        for tactic in tactics:
            if getattr(tactic, "lessons", "unset") is None:
                tactic.lessons = lessons

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
        policy=policy,
        critic=verification_critic(),
        gate=gate,
        budget=budget,
        max_workers=max_workers,
        max_rounds=max_rounds,
    )
