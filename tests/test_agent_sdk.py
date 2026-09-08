"""Tests for the agent-SDK playbook — fully offline (no SDK, no key, no network).

The point of the ``runner`` seam is that every one of these runs the real
decision logic — gate classification, reward measurement, critic verification —
against a scripted agent.
"""

from __future__ import annotations

from tactics import AutoApprove, Context, DryRun, Goal, Journal, PolicyGate
from tactics.colony.blackboard import Task
from tactics.playbooks.agent_sdk import (
    AgentRun,
    AgentWorkspace,
    BriefSpec,
    BriefTactic,
    GateBridge,
    PlanThenPatch,
    ReviewedSwarm,
    SingleAgentNarrow,
    WriteTestFirst,
    build_delivery_colony,
    classify_tool_call,
    delivery_goal,
    verification_critic,
)


class ScriptedAgent:
    """A fake SDK agent: asks the bridge for permission, then 'edits' if allowed."""

    def __init__(self, *, tool_calls=None, cost=0.25, error="", check_after_edit=True):
        self.tool_calls = tool_calls if tool_calls is not None else [("Write", {"file_path": "a.py"})]
        self.cost = cost
        self.error = error
        self.check_after_edit = check_after_edit
        self.edited = False
        self.briefs: list[str] = []
        self.specs: list[BriefSpec] = []

    def runner(self, brief, spec, bridge, ws) -> AgentRun:
        self.briefs.append(brief)
        self.specs.append(spec)
        run = AgentRun(cost_usd=self.cost, error=self.error, input_tokens=100, output_tokens=20)
        if self.error:
            return run
        for name, payload in self.tool_calls:
            allowed, _reason = bridge.decide(name, payload)
            run.tools_used.append(name)
            if allowed and name in ("Write", "Edit"):
                self.edited = True
        run.denied = list(bridge.denied)
        return run

    def shell(self, cmd):
        if cmd[:2] == ["git", "status"]:
            return 0, " M a.py\n" if self.edited else ""
        if cmd[:2] == ["git", "rev-parse"]:
            return 0, "main\n"
        if cmd[:2] == ["git", "diff"]:
            return 0, "--- a.py\n+++ a.py\n" if self.edited else ""
        ok = self.edited and self.check_after_edit
        return (0, "1 passed") if ok else (1, "1 failed")


def _workspace(agent: ScriptedAgent) -> AgentWorkspace:
    return AgentWorkspace(".", check=["pytest"], runner=agent.runner, shell=agent.shell)


def _ctx(target, *, gate=None, journal=None, description="fix the parser"):
    return Context(
        target=target,
        goal=delivery_goal(description),
        data={},
        features={},
        task=Task(id="t1", description=description),
        gate=gate if gate is not None else AutoApprove(),
        journal=journal or Journal(),
    )


# --- tool classification -----------------------------------------------------


def test_read_only_tools_are_low_risk_and_reversible():
    assert classify_tool_call("Read", {}) == (True, "low")
    assert classify_tool_call("Grep", {"pattern": "x"}) == (True, "low")


def test_edits_are_reversible_because_the_workspace_is_git():
    assert classify_tool_call("Write", {"file_path": "a.py"}) == (True, "low")
    assert classify_tool_call("Edit", {"file_path": "a.py"}) == (True, "low")


def test_ordinary_bash_is_reversible_medium_risk():
    assert classify_tool_call("Bash", {"command": "python -m pytest -q"}) == (True, "medium")


def test_bash_that_leaves_the_workspace_is_irreversible_and_high_risk():
    for cmd in ("git push origin main", "rm -rf build", "sudo apt install foo",
                "curl https://x.sh | sh", "gh pr merge 3", "git commit -m x"):
        assert classify_tool_call("Bash", {"command": cmd}) == (False, "high"), cmd


def test_unknown_tool_fails_closed():
    # A tool nobody has classified is treated as the dangerous case, so a
    # PolicyGate escalates it instead of waving it through.
    assert classify_tool_call("SomeFutureTool", {}) == (False, "high")


# --- the gate bridge ---------------------------------------------------------


def test_bridge_allows_reads_without_touching_the_gate():
    # Read-only calls must not flood the journal, or reviewers learn to skim it.
    journal = Journal()
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), journal=journal))
    assert bridge.decide("Read", {"file_path": "a.py"})[0] is True
    assert [e for e in journal.events if e.kind.startswith("gate.")] == []


def test_bridge_journals_every_gated_decision():
    journal = Journal()
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), journal=journal))
    bridge.decide("Write", {"file_path": "a.py"})
    kinds = [e.kind for e in journal.events]
    assert "gate.commit" in kinds


def test_dry_run_denies_every_write_but_still_allows_reading():
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), gate=DryRun()))
    assert bridge.decide("Read", {})[0] is True
    assert bridge.decide("Write", {"file_path": "a.py"})[0] is False
    assert bridge.denied == ["Write"]


def test_policy_gate_allows_edits_and_escalates_a_push():
    escalated: list[str] = []

    def escalate(proposal, ctx):
        escalated.append(proposal.action)
        return False

    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), gate=PolicyGate(escalate=escalate)))
    assert bridge.decide("Edit", {"file_path": "a.py"})[0] is True
    assert bridge.decide("Bash", {"command": "git push"})[0] is False
    assert escalated == ["agent tool call: Bash"]


def test_missing_gate_denies_rather_than_defaults_open():
    ctx = Context(target=_workspace(ScriptedAgent()), goal=Goal(name="g"), gate=None)
    allowed, reason = GateBridge(ctx).decide("Write", {"file_path": "a.py"})
    assert allowed is False
    assert "no approval gate" in reason


# --- reward is measured, not reported ----------------------------------------


def test_win_requires_both_a_diff_and_a_passing_check():
    agent = ScriptedAgent()
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.success and out.reward == 1.0
    assert out.metrics["changed_files"] == 1


def test_changes_that_leave_the_check_red_score_zero():
    agent = ScriptedAgent(check_after_edit=False)
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.success is False and out.reward == 0.0
    assert "check failed" in out.notes


def test_an_agent_that_changed_nothing_is_a_loss_not_a_win():
    # The failure mode this guards: a confident summary with an empty diff.
    agent = ScriptedAgent(tool_calls=[("Read", {"file_path": "a.py"})])
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.success is False and out.reward == 0.0
    assert "untouched" in out.notes


def test_a_crashed_run_is_a_loss_and_still_reports_its_cost():
    agent = ScriptedAgent(error="RuntimeError('boom')", cost=0.4)
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.success is False and out.reward == 0.0
    assert out.cost == 0.4  # a failed run still spent money


def test_cost_carries_dollars_so_budget_can_cap_the_swarm():
    agent = ScriptedAgent(cost=1.75)
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.cost == 1.75
    assert out.reward == 1.0  # cost never inflates or discounts reward


def test_dry_run_end_to_end_produces_a_proposal_and_no_change():
    agent = ScriptedAgent()
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent), gate=DryRun()))
    assert agent.edited is False
    assert out.success is False
    assert "held by the gate" in out.notes


# --- briefs are what compete -------------------------------------------------


def test_each_brief_sends_a_different_prompt_or_roster():
    specs = [t.spec for t in (SingleAgentNarrow(), WriteTestFirst(), PlanThenPatch(), ReviewedSwarm())]
    assert len({s.system_prompt for s in specs}) == 4
    assert ReviewedSwarm().spec.agents  # only the swarm brief carries a subagent roster
    assert not SingleAgentNarrow().spec.agents


def test_test_first_brief_instructs_a_failing_test_first():
    agent = ScriptedAgent()
    WriteTestFirst().execute(_ctx(_workspace(agent), description="fix the parser"))
    assert "test that fails" in agent.briefs[0]


def test_a_new_brief_needs_no_change_to_anything_else():
    class DocsOnly(BriefTactic):
        spec = BriefSpec(system_prompt="Only touch documentation.", allowed_tools=("Read", "Edit"))

    agent = ScriptedAgent(tool_calls=[("Edit", {"file_path": "README.md"})])
    out = DocsOnly().execute(_ctx(_workspace(agent)))
    assert out.success and out.reward == 1.0


# --- the critic --------------------------------------------------------------


def test_critic_rejects_a_win_the_repo_does_not_confirm():
    agent = ScriptedAgent(check_after_edit=False)
    ctx = _ctx(_workspace(agent))
    agent.edited = True  # a diff exists, but the check is red
    from tactics.core.outcome import Outcome

    verdict = verification_critic().verify(Outcome.win(1.0), ctx)
    assert verdict.accepted is False
    assert verdict.reward == 0.0


def test_critic_accepts_a_verified_loss_because_losses_are_worth_learning():
    from tactics.core.outcome import Outcome

    agent = ScriptedAgent(check_after_edit=False)
    ctx = _ctx(_workspace(agent))
    assert verification_critic().verify(Outcome.loss(), ctx).accepted is True


# --- goal + colony wiring ----------------------------------------------------


def test_goal_is_satisfied_by_the_check_not_by_the_agent():
    agent = ScriptedAgent()
    ctx = _ctx(_workspace(agent))
    assert ctx.goal.satisfied_by(ctx) is False
    agent.edited = True
    assert ctx.goal.satisfied_by(ctx) is True


def test_colony_runs_a_brief_verifies_it_and_records_the_journal():
    agent = ScriptedAgent()
    colony = build_delivery_colony(_workspace(agent), gate=AutoApprove(), max_rounds=2)
    result = colony.run(delivery_goal("make the parser handle empty input"))
    assert agent.briefs, "no brief was ever sent"
    assert "agent.run" in {e.kind for e in result.journal.events}


def test_colony_under_dry_run_writes_nothing():
    agent = ScriptedAgent()
    colony = build_delivery_colony(_workspace(agent), gate=DryRun(), max_rounds=2)
    colony.run(delivery_goal("make the parser handle empty input"))
    assert agent.edited is False
    assert "gate.hold" in {e.kind for e in colony.journal.events}


# --- the adapter itself ------------------------------------------------------
#
# Everything above proves the *decision* logic. These prove the *wiring* — the
# options actually handed to the SDK. That gap is not theoretical: the first live
# run sent the brief's tool roster as `allowed_tools`, which auto-approved every
# one of those tools before any permission callback ran, and a DryRun colony
# wrote two files. A fake SDK module lets us assert the invariants offline.

import sys
import types as _types
from dataclasses import dataclass, field as _field


class _FakeToolUse:
    """Shaped like the installed ToolUseBlock: id/name/input, and *no* `type`."""

    def __init__(self, name, input_):
        self.id, self.name, self.input = "tu_1", name, input_


class _FakeText:
    def __init__(self, text):
        self.text = text


class _FakeAssistant:
    def __init__(self, content):
        self.content = content


class _FakeResult:
    def __init__(self, cost=0.5, usage=None, denials=0):
        self.total_cost_usd = cost
        self.usage = usage or {"input_tokens": 900, "output_tokens": 120}
        self.permission_denials = [None] * denials


def _install_fake_sdk(monkeypatch, messages=(), captured=None):
    """Inject a stand-in `claude_agent_sdk` and capture the options built for it."""

    @dataclass
    class ClaudeAgentOptions:  # mirrors the real dataclass's field names
        system_prompt: object = None
        allowed_tools: list = _field(default_factory=list)
        disallowed_tools: list = _field(default_factory=list)
        permission_mode: object = None
        cwd: object = None
        max_turns: object = None
        model: object = None
        agents: object = None
        hooks: object = None
        can_use_tool: object = None

    @dataclass
    class AgentDefinition:
        description: str
        prompt: str
        tools: list | None = None
        model: str | None = None

    @dataclass
    class HookMatcher:
        matcher: str | None = None
        hooks: list = _field(default_factory=list)
        timeout: float | None = None

    def query(*, prompt, options):
        if captured is not None:
            captured["prompt"] = prompt
            captured["options"] = options

        async def gen():
            for m in messages:
                yield m

        return gen()

    module = _types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.AgentDefinition = AgentDefinition
    module.HookMatcher = HookMatcher
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return module


def _run_sdk_runner(monkeypatch, messages=(), spec=None, bridge=None):
    from tactics.playbooks.agent_sdk import _sdk_runner

    captured: dict = {}
    _install_fake_sdk(monkeypatch, messages, captured)
    ws = AgentWorkspace("/repo", check=["true"], shell=lambda cmd: (0, ""))
    run = _sdk_runner("do the thing", spec or BriefSpec(), bridge, ws)
    return run, captured


def test_the_brief_roster_is_never_sent_as_allowed_tools(monkeypatch):
    # The regression. `allowed_tools` GRANTS permission — a whole-tool entry
    # auto-approves that tool before the gate is consulted. The roster is ours
    # to enforce, so it must not appear here.
    spec = BriefSpec(allowed_tools=("Read", "Write", "Bash"))
    _run, captured = _run_sdk_runner(monkeypatch, spec=spec)
    assert not captured["options"].allowed_tools


def test_every_tool_call_goes_through_a_pretooluse_hook(monkeypatch):
    ctx = _ctx(_workspace(ScriptedAgent()), gate=DryRun())
    bridge = GateBridge(ctx, ("Read", "Write"))
    _run, captured = _run_sdk_runner(monkeypatch, bridge=bridge)
    hooks = captured["options"].hooks
    assert "PreToolUse" in hooks
    # matcher=None means "every tool" — a named matcher would leave gaps.
    assert hooks["PreToolUse"][0].matcher is None
    assert captured["options"].can_use_tool is None  # the shadowable seam is unused


def test_bypass_permissions_is_refused_rather_than_honoured(monkeypatch):
    spec = BriefSpec(permission_mode="bypassPermissions")
    run, captured = _run_sdk_runner(monkeypatch, spec=spec)
    assert "refused" in run.error
    assert captured == {}  # never even reached the SDK


def test_the_hook_denies_under_dry_run(monkeypatch):
    import asyncio

    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), gate=DryRun()), ("Write",))
    _run, captured = _run_sdk_runner(monkeypatch, bridge=bridge)
    hook = captured["options"].hooks["PreToolUse"][0].hooks[0]
    out = asyncio.run(hook({"tool_name": "Write", "tool_input": {"file_path": "a.py"}}, "tu_1", {}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_hook_allows_under_auto_approve(monkeypatch):
    import asyncio

    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), gate=AutoApprove()), ("Write",))
    _run, captured = _run_sdk_runner(monkeypatch, bridge=bridge)
    hook = captured["options"].hooks["PreToolUse"][0].hooks[0]
    out = asyncio.run(hook({"tool_name": "Write", "tool_input": {"file_path": "a.py"}}, "tu_1", {}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_a_tool_outside_the_brief_roster_is_denied():
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent())), ("Read", "Edit"))
    assert bridge.decide("Bash", {"command": "ls"})[0] is False
    assert bridge.decide("Edit", {"file_path": "a.py"})[0] is True


def test_tool_calls_are_counted_structurally_not_by_a_type_field(monkeypatch):
    # The installed content-block dataclasses carry no `type` field, so a
    # `block.type == "tool_use"` read matches nothing and reports zero calls.
    messages = [_FakeAssistant([_FakeToolUse("Write", {"file_path": "a.py"}), _FakeText("done")]),
                _FakeResult()]
    run, _captured = _run_sdk_runner(monkeypatch, messages)
    assert run.tools_used == ["Write"]
    assert run.text == "done"


def test_cost_and_denials_are_read_off_the_result_message(monkeypatch):
    run, _captured = _run_sdk_runner(monkeypatch, [_FakeResult(cost=1.25, denials=3)])
    assert run.cost_usd == 1.25
    assert run.sdk_denials == 3


def test_a_missing_sdk_is_a_loss_not_an_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    from tactics.playbooks.agent_sdk import _sdk_runner

    ws = AgentWorkspace("/repo", check=["true"], shell=lambda cmd: (0, ""))
    run = _sdk_runner("x", BriefSpec(), None, ws)
    assert run.error and run.cost_usd == 0.0


# --- baseline, not absolute dirtiness ----------------------------------------


def test_a_tree_that_was_already_dirty_is_not_counted_as_the_agents_work():
    # Found live: a pre-existing modified file made a run that wrote nothing
    # score 1.0. Absolute dirtiness is not evidence of work.
    agent = ScriptedAgent(tool_calls=[("Read", {"file_path": "a.py"})])
    agent.edited = True  # the tree is dirty before the agent ever runs
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.success is False and out.reward == 0.0
    assert "untouched" in out.notes


def test_roster_denials_are_journalled_like_every_other_refusal():
    journal = Journal()
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), journal=journal), ("Read",))
    bridge.decide("Bash", {"command": "ls"})
    assert "brief.deny" in {e.kind for e in journal.events}


# --- subagent rosters --------------------------------------------------------
#
# All three of these come from the first live ReviewedSwarm run.


def test_delegation_is_gated_but_not_treated_as_a_write():
    # Delegation itself changes nothing; the subagent's own calls fire the same
    # hook individually (verified live: 6 subagent calls, all gated).
    from tactics.playbooks.agent_sdk import DELEGATION_TOOLS

    for tool in DELEGATION_TOOLS:
        assert classify_tool_call(tool, {}) == (True, "low")
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent())), tuple(DELEGATION_TOOLS))
    assert bridge.decide("Agent", {})[0] is True


def test_the_swarm_roster_carries_the_delegation_tool_under_both_names():
    # Found live: the roster listed "Task" while the installed CLI names the tool
    # "Agent", so the gate denied the one call the brief existed to make — and it
    # still scored 1.0, as a solo run wearing a swarm's name.
    from tactics.playbooks.agent_sdk import DELEGATION_TOOLS

    assert DELEGATION_TOOLS <= set(ReviewedSwarm().spec.allowed_tools)


def test_a_brief_that_cannot_reach_its_subagents_fails_before_spending():
    from tactics.playbooks.agent_sdk import _roster_error

    broken = BriefSpec(allowed_tools=("Read", "Edit"), agents={"reviewer": {"description": "d", "prompt": "p"}})
    assert _roster_error(broken)

    class Broken(BriefTactic):
        spec = broken

    agent = ScriptedAgent()
    out = Broken().execute(_ctx(_workspace(agent)))
    assert out.success is False and out.cost == 0.0
    assert not agent.briefs  # refused before the SDK was ever called
    assert "misconfigured brief" in out.notes


def test_subagent_calls_are_attributed_in_the_journal():
    journal = Journal()
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), journal=journal))
    bridge.decide("Write", {"file_path": "a.py"}, agent="code-reviewer")
    event = next(e for e in journal.events if e.kind == "gate.commit")
    assert "code-reviewer subagent" in event.data["action"]


def test_the_hook_passes_subagent_attribution_through(monkeypatch):
    import asyncio

    journal = Journal()
    bridge = GateBridge(_ctx(_workspace(ScriptedAgent()), journal=journal), ("Write",))
    _run, captured = _run_sdk_runner(monkeypatch, bridge=bridge)
    hook = captured["options"].hooks["PreToolUse"][0].hooks[0]
    asyncio.run(hook(
        {"tool_name": "Write", "tool_input": {"file_path": "a.py"},
         "agent_id": "ag_1", "agent_type": "code-reviewer"}, "tu_1", {}))
    assert "code-reviewer subagent" in journal.events[-1].data["action"]


# --- cost accounting with subagents ------------------------------------------


class _FakeResultWithModels:
    def __init__(self, cost, usage, model_usage):
        self.total_cost_usd = cost
        self.usage = usage
        self.model_usage = model_usage
        self.permission_denials = []


def test_tokens_come_from_model_usage_because_usage_omits_subagents(monkeypatch):
    # Live numbers from a ReviewedSwarm run: `usage` reported 512 output tokens
    # for a call that actually produced 5388.
    msg = _FakeResultWithModels(
        cost=0.2178,
        usage={"input_tokens": 2, "output_tokens": 512},
        model_usage={
            "claude-sonnet-5": {"inputTokens": 34, "outputTokens": 5388},
            "claude-haiku-4-5": {"inputTokens": 950, "outputTokens": 13},
        },
    )
    run, _captured = _run_sdk_runner(monkeypatch, [msg])
    assert run.output_tokens == 5401  # not 512
    assert run.input_tokens == 984
    assert run.models == ["claude-haiku-4-5", "claude-sonnet-5"]


def test_the_last_result_wins_because_it_carries_the_call_total(monkeypatch):
    # A call can emit more than one result message; each later one carries the
    # running total for the whole call, so summing would double-count.
    run, _captured = _run_sdk_runner(monkeypatch, [_FakeResult(cost=0.1478), _FakeResult(cost=0.2178)])
    assert run.cost_usd == 0.2178


# --- worktree fan-out --------------------------------------------------------
#
# These use real git repositories in temp dirs. Mocking git would only prove the
# mock agrees with itself; the whole claim here is that two agents editing at the
# same time cannot see each other, and only real worktrees can show that.

import pathlib
import subprocess


def _git_repo(tmp_path) -> str:
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, check=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n")
    run("add", "-A")
    run("commit", "-qm", "seed")
    return str(repo)


def _writing_runner(filename="ant.txt"):
    """A runner that writes into whichever workspace it is handed, if allowed."""

    def runner(brief, spec, bridge, ws):
        run = AgentRun(cost_usd=0.01)
        allowed, _ = bridge.decide("Write", {"file_path": filename})
        run.tools_used.append("Write")
        if allowed:
            # Content names the tree, so two ants' patches are distinguishable.
            pathlib.Path(ws.path, filename).write_text(f"written in {pathlib.Path(ws.path).name}\n")
        run.denied = list(bridge.denied)
        return run

    return runner


def test_without_isolation_a_session_is_just_the_workspace(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path))
    assert ws.session(None) is ws
    ws.release(ws)  # a no-op, and must never delete the real tree
    assert pathlib.Path(ws.path, "seed.txt").exists()


def test_two_sessions_cannot_see_each_others_edits(tmp_path):
    # The whole point of the fan-out.
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True)
    try:
        a, b = ws.session(None), ws.session(None)
        assert a.path != b.path != ws.path
        pathlib.Path(a.path, "only_a.txt").write_text("a\n")
        assert pathlib.Path(a.path, "only_a.txt").exists()
        assert not pathlib.Path(b.path, "only_a.txt").exists()
        assert not pathlib.Path(ws.path, "only_a.txt").exists()  # main tree untouched
        assert pathlib.Path(b.path, "seed.txt").exists()  # but each is a real checkout
    finally:
        ws.cleanup()


def test_release_lifts_the_work_out_as_a_patch_then_removes_the_worktree(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True)
    try:
        from tactics.colony.blackboard import Task

        session = ws.session(Task(id="t7", description="d"))
        pathlib.Path(session.path, "new.txt").write_text("hello\n")
        ws.release(session)

        assert not pathlib.Path(session.path).exists()  # worktree gone
        assert len(ws.patches) == 1
        patch = ws.patches[0]
        assert patch.task == "t7"
        assert "new.txt" in patch.text and "hello" in patch.text  # untracked files included
    finally:
        ws.cleanup()


# --- the work outlives the run ------------------------------------------------


def test_a_patch_is_written_to_disk_before_its_worktree_is_destroyed(tmp_path):
    # Until release runs, the only copy is the worktree; after it, the only copy
    # is a list in memory a killed run never returns. So it must go to disk in
    # between — asserted on the *ordering*, not just the end state.
    archive = tmp_path / "archive"
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True, patch_dir=str(archive))
    saved_when_removed = []
    real_run = ws.run

    def watching(cmd):
        if cmd[:3] == ["git", "worktree", "remove"]:
            saved_when_removed.append(sorted(f.name for f in archive.glob("*.patch")))
        return real_run(cmd)

    ws.run = watching
    try:
        session = ws.session(None)
        session.produced_by = "WriteTestFirst"
        pathlib.Path(session.path, "new.txt").write_text("hello\n")
        ws.release(session)
    finally:
        ws.run = real_run
        ws.cleanup()

    assert saved_when_removed == [["001-WriteTestFirst.patch"]]
    assert "hello" in (archive / "001-WriteTestFirst.patch").read_text()
    assert ws.patches[0].saved_to == str(archive / "001-WriteTestFirst.patch")


def test_the_saved_file_is_a_patch_you_can_actually_apply(tmp_path):
    # An archive you cannot apply is a log, not a recovery.
    archive = tmp_path / "archive"
    repo = _git_repo(tmp_path)
    ws = AgentWorkspace(repo, isolate=True, patch_dir=str(archive))
    try:
        session = ws.session(None)
        pathlib.Path(session.path, "recovered.txt").write_text("from the ant\n")
        ws.release(session)
        saved = ws.patches[0].saved_to
    finally:
        ws.cleanup()

    # Nothing of the run survives but the file. Land it with plain git.
    done = subprocess.run(["git", "-C", repo, "apply", "--3way", saved],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert pathlib.Path(repo, "recovered.txt").read_text() == "from the ant\n"


def test_a_second_run_does_not_overwrite_the_first_ones_answers(tmp_path):
    archive = tmp_path / "archive"
    repo = _git_repo(tmp_path)
    for body in ("first\n", "second\n"):
        ws = AgentWorkspace(repo, isolate=True, patch_dir=str(archive))
        try:
            session = ws.session(None)
            pathlib.Path(session.path, "answer.txt").write_text(body)
            ws.release(session)
        finally:
            ws.cleanup()

    bodies = sorted(f.read_text() for f in archive.glob("*.patch"))
    assert len(bodies) == 2
    assert any("first" in b for b in bodies) and any("second" in b for b in bodies)


def test_an_unwritable_archive_costs_the_copy_not_the_patch(tmp_path):
    # Fail-soft: raising here would lose the very thing being protected.
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("a file, so makedirs cannot use it as a directory\n")
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True, patch_dir=str(blocked))
    try:
        session = ws.session(None)
        pathlib.Path(session.path, "still_here.txt").write_text("work\n")
        ws.release(session)
        assert len(ws.patches) == 1
        assert "still_here.txt" in ws.patches[0].text
        assert ws.patches[0].saved_to == ""   # and it does not claim otherwise
    finally:
        ws.cleanup()


def test_no_patch_dir_means_nothing_is_written(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True)
    try:
        session = ws.session(None)
        pathlib.Path(session.path, "new.txt").write_text("x\n")
        ws.release(session)
        assert ws.patches[0].saved_to == ""
    finally:
        ws.cleanup()


def test_a_session_that_changed_nothing_produces_no_patch(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True)
    try:
        ws.release(ws.session(None))
        assert ws.patches == []
    finally:
        ws.cleanup()


def test_a_patch_can_be_landed_on_the_main_repository(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True)
    try:
        session = ws.session(None)
        pathlib.Path(session.path, "landed.txt").write_text("from the ant\n")
        ws.release(session)

        assert not pathlib.Path(ws.path, "landed.txt").exists()
        ok, out = ws.apply_patch(ws.patches[0])
        assert ok, out
        assert pathlib.Path(ws.path, "landed.txt").read_text() == "from the ant\n"
    finally:
        ws.cleanup()


def test_a_workspace_that_is_not_a_git_repo_fails_loudly(tmp_path):
    # Silently sharing the main tree is the corruption isolation exists to stop,
    # so a broken worktree must raise rather than fall back.
    plain = tmp_path / "plain"
    plain.mkdir()
    ws = AgentWorkspace(str(plain), isolate=True)
    try:
        with __import__("pytest").raises(RuntimeError, match="worktree"):
            ws.session(None)
    finally:
        ws.cleanup()


def test_parallel_workers_are_refused_without_isolation(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path))
    try:
        build_delivery_colony(ws, max_workers=3)
        raise AssertionError("should have refused")
    except ValueError as exc:
        assert "isolate=True" in str(exc)


def test_a_parallel_colony_gives_each_ant_its_own_tree(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), check=["test", "-f", "ant.txt"],
                        runner=_writing_runner(), isolate=True)
    try:
        colony = build_delivery_colony(ws, gate=AutoApprove(), max_workers=3, max_rounds=1)
        result = colony.run(delivery_goal("make the change"))

        assert result.history[0].dispatched == 3           # three ants ran at once
        assert len(ws.patches) == 3                        # each produced its own work
        bodies = {p.text for p in ws.patches}
        assert len(bodies) == 3                            # in three distinct trees
        assert not pathlib.Path(ws.path, "ant.txt").exists()  # main repo never written
        assert ws.changed_files() == []
    finally:
        ws.cleanup()


def test_a_parallel_dry_run_writes_nothing_anywhere(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), check=["test", "-f", "ant.txt"],
                        runner=_writing_runner(), isolate=True)
    try:
        colony = build_delivery_colony(ws, gate=DryRun(), max_workers=2, max_rounds=1)
        colony.run(delivery_goal("make the change"))
        assert ws.patches == []
        assert ws.changed_files() == []
    finally:
        ws.cleanup()


def test_cleanup_removes_every_worktree_it_created(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True)
    sessions = [ws.session(None) for _ in range(3)]
    root = pathlib.Path(sessions[0].path).parent
    ws.cleanup()
    assert not root.exists()
    code, out = ws.run(["git", "worktree", "list"])
    assert code == 0 and out.strip().count("\n") == 0  # only the main tree remains


def test_delivery_goal_is_for_repair_and_is_satisfied_by_a_green_check():
    agent = ScriptedAgent()
    ctx = _ctx(_workspace(agent))
    assert ctx.goal.satisfied_by(ctx) is False
    agent.edited = True
    assert ctx.goal.satisfied_by(ctx) is True


def test_work_queue_goal_does_not_declare_victory_before_any_ant_runs():
    # Found live: on a repo whose suite already passes, a check-based goal is
    # satisfied at round 0 and the colony stops having done nothing.
    from tactics.playbooks.agent_sdk import work_queue_goal

    agent = ScriptedAgent()
    agent.edited = True  # check is already green
    ctx = _ctx(_workspace(agent))
    ctx.goal = work_queue_goal("add a new method")
    assert ctx.goal.satisfied_by(ctx) is False


def test_the_journal_entry_says_what_the_run_actually_changed():
    # Found live: the entry was written before the measurement, so every run
    # reported files=None — an audit trail of intent, not of effect.
    journal = Journal()
    SingleAgentNarrow().execute(_ctx(_workspace(ScriptedAgent()), journal=journal))
    entry = next(e for e in journal.events if e.kind == "agent.run")
    assert entry.data["changed_files"] == 1


def test_a_gate_held_run_teaches_the_policy_nothing_about_the_brief():
    # Found live: a parallel DryRun reported "2 tasks done" and learned a 0 for
    # the brief — but the brief was never allowed to try.
    from tactics.core.outcome import Outcome

    ctx = _ctx(_workspace(ScriptedAgent()))
    held = Outcome(success=False, reward=0.0, metrics={"denied": 4, "changed_files": 0})
    verdict = verification_critic().verify(held, ctx)
    assert verdict.accepted is False
    assert "held by the gate" in verdict.reason


def test_a_real_failure_is_still_learned_from():
    from tactics.core.outcome import Outcome

    agent = ScriptedAgent(check_after_edit=False)
    ctx = _ctx(_workspace(agent))
    tried = Outcome(success=False, reward=0.0, metrics={"denied": 0, "changed_files": 2})
    assert verification_critic().verify(tried, ctx).accepted is True


# --- compounding: JsonStore + Scribe -----------------------------------------
#
# The claim this section has to earn: a second run starts better off than the
# first, from disk, with no help from a live process.

import json as _json

from tactics import InMemoryLessons, InMemoryStore, JsonStore, Lesson
from tactics.llm import ScriptedClient
from tactics.playbooks.agent_sdk import (
    BriefScribe,
    brief_lessons,
    brief_memory,
    run_and_learn,
)


def test_a_brief_carries_no_lessons_when_no_store_is_wired():
    agent = ScriptedAgent()
    SingleAgentNarrow().execute(_ctx(_workspace(agent), description="fix the parser"))
    assert agent.briefs[0] == "fix the parser"  # unchanged


def test_past_lessons_are_prepended_to_the_brief():
    store = InMemoryLessons()
    store.add(Lesson(text="rspec is slow here; scope it to the changed file", playbook="agent_sdk"))
    agent = ScriptedAgent()
    SingleAgentNarrow(lessons=store).execute(_ctx(_workspace(agent), description="fix the parser"))
    assert "rspec is slow here" in agent.briefs[0]
    assert agent.briefs[0].endswith("fix the parser")  # the task still comes last


def test_a_lesson_from_another_playbook_does_not_leak_in():
    store = InMemoryLessons()
    store.add(Lesson(text="never place a market order at the open", playbook="trading"))
    store.add(Lesson(text="this repo's test suite needs PYTHONPATH", playbook="agent_sdk"))
    agent = ScriptedAgent()
    SingleAgentNarrow(lessons=store).execute(_ctx(_workspace(agent)))
    assert "market order" not in agent.briefs[0]
    assert "PYTHONPATH" in agent.briefs[0]


def test_the_system_prompt_is_never_varied_by_lessons():
    # The system prompt IS the brief shape being measured; quietly changing it
    # would make two runs of the same tactic incomparable.
    store = InMemoryLessons()
    store.add(Lesson(text="a lesson", playbook="agent_sdk"))
    agent = ScriptedAgent()
    tactic = SingleAgentNarrow(lessons=store)
    tactic.execute(_ctx(_workspace(agent)))
    assert agent.specs[0].system_prompt == SingleAgentNarrow.spec.system_prompt


def test_persist_puts_both_halves_of_memory_in_the_repo(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), runner=_writing_runner())
    colony = build_delivery_colony(ws, persist=True, gate=AutoApprove(), max_rounds=1)
    assert isinstance(colony.memory, JsonStore)
    assert colony.memory.path.endswith(".tactics/agent_sdk_memory.json")
    # and every brief was handed the lesson store
    assert all(t.lessons is not None for t in colony.tactics)


def test_what_a_brief_earns_survives_the_process(tmp_path):
    repo = _git_repo(tmp_path)

    first = brief_memory(repo)
    first.record("PlanThenPatch", '{"goal":"delivery"}', reward=1.0, success=True)
    first.record("PlanThenPatch", '{"goal":"delivery"}', reward=1.0, success=True)
    first.record("WriteTestFirst", '{"goal":"delivery"}', reward=0.0, success=False)

    reloaded = brief_memory(repo)  # a fresh process would do exactly this
    assert reloaded.stats("PlanThenPatch", '{"goal":"delivery"}').trials == 2
    assert reloaded.stats("PlanThenPatch", '{"goal":"delivery"}').mean_reward == 1.0
    assert reloaded.stats("WriteTestFirst", '{"goal":"delivery"}').mean_reward == 0.0


def test_the_scribe_sees_which_brief_is_winning():
    memory = InMemoryStore()
    memory.record("PlanThenPatch", "sig", reward=1.0, success=True, features={"dirty": False})
    memory.record("ReviewedSwarm", "sig", reward=0.0, success=False, features={"dirty": False})
    board = BriefScribe(ScriptedClient(["{}"]), InMemoryLessons(), memory=memory).scoreboard()
    assert "PlanThenPatch: 1 run(s), mean reward 1.00" in board
    assert "ReviewedSwarm" in board
    assert board.index("PlanThenPatch") < board.index("ReviewedSwarm")  # winner first


def test_run_and_learn_writes_lessons_a_later_run_reads_back(tmp_path):
    """The whole point, end to end and offline: run 1 teaches run 2."""
    repo = _git_repo(tmp_path)
    lesson_text = "the check must set PYTHONPATH or it measures the wrong tree"
    client = ScriptedClient([_json.dumps({"lessons": [{"text": lesson_text, "evidence": "round 1"}]})])

    ws = AgentWorkspace(repo, check=["test", "-f", "ant.txt"], runner=_writing_runner())
    colony = build_delivery_colony(ws, persist=True, gate=AutoApprove(), max_rounds=1)
    _result, written = run_and_learn(colony, delivery_goal("do the thing"), client=client)
    assert [lesson.text for lesson in written] == [lesson_text]

    # A second process: nothing in memory, everything from disk.
    agent = ScriptedAgent()
    ws2 = AgentWorkspace(repo, check=["true"], runner=agent.runner, shell=agent.shell)
    colony2 = build_delivery_colony(ws2, persist=repo, gate=AutoApprove(), max_rounds=1)
    tactic = colony2.tactics[0]
    tactic.execute(_ctx(ws2, description="do the next thing"))
    assert lesson_text in agent.briefs[0]


def test_run_and_learn_without_a_client_still_runs_and_still_persists(tmp_path):
    repo = _git_repo(tmp_path)
    ws = AgentWorkspace(repo, check=["test", "-f", "ant.txt"], runner=_writing_runner())
    colony = build_delivery_colony(ws, persist=True, gate=AutoApprove(), max_rounds=1)
    result, written = run_and_learn(colony, delivery_goal("do the thing"))
    assert written == []
    assert result.history[0].accepted == 1  # an ant really ran and was learned from

    on_disk = _json.loads(pathlib.Path(repo, ".tactics", "agent_sdk_memory.json").read_text())
    assert [row["tactic"] for row in on_disk if row["trials"]]  # a brief's record, persisted


def test_a_failed_distillation_writes_nothing(tmp_path):
    # One fabricated lesson pollutes every future brief that recalls it.
    repo = _git_repo(tmp_path)
    ws = AgentWorkspace(repo, check=["true"], runner=_writing_runner())
    colony = build_delivery_colony(ws, persist=True, gate=AutoApprove(), max_rounds=1)
    _result, written = run_and_learn(colony, delivery_goal("x"), client=ScriptedClient(["not json"]))
    assert written == []
    assert list(brief_lessons(repo).entries()) == []


def test_a_fresh_process_exploits_what_earlier_runs_proved(tmp_path):
    """The load-bearing claim: yesterday's results change today's choice.

    Note what this does *not* assert on a partly-explored store — UCB tries an
    untried brief before exploiting a proven one, which is correct and is why
    "it picked the winner" is only meaningful once every brief has a record.
    """
    repo = _git_repo(tmp_path)
    ws = AgentWorkspace(repo, check=["true"], shell=lambda cmd: (0, ""))
    ctx = _ctx(ws)
    ctx.features = ws.features({})
    signature = ctx.signature()

    seed = brief_memory(repo)
    scores = {"PlanThenPatch": 1.0, "SingleAgentNarrow": 0.0, "WriteTestFirst": 0.2,
              "ReviewedSwarm": 0.1}
    for _ in range(8):
        for name, reward in scores.items():
            seed.record(name, signature, reward=reward, success=reward > 0.5,
                        features=ctx.features, goal="delivery")

    # A new colony that has only what is on disk — no shared object, no warm cache.
    colony = build_delivery_colony(ws, persist=repo, gate=AutoApprove())
    picks = {colony.policy.choose(colony.tactics, ctx, colony.memory).name for _ in range(8)}
    assert picks == {"PlanThenPatch"}


def test_an_untried_brief_is_explored_before_a_proven_one_is_exploited(tmp_path):
    # Guards the reading of the test above: with a brief still unmeasured, the
    # policy should try it rather than settle early on a small sample.
    repo = _git_repo(tmp_path)
    ws = AgentWorkspace(repo, check=["true"], shell=lambda cmd: (0, ""))
    ctx = _ctx(ws)
    ctx.features = ws.features({})

    seed = brief_memory(repo)
    for _ in range(5):
        seed.record("PlanThenPatch", ctx.signature(), reward=1.0, success=True,
                    features=ctx.features, goal="delivery")

    colony = build_delivery_colony(ws, persist=repo, gate=AutoApprove())
    assert colony.policy.choose(colony.tactics, ctx, colony.memory).name != "PlanThenPatch"


# --- per-round brief exploration ---------------------------------------------


def test_a_parallel_colony_spreads_its_ants_across_briefs(tmp_path):
    # Found live: three parallel ants all picked WriteTestFirst, so a fan-out
    # round bought three samples of one brief instead of one each of three.
    ws = AgentWorkspace(_git_repo(tmp_path), check=["test", "-f", "ant.txt"],
                        runner=_writing_runner(), isolate=True)
    try:
        colony = build_delivery_colony(ws, gate=AutoApprove(), max_workers=3, max_rounds=1)
        result = colony.run(delivery_goal("do the work"))
        briefs = [e.data["tactic"] for e in result.journal.events if e.kind == "agent.run"]
        assert len(briefs) == 3
        assert len(set(briefs)) == 3  # three different briefs, one round
    finally:
        ws.cleanup()


def test_spreading_can_be_turned_off_to_resample_the_best_brief(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), check=["test", "-f", "ant.txt"],
                        runner=_writing_runner(), isolate=True)
    try:
        colony = build_delivery_colony(ws, gate=AutoApprove(), max_workers=3,
                                       max_rounds=1, spread=False)
        result = colony.run(delivery_goal("do the work"))
        briefs = [e.data["tactic"] for e in result.journal.events if e.kind == "agent.run"]
        assert len(set(briefs)) == 1
    finally:
        ws.cleanup()


def test_a_serial_colony_is_left_alone():
    # Nothing to spread across when one ant runs at a time.
    from tactics import UCBPolicy

    agent = ScriptedAgent()
    colony = build_delivery_colony(_workspace(agent), gate=AutoApprove(), max_workers=1)
    assert isinstance(colony.policy, UCBPolicy)


def test_the_spread_wrapper_is_wired_in_for_parallel_rounds(tmp_path):
    from tactics import WithoutReplacement

    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True)
    colony = build_delivery_colony(ws, max_workers=2)
    assert isinstance(colony.policy, WithoutReplacement)


# --- choosing among the patches a fan-out produced ---------------------------
#
# Real git repos again: the whole claim is that a candidate is re-measured
# against the *current* HEAD, which a fake can only assert about itself.

from tactics.playbooks.agent_sdk import ApplyBestPatch, Patch, land_best_patch


def _patch_from(ws, writes: dict, *, task="t1", tactic="Brief"):
    """Produce a real Patch by making `writes` in a throwaway worktree."""
    session = ws.session(None)
    session.task_id, session.produced_by = task, tactic
    for name, body in writes.items():
        pathlib.Path(session.path, name).write_text(body)
    ws.release(session)
    return ws.patches[-1]


def _repo_with_check(tmp_path, check):
    repo = _git_repo(tmp_path)
    return AgentWorkspace(repo, check=check, isolate=True)


def test_a_patch_is_re_verified_against_current_head(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "good.txt"])
    try:
        good = _patch_from(ws, {"good.txt": "ok\n"}, tactic="Good")
        assert ws.trial(good).ok
    finally:
        ws.cleanup()


def test_a_patch_that_fails_the_check_is_rejected(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "good.txt"])
    try:
        bad = _patch_from(ws, {"other.txt": "nope\n"}, tactic="Bad")
        trial = ws.trial(bad)
        assert trial.applies and not trial.passes and not trial.ok
    finally:
        ws.cleanup()


def test_a_patch_that_no_longer_applies_is_rejected(tmp_path):
    ws = _repo_with_check(tmp_path, ["true"])
    try:
        stale = _patch_from(ws, {"seed.txt": "rewritten by the ant\n"})
        # HEAD moves underneath it, exactly as it would if another patch landed.
        pathlib.Path(ws.path, "seed.txt").write_text("someone else got here first\n")
        ws.run(["git", "commit", "-qam", "moved on"])
        assert not ws.trial(stale).applies
    finally:
        ws.cleanup()


def test_the_best_candidate_is_landed_and_the_rest_retired(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "feature.txt"])
    try:
        _patch_from(ws, {"feature.txt": "a\n", "extra.txt": "noise\n"}, task="t1", tactic="Verbose")
        _patch_from(ws, {"feature.txt": "b\n"}, task="t2", tactic="Tight")
        assert len(ws.patches) == 2

        outcome = land_best_patch(ws)
        assert outcome.success and outcome.reward == 1.0
        assert outcome.metrics["candidates"] == 2 and outcome.metrics["verified"] == 2
        assert outcome.metrics["chose"] == "Tight"          # smallest verified diff
        assert pathlib.Path(ws.path, "feature.txt").exists()
        assert not pathlib.Path(ws.path, "extra.txt").exists()
        assert [p.tactic for p in ws.landed] == ["Tight"]
        assert [p.tactic for p in ws.discarded] == ["Verbose"]
        assert ws.patches == []                              # nothing left to re-land
    finally:
        ws.cleanup()


def test_a_failing_candidate_never_wins_however_small(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "feature.txt"])
    try:
        _patch_from(ws, {"tiny.txt": "x\n"}, task="t1", tactic="TinyButWrong")
        _patch_from(ws, {"feature.txt": "a\n", "more.txt": "b\n"}, task="t2", tactic="BigButRight")
        outcome = land_best_patch(ws)
        assert outcome.metrics["chose"] == "BigButRight"
        assert outcome.metrics["verified"] == 1
    finally:
        ws.cleanup()


def test_when_nothing_survives_nothing_is_landed(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "never.txt"])
    try:
        _patch_from(ws, {"a.txt": "a\n"}, task="t1", tactic="One")
        _patch_from(ws, {"b.txt": "b\n"}, task="t2", tactic="Two")
        outcome = land_best_patch(ws)
        assert outcome.success is False and outcome.reward == 0.0
        assert "no candidate survived" in outcome.notes
        assert ws.changed_files() == [] and ws.landed == []
    finally:
        ws.cleanup()


def test_the_gate_can_hold_the_landing(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "feature.txt"])
    try:
        _patch_from(ws, {"feature.txt": "a\n"}, tactic="Tight")
        journal = Journal()
        outcome = land_best_patch(ws, gate=DryRun(), journal=journal)
        assert outcome.success is False
        assert "the gate held it" in outcome.notes
        assert ws.changed_files() == [] and ws.patches  # still available to land later
        # The selection still ran, so DryRun leaves a review artifact.
        held = next(e for e in journal.events if e.kind == "gate.hold")
        assert "land patch from Tight" in held.data["action"]
        assert [e for e in journal.events if e.kind == "patch.trial"]
    finally:
        ws.cleanup()


def test_the_judge_breaks_a_tie_between_verified_candidates(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "feature.txt"])
    try:
        _patch_from(ws, {"feature.txt": "a\n", "notes.md": "why\n"}, task="t1", tactic="Documented")
        _patch_from(ws, {"feature.txt": "b\n"}, task="t2", tactic="Bare")
        # Candidates reach the judge in ranked order, so 0 is the deterministic
        # pick ("Bare", the smaller diff) and 1 is the one it must override to.
        judge = ScriptedClient([_json.dumps({"choice": 1, "why": "it explains itself"})])
        outcome = land_best_patch(ws, judge=judge)
        assert outcome.metrics["chose"] == "Documented"
        assert "it explains itself" in outcome.notes
        assert outcome.cost > 0  # judgment is not free
    finally:
        ws.cleanup()


def test_a_judge_that_names_an_unverified_candidate_is_ignored(tmp_path):
    # The judge may re-order proven options. It may never be the proof.
    ws = _repo_with_check(tmp_path, ["test", "-f", "feature.txt"])
    try:
        _patch_from(ws, {"feature.txt": "a\n", "x.txt": "x\n"}, task="t1", tactic="Bigger")
        _patch_from(ws, {"feature.txt": "b\n"}, task="t2", tactic="Smaller")
        judge = ScriptedClient([_json.dumps({"choice": 7, "why": "I like a third one"})])
        outcome = land_best_patch(ws, judge=judge)
        assert outcome.metrics["chose"] == "Smaller"  # the measured order stands
        assert "not on the list" in outcome.notes
    finally:
        ws.cleanup()


def test_an_unusable_judge_falls_back_to_the_measured_order(tmp_path):
    ws = _repo_with_check(tmp_path, ["test", "-f", "feature.txt"])
    try:
        _patch_from(ws, {"feature.txt": "a\n", "x.txt": "x\n"}, task="t1", tactic="Big")
        _patch_from(ws, {"feature.txt": "b\n"}, task="t2", tactic="Small")
        outcome = land_best_patch(ws, judge=ScriptedClient(["not json at all"]))
        assert outcome.success and outcome.metrics["chose"] == "Small"
        assert "judge unusable" in outcome.notes
    finally:
        ws.cleanup()


def test_the_tactic_stands_down_when_there_is_nothing_to_choose(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path))
    ctx = _ctx(ws)
    assert ApplyBestPatch().is_applicable(ctx) is False
    assert land_best_patch(ws).notes == "no candidate patches to choose from"


def test_a_patch_records_which_brief_wrote_it(tmp_path):
    ws = _repo_with_check(tmp_path, ["true"])
    try:
        agent = ScriptedAgent()
        session = ws.session(None)
        session._runner = agent.runner
        SingleAgentNarrow().execute(_ctx(session))
        ws.release(session)
        assert ws.patches == [] or ws.patches[-1].tactic == "SingleAgentNarrow"
    finally:
        ws.cleanup()


def test_work_the_agent_staged_is_not_silently_dropped(tmp_path):
    # `git add` through Bash is an ordinary thing for an agent to do, and plain
    # `git diff` would show none of it — the patch came back empty and the ant's
    # work vanished without a word.
    ws = AgentWorkspace(_git_repo(tmp_path), check=["true"], isolate=True)
    try:
        session = ws.session(None)
        session.produced_by = "Stager"
        pathlib.Path(session.path, "staged.py").write_text("print('work')\n")
        session.run(["git", "add", "staged.py"])
        ws.release(session)

        assert len(ws.patches) == 1
        assert "staged.py" in ws.patches[0].text
        assert ws.patches[0].files == ["staged.py"]
    finally:
        ws.cleanup()


def test_a_staged_patch_still_lands_and_verifies(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), check=["test", "-f", "staged.py"], isolate=True)
    try:
        session = ws.session(None)
        session.produced_by = "Stager"
        pathlib.Path(session.path, "staged.py").write_text("print('work')\n")
        session.run(["git", "add", "staged.py"])
        ws.release(session)

        outcome = land_best_patch(ws)
        assert outcome.success and outcome.metrics["chose"] == "Stager"
        assert pathlib.Path(ws.path, "staged.py").exists()
    finally:
        ws.cleanup()


def test_a_failed_delivery_is_retried_rather_than_declared_done():
    # The live symptom: a colony on a genuinely broken repo ran exactly one round
    # and stopped with "no open work", having recorded a 0. Failures could never
    # repeat, so nothing could ever be learned from them repeating.
    from tactics.core.outcome import Outcome

    agent = ScriptedAgent(check_after_edit=False)
    verdict = verification_critic().verify(
        Outcome(success=False, reward=0.0, metrics={"denied": 0, "changed_files": 2}),
        _ctx(_workspace(agent)),
    )
    assert verdict.accepted is True    # trustworthy, and worth learning from
    assert verdict.done is False       # but the work is not finished


def test_a_verified_success_finishes_the_task():
    from tactics.core.outcome import Outcome

    agent = ScriptedAgent()
    agent.edited = True
    verdict = verification_critic().verify(
        Outcome(success=True, reward=1.0, metrics={"changed_files": 2}),
        _ctx(_workspace(agent)),
    )
    assert verdict.accepted is True and verdict.done is True


def test_a_colony_on_a_repo_it_cannot_fix_keeps_trying(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), check=["test", "-f", "impossible.txt"],
                        runner=_writing_runner(), isolate=True)
    try:
        colony = build_delivery_colony(ws, gate=AutoApprove(), max_workers=1, max_rounds=3)
        result = colony.run(delivery_goal("do the impossible"))
        runs = [e for e in result.journal.events if e.kind == "agent.run"]
        assert len(runs) == 3                       # three attempts, not one
        assert result.board.counts()["done"] == 0
    finally:
        ws.cleanup()


def test_the_journal_says_why_a_check_failed_not_just_that_it_did():
    # Six live runs produced no lesson, and this is why: the journal carried a
    # tally, never a cause. A failure that recurs every round has to be legible
    # *as* a recurrence to anything reading the trail afterwards.
    journal = Journal()
    agent = ScriptedAgent(check_after_edit=False)
    SingleAgentNarrow().execute(_ctx(_workspace(agent), journal=journal))
    failure = next(e for e in journal.events if e.kind == "check.failed")
    assert failure.data["tactic"] == "SingleAgentNarrow"
    assert "1 failed" in failure.data["detail"]


def test_a_passing_check_records_no_failure():
    journal = Journal()
    SingleAgentNarrow().execute(_ctx(_workspace(ScriptedAgent()), journal=journal))
    assert not [e for e in journal.events if e.kind == "check.failed"]


def test_persist_wires_lessons_into_caller_supplied_tactics(tmp_path):
    # Otherwise `persist=True` silently means "numbers only" as soon as you pass
    # your own roster — which is exactly how three live runs distilled nothing
    # while looking like the scribe had declined.
    ws = AgentWorkspace(_git_repo(tmp_path), runner=_writing_runner())
    colony = build_delivery_colony(ws, tactics=[SingleAgentNarrow()], persist=True)
    assert colony.tactics[0].lessons is not None


def test_an_explicit_lesson_store_on_a_tactic_is_left_alone(tmp_path):
    mine = InMemoryLessons()
    ws = AgentWorkspace(_git_repo(tmp_path), runner=_writing_runner())
    colony = build_delivery_colony(ws, tactics=[SingleAgentNarrow(lessons=mine)], persist=True)
    assert colony.tactics[0].lessons is mine


def test_a_skipped_distillation_says_so_in_the_journal(tmp_path):
    ws = AgentWorkspace(_git_repo(tmp_path), check=["true"], runner=_writing_runner())
    colony = build_delivery_colony(ws, gate=AutoApprove(), max_rounds=1)
    result, written = run_and_learn(colony, delivery_goal("x"))  # no client
    assert written == []
    skipped = next(e for e in result.journal.events if e.kind == "scribe.skipped")
    assert skipped.data["reason"] == "no client"


def test_the_verdict_reason_distinguishes_a_fix_from_a_reproduced_failure():
    # The scribe read "check re-run agrees" on three failing rounds and had to
    # work out that agreement meant the failure reproduced, not that it passed.
    from tactics.core.outcome import Outcome

    agent = ScriptedAgent(check_after_edit=False)
    ctx = _ctx(_workspace(agent))
    failing = verification_critic().verify(
        Outcome(success=False, reward=0.0, metrics={"changed_files": 2}), ctx)
    assert "still failing" in failing.reason

    agent.edited = True
    agent.check_after_edit = True
    passing = verification_critic().verify(
        Outcome(success=True, reward=1.0, metrics={"changed_files": 2}), ctx)
    assert "confirms the fix" in passing.reason


def test_a_run_that_errored_is_still_measured_not_assumed_failed():
    # A turn limit or a dropped connection does not undo the work already done.
    # Scoring it 0 without looking is the same self-report this playbook refuses
    # to trust, just inverted.
    class ErrorsAfterWorking(ScriptedAgent):
        def runner(self, brief, spec, bridge, ws):
            run = super().runner(brief, spec, bridge, ws)
            run.error = "ResultError('Reached maximum number of turns')"
            return run

    agent = ErrorsAfterWorking()
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.success and out.reward == 1.0     # the check is what decides
    assert "also errored" in out.notes           # and the error is still on the record


def test_an_errored_run_that_did_nothing_is_still_a_loss():
    agent = ScriptedAgent(error="RuntimeError('boom')")
    out = SingleAgentNarrow().execute(_ctx(_workspace(agent)))
    assert out.success is False
    assert "no work done" in out.notes


def test_the_journal_says_how_many_lessons_reached_the_brief():
    # A store that is wired but empty looks exactly like a store with nothing
    # relevant to say. Three separate misreadings came from not being able to
    # tell those apart — including a 30-trial experiment that measured nothing.
    store = InMemoryLessons()
    store.add(Lesson(text="a thing we learned", playbook="agent_sdk"))
    journal = Journal()
    SingleAgentNarrow(lessons=store).execute(_ctx(_workspace(ScriptedAgent()), journal=journal))
    assert next(e for e in journal.events if e.kind == "lessons.recalled").data["count"] == 1


def test_an_empty_store_is_visibly_empty_rather_than_silent():
    journal = Journal()
    SingleAgentNarrow(lessons=InMemoryLessons()).execute(
        _ctx(_workspace(ScriptedAgent()), journal=journal))
    assert next(e for e in journal.events if e.kind == "lessons.recalled").data["count"] == 0


def test_no_store_records_nothing_at_all():
    journal = Journal()
    SingleAgentNarrow().execute(_ctx(_workspace(ScriptedAgent()), journal=journal))
    assert not [e for e in journal.events if e.kind == "lessons.recalled"]


# --- bounding what a growing store puts in front of the task -----------------


def test_a_growing_store_cannot_swamp_the_brief():
    # JsonlLessons is append-only: without a ceiling, a year of lessons ends up
    # in front of every task.
    store = InMemoryLessons()
    for i in range(6):
        store.add(Lesson(text=f"lesson {i} " + "x" * 500, playbook="agent_sdk"))
    agent = ScriptedAgent()
    journal = Journal()
    SingleAgentNarrow(lessons=store, lesson_budget=1200).execute(
        _ctx(_workspace(agent), journal=journal))
    recalled = next(e for e in journal.events if e.kind == "lessons.recalled")
    assert recalled.data["count"] < 6 and recalled.data["dropped"] > 0
    assert len(agent.briefs[0]) < 2200          # bounded, not unbounded


def test_the_most_relevant_lesson_always_survives_the_budget():
    # Truncation drops the tail, so a budget smaller than one lesson still keeps
    # the best one rather than silently recalling nothing.
    store = InMemoryLessons()
    store.add(Lesson(text="the important one " + "y" * 900, playbook="agent_sdk", goal="delivery"))
    agent = ScriptedAgent()
    SingleAgentNarrow(lessons=store, lesson_budget=100).execute(_ctx(_workspace(agent)))
    assert "the important one" in agent.briefs[0]


def test_a_store_within_budget_is_untouched():
    store = InMemoryLessons()
    store.add(Lesson(text="short and useful", playbook="agent_sdk"))
    journal = Journal()
    SingleAgentNarrow(lessons=store).execute(_ctx(_workspace(ScriptedAgent()), journal=journal))
    recalled = next(e for e in journal.events if e.kind == "lessons.recalled")
    assert recalled.data == {"tactic": "SingleAgentNarrow", "count": 1, "dropped": 0}


def test_the_scribe_is_told_to_diagnose_rather_than_prescribe():
    # The 450-trial result, wired back into the thing that writes the lessons.
    from tactics.llm import ScriptedClient
    from tactics.playbooks.agent_sdk import BriefScribe

    scribe = BriefScribe(ScriptedClient(["{}"]), InMemoryLessons(), memory=InMemoryStore())

    class _Result:
        goal, journal, findings = Goal(name="delivery"), Journal(), []
        def summary(self): return "s"

    prompt = scribe.build_prompt(_Result(), "agent_sdk")
    assert "Name the cause; do not prescribe a procedure" in prompt


def test_what_the_check_creates_is_not_captured_as_the_agents_work(tmp_path):
    # Found live: the check ran pytest, pytest wrote __pycache__, and the .pyc
    # files ended up in the patch — which then would not apply anywhere else.
    # The agent must do the writing, or the no-op guard short-circuits execute
    # and the check never runs — which is how the first version of this test
    # passed against the very bug it was meant to catch.
    def writing_runner(brief, spec, bridge, ws_):
        run = AgentRun(cost_usd=0.01)
        if bridge.decide("Write", {"file_path": "real_work.py"})[0]:
            pathlib.Path(ws_.path, "real_work.py").write_text("print('the agent did this')\n")
        return run

    ws = AgentWorkspace(_git_repo(tmp_path), isolate=True, runner=writing_runner,
                        check=["sh", "-c", "mkdir -p __pycache__ && echo x > __pycache__/a.pyc"])
    try:
        session = ws.session(None)
        SingleAgentNarrow().execute(_ctx(session))
        ws.release(session)

        patch = ws.patches[0]
        assert "real_work.py" in patch.text          # the agent's change is there
        assert ".pyc" not in patch.text              # the check's litter is not
        assert not any(".pyc" in f for f in patch.files)
    finally:
        ws.cleanup()


def test_a_session_with_no_pending_capture_still_captures_at_release(tmp_path):
    # release() is the fallback path when a tactic never ran (a bare session).
    ws = AgentWorkspace(_git_repo(tmp_path), check=["true"], isolate=True)
    try:
        session = ws.session(None)
        pathlib.Path(session.path, "by_hand.txt").write_text("hi\n")
        ws.release(session)
        assert "by_hand.txt" in ws.patches[0].text
    finally:
        ws.cleanup()
