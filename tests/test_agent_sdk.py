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
