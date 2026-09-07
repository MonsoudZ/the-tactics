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
