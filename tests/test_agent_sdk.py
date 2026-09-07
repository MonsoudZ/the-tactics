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
    assert "no changes" in out.notes


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
