"""Tests for delayed credit assignment."""

from __future__ import annotations

from tactics import DiscountedReturn, ImmediateCredit
from tactics.core.credit import TrajectoryStep


def _step(tactic: str, reward: float) -> TrajectoryStep:
    return TrajectoryStep(
        signature="sig", features={}, goal="g", tactic=tactic, reward=reward, success=reward > 0
    )


def test_immediate_credit_writes_each_step_online():
    c = ImmediateCredit()
    recs = c.on_step(_step("a", 1.0))
    assert len(recs) == 1
    assert recs[0].value == 1.0
    assert c.on_finish([_step("a", 1.0)]) == []


def test_discounted_return_waits_then_spreads_reward_backward():
    c = DiscountedReturn(gamma=0.5)
    # The win only lands on the last step; earlier setup moves should still get credit.
    traj = [_step("setup", 0.0), _step("build", 0.0), _step("payoff", 1.0)]
    assert c.on_step(traj[0]) == []  # nothing online

    recs = c.on_finish(traj)
    by_tactic = {r.tactic: r.value for r in recs}
    # G for payoff = 1.0; build = 0 + 0.5*1 = 0.5; setup = 0 + 0.5*0.5 = 0.25
    assert by_tactic["payoff"] == 1.0
    assert by_tactic["build"] == 0.5
    assert by_tactic["setup"] == 0.25


def test_discounted_return_records_preserve_situation():
    c = DiscountedReturn(gamma=0.9)
    traj = [
        TrajectoryStep("sigA", {"x": 1}, "g", "t1", 0.0, False),
        TrajectoryStep("sigB", {"x": 2}, "g", "t2", 2.0, True),
    ]
    recs = {r.tactic: r for r in c.on_finish(traj)}
    assert recs["t1"].signature == "sigA"
    assert recs["t1"].features == {"x": 1}
    assert recs["t2"].value == 2.0


def test_gamma_zero_is_myopic():
    c = DiscountedReturn(gamma=0.0)
    traj = [_step("a", 0.0), _step("b", 5.0)]
    by_tactic = {r.tactic: r.value for r in c.on_finish(traj)}
    assert by_tactic["a"] == 0.0  # no credit for the future
    assert by_tactic["b"] == 5.0
