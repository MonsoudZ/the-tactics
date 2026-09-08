"""Tests for the trading playbook. Deterministic, offline, no broker anywhere.

The properties that matter here are not "does it make money" — a fixed price
series can be made to say anything. They are: an order can only be created by
the gate, reward is measured from the account rather than reported by the
tactic, and the limits refuse what they say they refuse.
"""

from __future__ import annotations

import pytest

from tactics import AutoApprove, Context, DryRun, Journal, RecencyStore
from tactics.playbooks.trading import (
    BUY,
    SELL,
    CutLoss,
    FadeExtreme,
    Market,
    Order,
    PaperBroker,
    Position,
    RideMomentum,
    RiskLimits,
    StandAside,
    TakeProfit,
    build_trader,
    growth_goal,
    scoreboard,
    survive_goal,
    walk_forward,
)

RISING = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122]
FALLING = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98]


def _market(prices=None, *, cash=10_000.0, fee_bps=0.0, lookback=5) -> Market:
    return Market(PaperBroker(prices or {"AAA": list(RISING)}, cash=cash, fee_bps=fee_bps),
                  lookback=lookback)


def _ctx(market, *, gate=None, journal=None, goal=None):
    data = market.observe()
    return Context(target=market, goal=goal or survive_goal(), data=data,
                   features=market.features(data),
                   gate=gate if gate is not None else AutoApprove(),
                   journal=journal or Journal())


# --- the broker refuses what a real one would ---------------------------------


def test_there_is_no_overdraft():
    broker = PaperBroker({"AAA": list(RISING)}, cash=150.0)
    assert broker.submit(Order("AAA", BUY, 5)) is None          # 500 > 150
    assert broker.cash() == 150.0


def test_there_is_no_shorting():
    broker = PaperBroker({"AAA": list(RISING)}, cash=1000.0)
    assert broker.submit(Order("AAA", SELL, 1)) is None
    broker.submit(Order("AAA", BUY, 2))
    assert broker.submit(Order("AAA", SELL, 3)) is None
    assert broker.submit(Order("AAA", SELL, 2)) is not None


def test_fees_come_out_of_cash():
    broker = PaperBroker({"AAA": list(RISING)}, cash=1000.0, fee_bps=100.0)  # 1%
    fill = broker.submit(Order("AAA", BUY, 5))
    assert fill.fee == pytest.approx(5.0)                        # 1% of 500
    assert broker.cash() == pytest.approx(1000 - 500 - 5)


def test_cost_basis_averages_across_buys():
    broker = PaperBroker({"AAA": [100, 200]}, cash=10_000.0)
    broker.submit(Order("AAA", BUY, 1))
    broker.advance()
    broker.submit(Order("AAA", BUY, 1))
    assert broker.positions()["AAA"].cost_basis == pytest.approx(150.0)


def test_the_series_runs_out():
    broker = PaperBroker({"AAA": [1, 2]}, cash=10.0)
    assert broker.advance() is True
    assert broker.advance() is False


def test_price_series_must_line_up():
    with pytest.raises(ValueError, match="same length"):
        PaperBroker({"AAA": [1, 2, 3], "BBB": [1, 2]})


# --- reward is measured, and measured against the market ----------------------


def test_equity_is_cash_plus_holdings():
    market = _market(cash=1000.0)
    assert market.equity() == 1000.0
    market.place(Order("AAA", BUY, 5))          # 5 @ 100
    assert market.equity() == pytest.approx(1000.0)   # value unchanged by the swap


def test_standing_aside_in_a_falling_market_earns_a_positive_reward():
    # The point of a benchmark-relative reward: holding cash through a decline is
    # a good decision, and raw P&L would score it zero.
    market = _market({"AAA": list(FALLING)})
    outcome = StandAside().execute(_ctx(market))
    assert outcome.reward > 0 and outcome.success


def test_riding_a_rising_market_beats_the_benchmark_by_nothing_much():
    # Fully invested in the only symbol *is* the benchmark, so excess ~ 0 minus
    # fees. A framework that scored this highly would be measuring the weather.
    market = _market({"AAA": list(RISING)}, fee_bps=0.0)
    market.place(Order("AAA", BUY, 100))        # 100 @ 100 = all the cash
    assert market.settle() == pytest.approx(0.0, abs=1e-9)


def test_a_tactic_cannot_report_its_own_reward():
    # The reward is whatever the account did over the bar, full stop.
    market = _market({"AAA": list(FALLING)})
    outcome = RideMomentum().execute(_ctx(market))
    assert outcome.reward == pytest.approx(outcome.metrics["excess_return"], abs=1e-6)
    assert outcome.metrics["equity"] == pytest.approx(market.equity(), abs=0.01)


# --- nothing fills except through the gate ------------------------------------


def test_a_held_order_never_reaches_the_broker():
    market = _market({"AAA": list(RISING)}, cash=1000.0)
    outcome = RideMomentum(threshold=-1.0).execute(_ctx(market, gate=DryRun()))
    assert market.broker.fills == []            # the money never moved
    assert market.broker.cash() == 1000.0
    assert outcome.metrics["proposed"] is True and outcome.metrics["traded"] is False
    assert "not filled" in outcome.notes


def test_an_approved_order_does_reach_the_broker():
    market = _market({"AAA": list(RISING)}, cash=1000.0)
    outcome = RideMomentum(threshold=-1.0).execute(_ctx(market, gate=AutoApprove()))
    assert market.broker.fills and outcome.metrics["traded"] is True


def test_every_order_is_proposed_as_irreversible_and_high_risk():
    seen = []

    class Watch(DryRun):
        def decide(self, proposal, ctx):
            seen.append(proposal)
            return False

    market = _market({"AAA": list(RISING)}, cash=1000.0)
    RideMomentum(threshold=-1.0).execute(_ctx(market, gate=Watch()))
    assert seen and all(not p.reversible and p.risk == "high" for p in seen)


def test_a_bar_still_settles_when_the_order_is_held():
    # Otherwise a DryRun run would freeze time and learn nothing about the market.
    market = _market({"AAA": list(FALLING)}, cash=1000.0)
    before = market.broker.quote("AAA")
    StandAside().execute(_ctx(market, gate=DryRun()))
    assert market.broker.quote("AAA") != before


# --- risk limits sit in front of the posture ----------------------------------


def test_an_oversized_order_is_refused_even_under_auto_approve():
    market = _market({"AAA": list(RISING)}, cash=10_000.0)
    journal = Journal()
    limits = RiskLimits(AutoApprove(), max_order_value=100.0)
    outcome = RideMomentum(threshold=-1.0).execute(_ctx(market, gate=limits, journal=journal))
    assert market.broker.fills == []
    assert "risk limit" in outcome.notes
    assert journal.of_kind("risk.refused")


def test_a_position_cap_stops_the_position_growing():
    market = _market({"AAA": list(RISING)}, cash=10_000.0)
    limits = RiskLimits(AutoApprove(), max_position_value=250.0)
    RideMomentum(threshold=-1.0, fraction=1.0).execute(_ctx(market, gate=limits))
    assert market.broker.fills == []


def test_trading_stops_once_the_drawdown_limit_is_breached():
    limits = RiskLimits(AutoApprove(), max_drawdown=0.10)
    assert limits.breach({"equity": 10_000.0}) == ""      # sets the high-water mark
    assert limits.breach({"equity": 9_500.0}) == ""       # -5%, still inside
    reason = limits.breach({"equity": 8_500.0})           # -15%
    assert "drawdown" in reason and "do not trade back" in reason


def test_within_limits_the_inner_posture_still_decides():
    market = _market({"AAA": list(RISING)}, cash=10_000.0)
    limits = RiskLimits(DryRun(), max_order_value=1e9)
    RideMomentum(threshold=-1.0).execute(_ctx(market, gate=limits))
    assert market.broker.fills == []          # generous limits, but DryRun still holds


# --- the wiring is the safety posture -----------------------------------------


def test_a_simulated_broker_auto_approves():
    assert isinstance(build_trader(_market()).gate, AutoApprove)


def test_anything_that_is_not_a_paper_broker_defaults_to_dry_run():
    # The default has to be safe for the case the author did not think about.
    class SomeVenue:
        def quote(self, s): return 1.0
        def cash(self): return 0.0
        def positions(self): return {}
        def submit(self, order): return None
        def advance(self): return False
        def symbols(self): return ["AAA"]

    assert isinstance(build_trader(Market(SomeVenue())).gate, DryRun)


def test_memory_is_recency_weighted_by_default():
    assert isinstance(build_trader(_market()).memory, RecencyStore)


def test_persisting_keeps_the_decay(tmp_path):
    from tactics import JsonRecencyStore

    agent = build_trader(_market(), persist=str(tmp_path / "m.json"), decay=0.8)
    assert isinstance(agent.memory, JsonRecencyStore) and agent.memory.decay == 0.8


def test_credit_is_delayed_because_a_bar_answers_the_previous_decision():
    from tactics.core.credit import DiscountedReturn

    assert isinstance(build_trader(_market()).credit, DiscountedReturn)


# --- episodes are the unit of learning ----------------------------------------


def _wave(n=400, period=20, amp=8.0, base=100.0):
    import math
    return [round(base + amp * math.sin(2 * math.pi * i / period), 2) for i in range(n)]


def test_one_long_run_cannot_learn_within_itself():
    # Not a defect but a consequence: DiscountedReturn writes nothing until the
    # episode ends, so a single long pursue consults an empty memory every step
    # and takes the first tactic every time. This pins the reason walk_forward
    # exists — it was found by watching one tactic get chosen 290 times.
    market = Market(PaperBroker({"AAA": _wave()}, cash=10_000.0))
    agent = build_trader(market, max_steps=60)
    result = agent.pursue(survive_goal())
    assert len({s.tactic for s in result.steps}) == 1
    assert list(agent.memory.entries()) != []      # it all lands at the end


def test_walking_forward_lets_every_tactic_compete():
    prices = {"AAA": _wave(), "BBB": _wave(period=13, amp=5.0, base=50.0)}
    results, memory, markets = walk_forward(prices, bars=30, episodes=8)
    assert len(results) == 8 and len(markets) == 8
    chosen = {s.tactic for r in results for s in r.steps}
    assert len(chosen) >= 4                        # a real competition, not a default
    assert scoreboard(memory) != "(nothing learned yet)"


def test_episodes_share_one_memory():
    prices = {"AAA": _wave()}
    _results, memory, _m = walk_forward(prices, bars=30, episodes=5)
    trials = sum(e.stats.trials for e in memory.entries())
    assert trials > 5           # more than one episode's worth accumulated


def test_each_episode_starts_from_a_clean_account():
    prices = {"AAA": _wave()}
    _r, _mem, markets = walk_forward(prices, bars=30, episodes=3, cash=5_000.0)
    assert all(m.broker.symbols() == ["AAA"] for m in markets)
    assert len({id(m.broker) for m in markets}) == 3


def test_it_refuses_to_walk_further_than_the_data():
    with pytest.raises(ValueError, match="need"):
        walk_forward({"AAA": _wave(n=50)}, bars=30, episodes=10)


# --- goals --------------------------------------------------------------------


def test_the_growth_goal_reads_the_account():
    market = _market(cash=1000.0)
    ctx = _ctx(market, goal=growth_goal(1500.0))
    assert ctx.goal.satisfied_by(ctx) is False
    assert _ctx(market, goal=growth_goal(500.0)).goal.satisfied_by(ctx) is True


def test_the_open_ended_goal_never_declares_victory():
    market = _market()
    assert survive_goal().satisfied_by(_ctx(market)) is False


def test_tactics_stand_down_once_the_series_is_exhausted():
    market = Market(PaperBroker({"AAA": [100, 101]}, cash=1000.0))
    market.settle()
    market.settle()             # runs off the end
    assert market.exhausted
    assert StandAside().is_applicable(_ctx(market)) is False


# --- each tactic's actual decision --------------------------------------------


def _warm(market, bars):
    """Indicators need history; at bar 0 momentum and z-score are both zero."""
    for _ in range(bars):
        market.settle()
    return market


def test_momentum_buys_the_strongest_symbol_and_only_past_its_threshold():
    market = _warm(_market({"WEAK": [100, 100, 100, 100, 100, 100],
                            "STRONG": [100, 104, 108, 112, 116, 120]}), 5)
    assert RideMomentum(threshold=0.05).decide(_ctx(market)).symbol == "STRONG"
    assert RideMomentum(threshold=0.90).decide(_ctx(market)) is None   # nothing is that strong


def test_fade_buys_what_is_furthest_below_its_own_mean():
    market = _warm(_market({"CALM": [100, 100, 100, 100, 100, 100],
                            "SLUMPED": [120, 118, 112, 104, 96, 88]}), 5)
    order = FadeExtreme(threshold=-1.0).decide(_ctx(market))
    assert order is not None and order.symbol == "SLUMPED" and order.side == BUY
    assert FadeExtreme(threshold=-5.0).decide(_ctx(market)) is None    # not extreme enough


def test_take_profit_sells_a_winner_and_leaves_a_loser_alone():
    market = _market({"AAA": [100, 100, 100, 130]}, cash=10_000.0)
    market.place(Order("AAA", BUY, 10))            # basis 100
    assert TakeProfit(gain=0.05).decide(_ctx(market)) is None   # still at 100
    market.settle(); market.settle(); market.settle()           # price runs to 130
    order = TakeProfit(gain=0.05).decide(_ctx(market))
    assert order is not None and order.side == SELL and order.qty == 10


def test_cut_loss_sells_a_loser_and_leaves_a_winner_alone():
    market = _market({"AAA": [100, 100, 100, 80]}, cash=10_000.0)
    market.place(Order("AAA", BUY, 10))
    assert CutLoss(loss=0.05).decide(_ctx(market)) is None
    market.settle(); market.settle(); market.settle()           # price falls to 80
    order = CutLoss(loss=0.05).decide(_ctx(market))
    assert order is not None and order.side == SELL and order.qty == 10


def test_neither_exit_fires_with_nothing_held():
    market = _market({"AAA": [100, 130]})
    assert TakeProfit().decide(_ctx(market)) is None
    assert CutLoss().decide(_ctx(market)) is None


def test_a_position_reports_its_unrealised_pnl():
    assert Position("AAA", qty=10, cost_basis=100.0).unrealized(112.0) == pytest.approx(120.0)
    assert Position("AAA").unrealized(112.0) == 0.0


def test_a_tactic_that_cannot_afford_anything_stands_aside():
    market = _market({"AAA": [100, 104, 108, 112, 116, 120]}, cash=1.0)
    assert RideMomentum(threshold=0.01).decide(_ctx(market)) is None


def test_a_backtest_decays_by_observation_and_a_live_account_by_the_clock():
    # Replaying five years of bars in three seconds should fade nothing, so the
    # default is observation-based. A live account wants the opposite.
    from tactics import RecencyStore, TimeDecayStore

    assert isinstance(build_trader(_market()).memory, RecencyStore)
    timed = build_trader(_market(), half_life=3600.0).memory
    assert isinstance(timed, TimeDecayStore) and timed.half_life == 3600.0


def test_a_clock_decayed_account_persists_its_timestamps(tmp_path):
    from tactics import JsonTimeDecayStore

    agent = build_trader(_market(), persist=str(tmp_path / "m.json"), half_life=900.0)
    assert isinstance(agent.memory, JsonTimeDecayStore)
    assert agent.memory.half_life == 900.0
