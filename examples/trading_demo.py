"""Trading demo — the safety layer doing its job on a market. Offline.

Prices are a fixed synthetic series and the broker is simulated, so nothing here
touches money. That is the point: you can watch the gate, the risk limits and
the benchmark-relative reward work before anything real is at stake.

Run it:  python3 examples/trading_demo.py
"""

from __future__ import annotations

import math

from tactics import AutoApprove, Budget, Context, DryRun, Journal
from tactics.playbooks.trading import (
    Market,
    PaperBroker,
    RideMomentum,
    RiskLimits,
    StandAside,
    scoreboard,
    survive_goal,
    walk_forward,
)


def wave(n=400, period=20, amp=8.0, base=100.0, drift=0.0):
    return [round(base + amp * math.sin(2 * math.pi * i / period) + drift * i, 2)
            for i in range(n)]


def falling(n=12):
    return [round(120 - 2 * i, 2) for i in range(n)]


def one_decision(label, gate):
    """One tactic, one bar, one gate — so the posture is visible on its own."""
    market = Market(PaperBroker({"AAA": [100] * 3 + [110] * 9}, cash=10_000.0))
    journal = Journal()
    data = market.observe()
    ctx = Context(target=market, goal=survive_goal(), data=data,
                  features=market.features(data), gate=gate, journal=journal)
    RideMomentum(threshold=-1.0).execute(ctx)
    print(f"  {label:<34} fills={len(market.broker.fills)}  cash={market.broker.cash():,.0f}")
    for e in journal.events:
        if e.kind in ("gate.commit", "gate.hold", "risk.refused"):
            print(f"      {e.kind:<12} {e.data}")


if __name__ == "__main__":
    print("=== One decision, three postures ===")
    one_decision("AutoApprove (a backtest)", AutoApprove())
    one_decision("DryRun (a real venue)", DryRun())
    one_decision("RiskLimits: order cap $250", RiskLimits(AutoApprove(), max_order_value=250.0))

    print("\n=== Standing aside is a real move ===")
    market = Market(PaperBroker({"AAA": falling()}, cash=10_000.0))
    data = market.observe()
    ctx = Context(target=market, goal=survive_goal(), data=data,
                  features=market.features(data), gate=AutoApprove(), journal=Journal())
    out = StandAside().execute(ctx)
    print(f"  market fell; holding cash scored {out.reward:+.3%} against the benchmark")
    print("  (raw P&L would have called this a zero — the decision would be invisible)")

    print("\n=== Walking forward: episodes are the unit of learning ===")
    prices = {"AAA": wave(), "BBB": wave(period=13, amp=5.0, base=50.0, drift=0.02)}
    results, memory, markets = walk_forward(
        prices, bars=30, episodes=10,
        limits=RiskLimits(max_order_value=4_000.0, max_drawdown=0.25),
        budget=Budget(max_cost=50.0),
    )
    from collections import Counter
    print(f"  {len(results)} episodes, "
          f"choices: {dict(Counter(s.tactic for r in results for s in r.steps))}")
    print(f"  final equity per episode: {[round(m.equity()) for m in markets]}")
    print("\n  what it learned, per regime:")
    print(scoreboard(memory, limit=8))

    print("\n  Every order was a proposal the gate decided on; reward came from the")
    print("  account, net of buy-and-hold. No broker was contacted at any point.")
