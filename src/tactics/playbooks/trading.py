"""Trading playbook — a market as a Target the framework governs.

The domain that makes every safety rule in this framework load-bearing at once.
A bad tactic in the repo playbook wastes a few dollars of tokens and a git
revert; a bad tactic here spends money that does not come back. So:

  * **every order goes through the gate** as an irreversible, high-risk
    :class:`~tactics.core.approval.Proposal` — a tactic proposes, it never fills;
  * the default posture is chosen by *what the broker is*: simulated fills are
    auto-approved, anything else defaults to :class:`DryRun`;
  * :class:`RiskLimits` sits in front of whatever posture you pick, so an order
    that breaches position, order or drawdown limits is refused before the
    question of approval is even asked;
  * reward is measured from the account, and measured **against a benchmark**.

**Why benchmark-relative reward.** The obvious signal — change in equity — is
mostly the market, not the decision. In a rising market every tactic looks
brilliant, including "buy something at random", and the policy would learn
nothing except that it is nice to trade in a bull market. So a step's reward is
the account's return *minus* an equal-weight buy-and-hold of the same symbols
over the same bar. Standing aside in a falling market earns a positive reward,
which is exactly right and is the sort of thing raw P&L cannot express.

**Why the single Agent loop and not the Colony.** The repo playbook fans out by
giving each ant its own git worktree. There is no worktree for a brokerage
account: two tactics trading one account interleave into a position neither of
them chose. Run this with :class:`~tactics.core.engine.Agent`, one decision at a
time. ``build_trader`` does that for you.

**Why delayed credit.** A decision at bar *t* is answered by bar *t+1*, so each
tactic settles the bar it acted on before reporting — and across a run, use
:class:`~tactics.core.credit.DiscountedReturn` so a payoff several bars later is
credited back to the decisions that set it up.

**Why a persistent recency store.** Markets are non-stationary *and* long
running. ``build_trader(persist=...)`` uses :class:`JsonRecencyStore` so what
worked last quarter fades instead of anchoring the policy, without losing
everything when the process restarts.

    from tactics.playbooks.trading import Market, PaperBroker, build_trader, growth_goal
    market = Market(PaperBroker({"ACME": prices}, cash=10_000))
    agent = build_trader(market)
    result = agent.pursue(growth_goal(11_000), max_steps=200)

Nothing here talks to a real broker. The :class:`Broker` protocol is the seam
where one would go, and no live broker has ever been run against this code.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..core.approval import ApprovalGate, AutoApprove, DryRun, GateResult, Proposal
from ..core.credit import DiscountedReturn
from ..core.engine import Agent
from ..core.estimator import SimilarityEstimator
from ..core.goal import Goal
from ..core.memory import JsonRecencyStore, MemoryStore, RecencyStore
from ..core.outcome import Outcome
from ..core.policy import Policy, UCBPolicy
from ..core.tactic import Tactic
from ..core.target import Target

BUY, SELL = "buy", "sell"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str          # BUY or SELL
    qty: float

    def __str__(self) -> str:
        return f"{self.side.upper()} {self.qty:g} {self.symbol}"


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    cost_basis: float = 0.0   # average price paid for the open quantity

    def unrealized(self, price: float) -> float:
        return (price - self.cost_basis) * self.qty


class Broker(Protocol):
    """The seam a real brokerage would plug into. Nothing here is implemented
    against a live venue, and the framework never assumes one."""

    def quote(self, symbol: str) -> float: ...
    def cash(self) -> float: ...
    def positions(self) -> dict[str, Position]: ...
    def submit(self, order: Order) -> Fill | None: ...
    def advance(self) -> bool: ...
    def symbols(self) -> list[str]: ...


class PaperBroker:
    """Deterministic simulated fills against a fixed price series.

    Fills at the current bar's price plus a fee, refuses to spend cash it does
    not have and to sell stock it does not hold. It is a test instrument, not a
    market simulator: no partial fills, no queue position, no gaps, and no
    slippage beyond the fee you configure. A strategy that only ever looks good
    here has been measured against a world that is kinder than the real one.
    """

    def __init__(
        self,
        prices: dict[str, list[float]],
        *,
        cash: float = 10_000.0,
        fee_bps: float = 5.0,
    ) -> None:
        if not prices or not all(prices.values()):
            raise ValueError("PaperBroker needs at least one non-empty price series")
        lengths = {len(v) for v in prices.values()}
        if len(lengths) != 1:
            raise ValueError(f"price series must be the same length, got {lengths}")
        self._prices = {k: list(v) for k, v in prices.items()}
        self._i = 0
        self._cash = float(cash)
        self._fee = fee_bps / 10_000.0
        self._positions: dict[str, Position] = {}
        self.fills: list[Fill] = []

    # --- Broker ---------------------------------------------------------------

    def symbols(self) -> list[str]:
        return sorted(self._prices)

    def quote(self, symbol: str) -> float:
        return self._prices[symbol][self._i]

    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return {s: Position(p.symbol, p.qty, p.cost_basis) for s, p in self._positions.items()}

    def advance(self) -> bool:
        """Move to the next bar. False when the series is exhausted."""
        if self._i + 1 >= len(next(iter(self._prices.values()))):
            return False
        self._i += 1
        return True

    def submit(self, order: Order) -> Fill | None:
        if order.qty <= 0 or order.symbol not in self._prices:
            return None
        price = self.quote(order.symbol)
        pos = self._positions.setdefault(order.symbol, Position(order.symbol))
        if order.side == BUY:
            cost = order.qty * price
            fee = cost * self._fee
            if cost + fee > self._cash:
                return None                       # no leverage, no overdraft
            self._cash -= cost + fee
            total = pos.cost_basis * pos.qty + cost
            pos.qty += order.qty
            pos.cost_basis = total / pos.qty
        elif order.side == SELL:
            if order.qty > pos.qty:
                return None                       # no shorting
            proceeds = order.qty * price
            fee = proceeds * self._fee
            self._cash += proceeds - fee
            pos.qty -= order.qty
            if pos.qty == 0:
                pos.cost_basis = 0.0
        else:
            return None
        fill = Fill(order.symbol, order.side, order.qty, price, fee)
        self.fills.append(fill)
        return fill

    # --- history, for indicators ---------------------------------------------

    def history(self, symbol: str, bars: int) -> list[float]:
        start = max(0, self._i - bars + 1)
        return self._prices[symbol][start : self._i + 1]


class Market(Target):
    """An account and the prices it trades against.

    ``observe`` is total and cheap; every domain method a tactic needs hangs off
    here. ``settle`` is the one that matters: it advances a bar and reports the
    account's return *net of* an equal-weight buy-and-hold of the same symbols,
    which is the reward signal.
    """

    name = "market"

    def __init__(self, broker: Broker, *, lookback: int = 10) -> None:
        self.broker = broker
        self.lookback = lookback
        self._benchmark_basis = {s: broker.quote(s) for s in broker.symbols()}
        self._last_equity = self.equity()
        self._last_benchmark = self.benchmark()
        self.exhausted = False

    # --- state ----------------------------------------------------------------

    def equity(self) -> float:
        cash = self.broker.cash()
        held = sum(p.qty * self.broker.quote(s) for s, p in self.broker.positions().items())
        return cash + held

    def benchmark(self) -> float:
        """An equal-weight buy-and-hold of the same symbols, indexed to 1.0."""
        legs = [self.broker.quote(s) / b for s, b in self._benchmark_basis.items() if b]
        return sum(legs) / len(legs) if legs else 1.0

    def observe(self) -> dict[str, Any]:
        prices = {s: self.broker.quote(s) for s in self.broker.symbols()}
        positions = self.broker.positions()
        return {
            "cash": self.broker.cash(),
            "equity": self.equity(),
            "prices": prices,
            "held": {s: p.qty for s, p in positions.items() if p.qty},
            "exhausted": self.exhausted,
        }

    def features(self, data: dict[str, Any]) -> dict[str, Any]:
        """Bucket learning by *regime*. A tactic that wins in a calm uptrend and
        loses in a choppy one has two records here, not one average that hides
        both — which is the whole reason a market needs situation features."""
        moves = [self.momentum(s, self.lookback) for s in self.broker.symbols()]
        drift = sum(moves) / len(moves) if moves else 0.0
        vols = [self.volatility(s, self.lookback) for s in self.broker.symbols()]
        vol = sum(vols) / len(vols) if vols else 0.0
        return {
            "trend": "up" if drift > 0.01 else "down" if drift < -0.01 else "flat",
            "vol": "high" if vol > 0.02 else "low",
        }

    # --- indicators -----------------------------------------------------------

    def price(self, symbol: str) -> float:
        return self.broker.quote(symbol)

    def history(self, symbol: str, bars: int | None = None) -> list[float]:
        return self.broker.history(symbol, bars or self.lookback)

    def momentum(self, symbol: str, bars: int | None = None) -> float:
        h = self.history(symbol, bars)
        return (h[-1] / h[0] - 1.0) if len(h) > 1 and h[0] else 0.0

    def volatility(self, symbol: str, bars: int | None = None) -> float:
        h = self.history(symbol, bars)
        rets = [h[i] / h[i - 1] - 1.0 for i in range(1, len(h)) if h[i - 1]]
        return statistics.pstdev(rets) if len(rets) > 1 else 0.0

    def zscore(self, symbol: str, bars: int | None = None) -> float:
        h = self.history(symbol, bars)
        if len(h) < 3:
            return 0.0
        spread = statistics.pstdev(h)
        return (h[-1] - statistics.fmean(h)) / spread if spread else 0.0

    def held(self, symbol: str) -> float:
        return self.broker.positions().get(symbol, Position(symbol)).qty

    def unrealized(self, symbol: str) -> float:
        pos = self.broker.positions().get(symbol)
        return pos.unrealized(self.price(symbol)) if pos else 0.0

    # --- acting and measuring -------------------------------------------------

    def place(self, order: Order) -> Fill | None:
        """Fill an order. Tactics must not call this directly — propose through
        ``ctx.gate`` so the decision is gated and journalled."""
        return self.broker.submit(order)

    def settle(self) -> float:
        """Advance one bar and report the account's excess return over the bar.

        This is the reward. It is the account's return minus the benchmark's, so
        it measures the *decision* rather than the weather.
        """
        before_equity, before_benchmark = self.equity(), self.benchmark()
        if not self.broker.advance():
            self.exhausted = True
            return 0.0
        mine = (self.equity() / before_equity - 1.0) if before_equity else 0.0
        theirs = (self.benchmark() / before_benchmark - 1.0) if before_benchmark else 0.0
        self._last_equity, self._last_benchmark = self.equity(), self.benchmark()
        return mine - theirs


class RiskLimits(ApprovalGate):
    """Hard limits in front of whatever approval posture you chose.

    Two separate questions, deliberately not merged: *is this order within the
    limits we set* and *is it approved*. This answers the first, and defers the
    second to ``inner``. So a limit breach is refused even under
    :class:`AutoApprove`, and a within-limits order still has to face the gate
    that would have judged it anyway.

    ``max_drawdown`` is measured against the high-water mark of equity, so a
    strategy that has already lost more than you were prepared to lose stops
    trading rather than trying to win it back.
    """

    def __init__(
        self,
        inner: ApprovalGate | None = None,
        *,
        max_order_value: float | None = None,
        max_position_value: float | None = None,
        max_drawdown: float | None = None,
    ) -> None:
        self.inner = inner or DryRun()
        self.max_order_value = max_order_value
        self.max_position_value = max_position_value
        self.max_drawdown = max_drawdown
        self._high_water: float | None = None

    def breach(self, detail: dict[str, Any]) -> str:
        """Why this order is refused, or "" if it is within limits."""
        equity = detail.get("equity")
        if equity is not None:
            self._high_water = equity if self._high_water is None else max(self._high_water, equity)
            if self.max_drawdown is not None and self._high_water:
                drawdown = 1.0 - equity / self._high_water
                if drawdown > self.max_drawdown:
                    return (f"drawdown {drawdown:.1%} exceeds the {self.max_drawdown:.1%} "
                            "limit — stop trading, do not trade back")
        value = detail.get("order_value")
        if self.max_order_value is not None and value is not None and value > self.max_order_value:
            return f"order value {value:,.2f} exceeds the {self.max_order_value:,.2f} limit"
        after = detail.get("position_value_after")
        if self.max_position_value is not None and after is not None and after > self.max_position_value:
            return (f"position would reach {after:,.2f}, over the "
                    f"{self.max_position_value:,.2f} limit")
        return ""

    def decide(self, proposal: Proposal, ctx) -> bool:  # noqa: ANN001
        if self.breach(proposal.detail):
            return False
        return self.inner.decide(proposal, ctx)

    def submit(self, proposal: Proposal, ctx=None) -> GateResult:  # noqa: ANN001
        reason = self.breach(proposal.detail)
        if reason:
            journal = getattr(ctx, "journal", None)
            if journal is not None:
                journal.record("risk.refused", action=proposal.action, reason=reason)
            return GateResult(approved=False, committed=False, reason=f"risk limit: {reason}")
        return self.inner.submit(proposal, ctx)


class TradingTactic(Tactic):
    """Base for anything that trades. Subclasses implement :meth:`decide` only.

    The template is the safety property: a subclass returns an *intention*, and
    this class is the only thing that turns one into an order — through the
    gate, never around it. It then settles the bar and reports what the account
    actually did, so a tactic cannot report its own reward.
    """

    def decide(self, ctx: Any) -> Order | None:
        """Return the order this tactic wants, or None to stand aside."""
        raise NotImplementedError

    def is_applicable(self, ctx: Any) -> bool:
        return not ctx.data.get("exhausted", False)

    def execute(self, ctx: Any) -> Outcome:
        market: Market = ctx.target
        order = self.decide(ctx)
        fill, refused = None, ""
        if order is not None:
            price = market.price(order.symbol)
            held_after = market.held(order.symbol) + (order.qty if order.side == BUY else -order.qty)
            result = ctx.gate.submit(
                Proposal(
                    action=f"{order} @ ~{price:,.2f}",
                    commit=lambda: market.place(order),
                    reversible=False,   # a fill cannot be taken back, only offset
                    risk="high",        # it is money
                    detail={
                        "symbol": order.symbol, "side": order.side, "qty": order.qty,
                        "price": price, "order_value": order.qty * price,
                        "position_value_after": max(0.0, held_after) * price,
                        "equity": market.equity(),
                    },
                ),
                ctx,
            )
            fill = result.result if result.committed else None
            refused = "" if result.committed else result.reason

        excess = market.settle()          # the bar answers the decision
        metrics = {
            "excess_return": round(excess, 6),
            "equity": round(market.equity(), 2),
            "traded": fill is not None,
            "proposed": order is not None,
        }
        note = f"{order}" if order is not None else "stood aside"
        if refused:
            note = f"{note} — not filled: {refused}"
        return Outcome(
            success=excess > 0,
            reward=excess,
            cost=fill.fee if fill else 0.0,   # fees are what trading consumes
            metrics=metrics,
            notes=note,
        )


def _affordable_qty(market: Market, symbol: str, fraction: float) -> float:
    price = market.price(symbol)
    if price <= 0:
        return 0.0
    budget = market.broker.cash() * fraction
    return float(int(budget / price))


class RideMomentum(TradingTactic):
    """Buy what has been going up. Wins in trends, bleeds in chop."""

    def __init__(self, *, threshold: float = 0.02, fraction: float = 0.25, name=None) -> None:
        super().__init__(name=name)
        self.threshold, self.fraction = threshold, fraction

    def decide(self, ctx: Any) -> Order | None:
        market: Market = ctx.target
        ranked = sorted(market.broker.symbols(), key=market.momentum, reverse=True)
        best = ranked[0] if ranked else None
        if best is None or market.momentum(best) < self.threshold:
            return None
        qty = _affordable_qty(market, best, self.fraction)
        return Order(best, BUY, qty) if qty else None


class FadeExtreme(TradingTactic):
    """Buy what has fallen far from its own mean. The opposite bet to momentum,
    and it is here so the two can be measured against each other rather than
    argued about."""

    def __init__(self, *, threshold: float = -1.5, fraction: float = 0.25, name=None) -> None:
        super().__init__(name=name)
        self.threshold, self.fraction = threshold, fraction

    def decide(self, ctx: Any) -> Order | None:
        market: Market = ctx.target
        ranked = sorted(market.broker.symbols(), key=market.zscore)
        worst = ranked[0] if ranked else None
        if worst is None or market.zscore(worst) > self.threshold:
            return None
        qty = _affordable_qty(market, worst, self.fraction)
        return Order(worst, BUY, qty) if qty else None


class TakeProfit(TradingTactic):
    """Sell a position that is up more than ``gain``. Realising a gain is a
    decision, and it competes with holding on like any other."""

    def __init__(self, *, gain: float = 0.05, name=None) -> None:
        super().__init__(name=name)
        self.gain = gain

    def decide(self, ctx: Any) -> Order | None:
        market: Market = ctx.target
        for symbol, pos in market.broker.positions().items():
            if pos.qty and pos.cost_basis and market.price(symbol) / pos.cost_basis - 1 >= self.gain:
                return Order(symbol, SELL, pos.qty)
        return None


class CutLoss(TradingTactic):
    """Sell a position that is down more than ``loss``. The one tactic whose job
    is to be wrong cheaply."""

    def __init__(self, *, loss: float = 0.05, name=None) -> None:
        super().__init__(name=name)
        self.loss = loss

    def decide(self, ctx: Any) -> Order | None:
        market: Market = ctx.target
        for symbol, pos in market.broker.positions().items():
            if pos.qty and pos.cost_basis and 1 - market.price(symbol) / pos.cost_basis >= self.loss:
                return Order(symbol, SELL, pos.qty)
        return None


class StandAside(TradingTactic):
    """Do nothing, on purpose.

    It is a real competitor, not a null option: under a benchmark-relative
    reward, holding cash through a decline earns a positive score. A framework
    that cannot express "the best available move is not to trade" will always
    find a reason to trade.
    """

    def decide(self, ctx: Any) -> Order | None:
        return None


def growth_goal(target_equity: float, name: str = "grow") -> Goal:
    """Satisfied when the account reaches ``target_equity``. An honest finish
    line: it reads the account, not the tactics' opinions of themselves."""
    return Goal(
        name=name,
        description=f"grow the account to {target_equity:,.2f}",
        is_satisfied=lambda ctx: ctx.target.equity() >= target_equity,
    )


def survive_goal(name: str = "trade") -> Goal:
    """No finish line — trade until the bars, the budget or the step cap run out.

    The honest shape for an open-ended pursuit. Pair it with a ``Budget`` and
    ``RiskLimits``; a goal that never completes is only safe because something
    else is counting.
    """
    return Goal(name=name, description="trade the series without a fixed target")


def default_tactics() -> list[Tactic]:
    return [RideMomentum(), FadeExtreme(), TakeProfit(), CutLoss(), StandAside()]


def build_trader(
    market: Market,
    *,
    tactics: list[Tactic] | None = None,
    gate: ApprovalGate | None = None,
    limits: RiskLimits | None = None,
    policy: Policy | None = None,
    memory: MemoryStore | None = None,
    persist: str | None = None,
    decay: float = 0.95,
    gamma: float = 0.9,
    budget: Any = None,
    max_steps: int = 200,
) -> Agent:
    """Wire the loop the way this domain needs it.

    Three defaults here are decisions, not conveniences:

    * **The Agent, not the Colony.** One account cannot be forked, so decisions
      are serial. Parallel ants would interleave into a position none of them
      chose.
    * **The posture follows the broker.** Simulated fills auto-approve, because
      a backtest that needs a human to click through every order is not a
      backtest. Anything else — a real venue — defaults to :class:`DryRun`, and
      you have to say otherwise on purpose.
    * **Recency-weighted memory.** A market is non-stationary; a plain mean over
      all history anchors the policy to a regime that has ended. Pass ``persist``
      to keep that memory across restarts without losing the decay.
    * **A similarity estimator, not exact matching.** Regime features change
      almost every bar, so under exact matching nearly every step is a situation
      never seen before, UCB explores rather than compares, and the first tactic
      in the list gets picked forever. Measured: with `ExactEstimator` a 300-bar
      run chose one tactic on every step. Borrowing from *similar* regimes is
      what lets the comparison actually happen.

    ``gamma`` feeds :class:`DiscountedReturn`, because a decision is usually
    answered several bars after it is made.
    """
    is_paper = isinstance(market.broker, PaperBroker)
    posture = gate or (AutoApprove() if is_paper else DryRun())
    if limits is not None:
        limits.inner = posture
        posture = limits

    if memory is None:
        memory = (JsonRecencyStore(persist, decay=decay) if persist
                  else RecencyStore(decay=decay))

    return Agent(
        market,
        tactics or default_tactics(),
        policy=policy or UCBPolicy(estimator=SimilarityEstimator()),
        memory=memory,
        credit=DiscountedReturn(gamma=gamma),
        gate=posture,
        budget=budget,
        max_steps=max_steps,
    )


def scoreboard(memory: MemoryStore, *, limit: int | None = None) -> str:
    """What each tactic has earned, per regime. The thing you actually read.

    One row per (tactic, regime): the whole point of situation features is that
    a tactic's record is not a single number, so this does not average them away.
    """
    rows = sorted(memory.entries(), key=lambda e: e.stats.mean_reward, reverse=True)
    if not rows:
        return "(nothing learned yet)"
    if limit:
        rows = rows[:limit]
    return "\n".join(
        f"  {e.tactic:<16} {e.features}  n={e.stats.trials:5.1f}  "
        f"mean excess {e.stats.mean_reward:+.4%}"
        for e in rows
    )


def walk_forward(
    prices: dict[str, list[float]],
    *,
    bars: int = 40,
    episodes: int = 10,
    stride: int | None = None,
    cash: float = 10_000.0,
    fee_bps: float = 5.0,
    lookback: int = 10,
    memory: MemoryStore | None = None,
    **trader_kw: Any,
) -> tuple[list[Any], MemoryStore, list[Market]]:
    """Run independent episodes over successive windows, sharing one memory.

    **This, and not one long run, is how to use this playbook.**
    :class:`~tactics.core.credit.DiscountedReturn` deliberately writes nothing
    until an episode ends, because it cannot know a decision's return until the
    episode is over. A single 300-bar ``pursue`` therefore learns nothing *while*
    it runs: every step consults an empty memory, UCB finds nothing tried, and
    the first tactic in the list is picked 300 times. Measured, before this
    existed — the scoreboard showed one tactic and 290 identical choices.

    So the episode is the unit of learning, and this makes episodes: each one is
    a fresh account over the next window of bars, and memory carries across.
    Walking forward is also the honest way to evaluate a strategy — every episode
    is scored on bars the policy had not seen when it learned from the last one.
    """
    stride = stride or bars
    length = len(next(iter(prices.values())))
    needed = (episodes - 1) * stride + bars
    if length < needed:
        raise ValueError(f"need {needed} bars for {episodes} episodes of {bars}, have {length}")

    results, markets = [], []
    for i in range(episodes):
        start = i * stride
        window = {s: v[start : start + bars] for s, v in prices.items()}
        market = Market(PaperBroker(window, cash=cash, fee_bps=fee_bps), lookback=lookback)
        agent = build_trader(market, memory=memory, max_steps=bars - 1, **trader_kw)
        memory = agent.memory          # first episode creates it; the rest share it
        results.append(agent.pursue(survive_goal()))
        markets.append(market)
    return results, memory, markets
