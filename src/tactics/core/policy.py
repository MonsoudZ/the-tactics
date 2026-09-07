"""Policy — how the agent decides which tactic to try next.

This is the explore/exploit brain. It balances using what already works
(exploitation) against trying things that might be better (exploration). It judges
tactics through an :class:`~tactics.core.estimator.Estimator`, so the *same* policy
gets situation-generalization for free just by swapping the estimator.

Two ready-made policies:

  * :class:`UCBPolicy` — Upper Confidence Bound. Deterministic, principled,
    tries each tactic once then favors high-mean / under-explored ones. Default.
  * :class:`EpsilonGreedyPolicy` — mostly picks the current best, occasionally
    rolls the dice. Simple and tunable via ``epsilon``.

And one that wraps either of them:

  * :class:`WithoutReplacement` — makes the ants of a *parallel* round pick
    different tactics instead of all reaching the same conclusion.
"""

from __future__ import annotations

import math
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence

from .context import Context
from .estimator import Estimator, ExactEstimator
from .memory import MemoryStore
from .tactic import Tactic


class Policy(ABC):
    @abstractmethod
    def choose(self, tactics: Sequence[Tactic], ctx: Context, memory: MemoryStore) -> Tactic:
        """Return the tactic to execute next. ``tactics`` is already filtered to
        the ones applicable in ``ctx`` and is guaranteed non-empty."""
        raise NotImplementedError

    def begin_round(self) -> None:
        """Called by the Colony before it dispatches a parallel batch.

        Default: nothing. A policy that coordinates across the concurrent callers
        of one round resets its per-round bookkeeping here.
        """


class UCBPolicy(Policy):
    """UCB1: score = mean_reward + c * sqrt(ln(total) / trials).

    Untried tactics score infinitely high, so each is tried once before the
    formula takes over. ``c`` controls how strongly novelty is rewarded. Pass a
    ``SimilarityEstimator`` to let unseen situations borrow from similar ones.
    """

    def __init__(self, c: float = 1.4, estimator: Estimator | None = None) -> None:
        self.c = c
        self.estimator = estimator or ExactEstimator()

    def choose(self, tactics: Sequence[Tactic], ctx: Context, memory: MemoryStore) -> Tactic:
        total = self.estimator.total(ctx, memory)
        best: Tactic | None = None
        best_score = -math.inf
        for tactic in tactics:
            est = self.estimator.estimate(tactic.name, ctx, memory)
            if est.trials <= 0:
                return tactic  # explore the unknown first
            bonus = self.c * math.sqrt(math.log(total + 1) / est.trials)
            score = est.mean + bonus
            if score > best_score:
                best_score, best = score, tactic
        assert best is not None
        return best


class EpsilonGreedyPolicy(Policy):
    """With probability ``epsilon`` pick a random tactic, else the best so far.

    Pass a seeded ``rng`` (``random.Random(0)``) for reproducible runs/tests.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
        estimator: Estimator | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.estimator = estimator or ExactEstimator()

    def choose(self, tactics: Sequence[Tactic], ctx: Context, memory: MemoryStore) -> Tactic:
        untried = [t for t in tactics if self.estimator.estimate(t.name, ctx, memory).trials <= 0]
        if untried:
            return self.rng.choice(untried)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(tactics))
        return max(tactics, key=lambda t: self.estimator.estimate(t.name, ctx, memory).mean)


class WithoutReplacement(Policy):
    """Make the ants of one parallel round pick *different* tactics.

    The problem this exists for: the Colony asks the policy once per ant, and
    every ant of a round asks against the same memory snapshot — so they all
    reach the same conclusion. A three-ant round returns three samples of one
    tactic where you wanted one sample each of three. Fan-out then buys
    throughput but no information, which is backwards precisely when the tactics
    are the thing you are trying to compare.

    Each call hides what has already been handed out this round, so the wrapped
    policy picks the best of what is left. When the tactics run out — more ants
    than tactics — the slate is wiped and the next ant chooses from all of them
    again, so six ants over three tactics come out two apiece rather than four
    piled onto the winner.

    This trades some exploitation for information on purpose: an ant is spent on
    a tactic that is not currently the best. That is the right trade when you are
    paying for the parallelism anyway, and the wrong one if you meant to reduce
    variance on a known winner — in which case use the inner policy directly.

    ``begin_round`` is what scopes it to a round; without a Colony to call it,
    the wrapper degrades to one cycle through the tactics and then repeats.
    """

    def __init__(self, inner: Policy) -> None:
        self.inner = inner
        self._taken: set[str] = set()
        self._lock = threading.Lock()

    def begin_round(self) -> None:
        with self._lock:
            self._taken.clear()
        self.inner.begin_round()

    def choose(self, tactics: Sequence[Tactic], ctx: Context, memory: MemoryStore) -> Tactic:
        # The whole read-choose-mark has to be atomic: two ants that both read
        # the taken set before either wrote to it would pick the same tactic,
        # which is the bug this class exists to fix. `inner.choose` is pure
        # computation over memory, so holding the lock across it is cheap.
        with self._lock:
            remaining = [t for t in tactics if t.name not in self._taken]
            if not remaining:  # more ants than tactics — start a fresh cycle
                self._taken.clear()
                remaining = list(tactics)
            chosen = self.inner.choose(remaining, ctx, memory)
            self._taken.add(chosen.name)
            return chosen
