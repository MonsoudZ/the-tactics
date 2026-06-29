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
"""

from __future__ import annotations

import math
import random
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
