"""Policy — how the agent decides which tactic to try next.

This is the explore/exploit brain. It balances using what already works
(exploitation) against trying things that might be better (exploration). Because
it reads only from :class:`MemoryStore`, the same policy works for every domain.

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
    formula takes over. ``c`` controls how strongly novelty is rewarded.
    """

    def __init__(self, c: float = 1.4) -> None:
        self.c = c

    def choose(self, tactics: Sequence[Tactic], ctx: Context, memory: MemoryStore) -> Tactic:
        sig = ctx.signature()
        total = memory.total_trials(sig)
        best: Tactic | None = None
        best_score = -math.inf
        for tactic in tactics:
            st = memory.stats(tactic.name, sig)
            if st.trials == 0:
                return tactic  # explore the unknown first
            bonus = self.c * math.sqrt(math.log(total + 1) / st.trials)
            score = st.mean_reward + bonus
            if score > best_score:
                best_score, best = score, tactic
        assert best is not None
        return best


class EpsilonGreedyPolicy(Policy):
    """With probability ``epsilon`` pick a random tactic, else the best so far.

    Pass a seeded ``rng`` (``random.Random(0)``) for reproducible runs/tests.
    """

    def __init__(self, epsilon: float = 0.1, rng: random.Random | None = None) -> None:
        self.epsilon = epsilon
        self.rng = rng or random.Random()

    def choose(self, tactics: Sequence[Tactic], ctx: Context, memory: MemoryStore) -> Tactic:
        sig = ctx.signature()
        untried = [t for t in tactics if memory.stats(t.name, sig).trials == 0]
        if untried:
            return self.rng.choice(untried)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(tactics))
        return max(tactics, key=lambda t: memory.stats(t.name, sig).mean_reward)
