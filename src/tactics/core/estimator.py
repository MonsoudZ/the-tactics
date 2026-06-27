"""Estimator — how a tactic's value in *this* situation is judged.

The Policy doesn't read memory directly; it asks an Estimator "given everything
you've learned, how good is tactic T here, and how sure are you?" That indirection
is what lets us add **generalization across situations** without touching the
policy or the engine.

  * :class:`ExactEstimator` — only counts experience from the identical situation
    (same goal + same features). Fast, simple, the default.
  * :class:`SimilarityEstimator` — borrows experience from *similar* situations,
    weighted by how alike their features are. So a tactic that worked when
    ``{"volatile": True, "load": 0.8}`` can inform a brand-new
    ``{"volatile": True, "load": 0.75}`` instead of starting from zero.

The similarity kernel is domain-free: it compares feature dicts generically
(numeric closeness for numbers, equality for everything else), so it works for
any playbook without configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .context import Context
from .memory import MemoryStore


@dataclass
class Estimate:
    """A value judgment for one tactic in one situation."""

    mean: float  # expected reward
    trials: float  # evidence behind it (fractional when similarity-weighted)


def value_similarity(a: Any, b: Any) -> float:
    """Similarity of two feature values in [0, 1]. Domain-free."""
    a_num = isinstance(a, (int, float)) and not isinstance(a, bool)
    b_num = isinstance(b, (int, float)) and not isinstance(b, bool)
    if a_num and b_num:
        denom = max(abs(a), abs(b), 1.0)
        return max(0.0, 1.0 - abs(a - b) / denom)
    return 1.0 if a == b else 0.0


def feature_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Similarity of two feature dicts in [0, 1], averaged over the union of keys.

    Missing keys count as fully dissimilar on that dimension, so contexts that
    describe different things don't masquerade as alike. Empty-vs-empty is 1.0.
    """
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    total = 0.0
    for k in keys:
        if k in a and k in b:
            total += value_similarity(a[k], b[k])
        # else contributes 0.0
    return total / len(keys)


class Estimator(ABC):
    @abstractmethod
    def estimate(self, tactic_name: str, ctx: Context, memory: MemoryStore) -> Estimate: ...

    @abstractmethod
    def total(self, ctx: Context, memory: MemoryStore) -> float:
        """Total evidence across all tactics for this situation (UCB needs it)."""


class ExactEstimator(Estimator):
    def estimate(self, tactic_name: str, ctx: Context, memory: MemoryStore) -> Estimate:
        st = memory.stats(tactic_name, ctx.signature())
        return Estimate(mean=st.mean_reward, trials=float(st.trials))

    def total(self, ctx: Context, memory: MemoryStore) -> float:
        return float(memory.total_trials(ctx.signature()))


class SimilarityEstimator(Estimator):
    """Kernel-weighted generalization over situations within the same goal.

    ``sharpness`` controls how quickly influence falls off with dissimilarity
    (higher = pickier, closer to exact match). ``floor`` ignores weak matches.
    """

    def __init__(self, sharpness: float = 4.0, floor: float = 0.05) -> None:
        self.sharpness = sharpness
        self.floor = floor

    def _weight(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        w = feature_similarity(a, b) ** self.sharpness
        return w if w >= self.floor else 0.0

    def estimate(self, tactic_name: str, ctx: Context, memory: MemoryStore) -> Estimate:
        goal = ctx.goal.name
        feats = ctx.features
        weighted_reward = 0.0
        weighted_trials = 0.0
        for e in memory.entries():
            if e.tactic != tactic_name or e.goal != goal:
                continue
            w = self._weight(feats, e.features)
            if w <= 0:
                continue
            weighted_reward += w * e.stats.total_reward
            weighted_trials += w * e.stats.trials
        mean = weighted_reward / weighted_trials if weighted_trials else 0.0
        return Estimate(mean=mean, trials=weighted_trials)

    def total(self, ctx: Context, memory: MemoryStore) -> float:
        goal = ctx.goal.name
        feats = ctx.features
        total = 0.0
        for e in memory.entries():
            if e.goal != goal:
                continue
            total += self._weight(feats, e.features) * e.stats.trials
        return total
