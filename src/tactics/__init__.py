"""tactics — a goal-driven agent framework whose tactics learn from outcomes.

The whole framework is one loop:

    Goal -> pick a Tactic -> act on a Target -> measure the Outcome -> learn -> repeat

Every domain (a Rails app, a trading account, a lead list, a website) plugs in by
implementing a `Target`. Every unit of work is a `Tactic`. The `Agent` runs the
loop; the `Policy` decides which tactic to try next using what `Memory` has learned
from past `Outcome`s — judged through an `Estimator` (which can generalize across
similar situations) and credited through a `CreditAssigner` (which can spread a
delayed reward back over the moves that earned it).

The `colony` layer (``tactics.colony``) runs many tactics in parallel over a
shared `Blackboard`, coordinated by pheromones, planned by a `Planner`, and
verified by a `Critic`.
"""

from .core.context import Context
from .core.credit import CreditAssigner, DiscountedReturn, ImmediateCredit, Record, TrajectoryStep
from .core.engine import Agent, RunResult, Step
from .core.estimator import (
    Estimate,
    Estimator,
    ExactEstimator,
    SimilarityEstimator,
    feature_similarity,
)
from .core.goal import Goal
from .core.memory import Entry, InMemoryStore, JsonStore, MemoryStore, TacticStats
from .core.outcome import Outcome
from .core.policy import EpsilonGreedyPolicy, Policy, UCBPolicy
from .core.tactic import FunctionTactic, Tactic
from .core.target import Target

__all__ = [
    "Agent",
    "Context",
    "CreditAssigner",
    "DiscountedReturn",
    "Entry",
    "Estimate",
    "Estimator",
    "EpsilonGreedyPolicy",
    "ExactEstimator",
    "FunctionTactic",
    "Goal",
    "ImmediateCredit",
    "InMemoryStore",
    "JsonStore",
    "MemoryStore",
    "Outcome",
    "Policy",
    "Record",
    "RunResult",
    "SimilarityEstimator",
    "Step",
    "Tactic",
    "TacticStats",
    "Target",
    "TrajectoryStep",
    "UCBPolicy",
    "feature_similarity",
]

__version__ = "0.2.0"
