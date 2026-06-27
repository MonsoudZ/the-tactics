"""tactics — a goal-driven agent framework whose tactics learn from outcomes.

The whole framework is one loop:

    Goal -> pick a Tactic -> act on a Target -> measure the Outcome -> learn -> repeat

Every domain (a Rails app, a trading account, a lead list, a website) plugs in by
implementing a `Target`. Every unit of work is a `Tactic`. The `Agent` runs the
loop; the `Policy` decides which tactic to try next using what `Memory` has learned
from past `Outcome`s.
"""

from .core.context import Context
from .core.goal import Goal
from .core.memory import InMemoryStore, JsonStore, MemoryStore, TacticStats
from .core.outcome import Outcome
from .core.policy import EpsilonGreedyPolicy, Policy, UCBPolicy
from .core.engine import Agent, RunResult, Step
from .core.tactic import Tactic
from .core.target import Target

__all__ = [
    "Agent",
    "Context",
    "EpsilonGreedyPolicy",
    "Goal",
    "InMemoryStore",
    "JsonStore",
    "MemoryStore",
    "Outcome",
    "Policy",
    "RunResult",
    "Step",
    "Tactic",
    "TacticStats",
    "Target",
    "UCBPolicy",
]

__version__ = "0.1.0"
