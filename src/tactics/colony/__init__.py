"""The colony layer — many small ants, coordinating without a boss.

Built on top of the core loop, not instead of it:

  * :class:`Blackboard` — shared memory. Ants post Tasks and Findings here and
    leave **pheromones** (signals that decay over time and get reinforced when a
    path pays off). This is *stigmergy*: coordination through the environment.
  * :class:`Planner` — turns a Goal into Tasks on the blackboard, and can post
    follow-up Tasks as Findings come in.
  * :class:`Critic` — verifies an Outcome before the colony trusts it. The
    immune system: stops fake wins from poisoning learning, and acts as the
    safety gate for irreversible actions.
  * :class:`Colony` — the orchestrator. Each round it claims the highest-value
    open Tasks (biased by pheromones), runs ants on them **in parallel**, lets
    the Critic verify, records learning, and reinforces what worked.

Everything here is domain-free. A playbook supplies a Target, Tactics, a Planner,
and (optionally) a Critic; the colony machinery never changes.
"""

from .blackboard import Blackboard, Finding, Task
from .critic import AcceptCritic, Critic, FunctionCritic, Verdict
from .planner import FunctionPlanner, Planner, SingleTaskPlanner
from .colony import Colony, ColonyResult

__all__ = [
    "AcceptCritic",
    "Blackboard",
    "Colony",
    "ColonyResult",
    "Critic",
    "Finding",
    "FunctionCritic",
    "FunctionPlanner",
    "Planner",
    "SingleTaskPlanner",
    "Task",
    "Verdict",
]
