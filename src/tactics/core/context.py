"""The Context — a snapshot the Agent hands to every Tactic on each step.

It bundles three things a tactic needs:

  * ``target``   — the live domain object, so a tactic can read or act on it.
  * ``goal``     — what we're pursuing right now.
  * ``data``     — the raw state the Target observed this step.
  * ``features`` — the small, hashable summary used to *bucket learning*.

Why ``features`` matters: the policy learns "tactic X works well when the
situation looks like Y." ``signature()`` turns ``features`` into that "Y" key,
so a tactic's track record is kept per kind-of-situation, not globally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .goal import Goal
    from .target import Target


@dataclass
class Context:
    target: "Target"
    goal: "Goal"
    data: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    # When a Colony worker runs a tactic, the Task it's working is attached here.
    # Typed loosely so the domain-free core never imports the colony layer.
    task: Any = None

    def signature(self) -> str:
        """Stable key for the learning bucket this context belongs to.

        Combines the goal name with the observed features so a tactic's stats are
        scoped to "this goal, this kind of situation."
        """
        payload = {"goal": self.goal.name, "features": self.features}
        return json.dumps(payload, sort_keys=True, default=str)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
