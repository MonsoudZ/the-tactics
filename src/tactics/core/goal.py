"""The Goal — what we are trying to achieve, and how we know we got there."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # avoid import cycle; Context only needed for typing
    from .context import Context


def _never(_: "Context") -> bool:
    return False


@dataclass
class Goal:
    """A target state plus a predicate that recognizes when we've reached it.

    ``is_satisfied`` receives the current :class:`Context` and returns True when
    the goal is met. If you leave it as the default, the Agent simply runs until
    it exhausts its step budget — useful for open-ended pursuits like trading,
    where "done" is a moving target rather than a fixed line.
    """

    name: str
    description: str = ""
    is_satisfied: Callable[["Context"], bool] = field(default=_never, repr=False)
    metric: str | None = None  # optional key in Outcome.metrics worth tracking

    def satisfied_by(self, ctx: "Context") -> bool:
        return bool(self.is_satisfied(ctx))
