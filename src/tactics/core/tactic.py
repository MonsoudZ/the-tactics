"""The Tactic — a single, reusable strategy that "does its job."

A tactic is the pluggable unit of the whole framework. Each one:

  * declares whether it ``is_applicable`` to the current situation, and
  * ``execute``s, returning an :class:`Outcome` that scores how it did.

Tactics never choose *when* they run (the Policy does) and never know about each
other. That isolation is what lets you add a new tactic later and have it slot in
without touching anything else — it just starts competing on its results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .context import Context
from .outcome import Outcome


class Tactic(ABC):
    """Base class for all tactics.

    Subclass it, set ``name`` (or let it default to the class name), and implement
    ``execute``. Override ``is_applicable`` to opt out of situations you can't help.
    """

    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name
        if not self.name:
            self.name = type(self).__name__

    def is_applicable(self, ctx: Context) -> bool:
        """Whether this tactic can act on the given context. Default: always."""
        return True

    @abstractmethod
    def execute(self, ctx: Context) -> Outcome:
        """Do the work and report the outcome."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Tactic {self.name}>"


class FunctionTactic(Tactic):
    """Wrap a plain function ``(ctx) -> Outcome`` as a tactic.

    Handy for quick tactics and tests without declaring a class.
    """

    def __init__(self, name, fn, applies=None):  # noqa: ANN001 - thin adapter
        super().__init__(name=name)
        self._fn = fn
        self._applies = applies

    def is_applicable(self, ctx: Context) -> bool:
        return True if self._applies is None else bool(self._applies(ctx))

    def execute(self, ctx: Context) -> Outcome:
        return self._fn(ctx)
