"""The Target — the thing you plug the framework into.

This is the seam that lets one engine drive a Rails repo, a trading account, a
lead list, or a website. To onboard a new domain you implement exactly one class:

    class TradingAccount(Target):
        name = "trading"
        def observe(self) -> dict: ...           # read current state
        def features(self, data) -> dict: ...     # (optional) summarize for learning

Tactics reach the domain through ``Context.target``, so any methods you add to
your Target (``place_order``, ``run_tests``, ``send_email`` …) are available to
the tactics you write for it. The core never calls those — only your tactics do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Target(ABC):
    """A domain the agent acts upon."""

    name: str = "target"

    @abstractmethod
    def observe(self) -> dict[str, Any]:
        """Return a fresh snapshot of the domain's current state."""
        raise NotImplementedError

    def features(self, data: dict[str, Any]) -> dict[str, Any]:
        """Summarize a snapshot into a small, hashable dict for learning buckets.

        Default: no features, so all situations share one bucket. Override to let
        the policy learn situation-specific preferences (e.g. ``{"volatile": True}``).
        """
        return {}
