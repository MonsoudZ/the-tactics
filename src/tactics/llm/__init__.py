"""LLM-backed tactics and critic — the judgment brain (Tier 3).

The judgment-heavy work — "is this a *real* security bug?", "is this outreach
email any good?", "is this a sane trade?" — wants a model, not a rule. This layer
adds that, while keeping the framework's safety posture: an LLM tactic still
returns a normal `Outcome` (so it competes and learns like any other), and the
`LLMCritic` is an adversarial verifier that gates what the colony trusts.

It is provider-agnostic and offline-testable by design:

  * `LLMClient` — the one interface everything depends on (`complete(...)`).
  * `ClaudeClient` — talks to Claude via the official `anthropic` SDK (lazy
    import, so the framework installs and tests without it).
  * `ScriptedClient` — deterministic canned responses for tests and demos; no
    network, no API key.

Cost flows through naturally: each call reports tokens as `Outcome.cost`, so a
`Budget(max_cost=...)` caps token spend across the swarm.
"""

from .client import ClaudeClient, LLMClient, LLMResponse, ScriptedClient, extract_json
from .critic import LLMCritic
from .scribe import Scribe
from .tactic import LLMTactic

__all__ = [
    "ClaudeClient",
    "LLMClient",
    "LLMResponse",
    "LLMCritic",
    "LLMTactic",
    "Scribe",
    "ScriptedClient",
    "extract_json",
]
