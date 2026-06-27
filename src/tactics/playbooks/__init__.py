"""Playbooks — domain-specific bundles of Targets and Tactics.

Each real use case (focumate hardening, trading, lead-finder, gift-card launch)
becomes a module in here: one `Target` that plugs into the domain plus the
`Tactic`s that act on it. The core engine never changes; you only add playbooks.

See ``examples/lead_finder_demo.py`` for a complete, runnable shape to copy.
"""
