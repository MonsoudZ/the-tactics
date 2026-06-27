# the-tactics — working notes for Claude

This file is the project's memory. Read it first. It tells you how every new piece
must fit so the whole stays coherent. Keep it updated when the architecture moves.

## What this is

A goal-driven agent framework. One loop runs everything:

```
Goal → pick a Tactic → act on a Target → measure the Outcome → learn → repeat
```

Nothing in `src/tactics/core/` knows about any specific domain. Domains plug in
from the outside. That separation is the whole point: you can add a new tactic or
a new domain without editing the core, and it just starts competing on results.

## The five contracts (do not break these)

| Concept   | File                     | Job |
|-----------|--------------------------|-----|
| `Goal`    | `core/goal.py`           | What we want + a predicate that recognizes "done". |
| `Target`  | `core/target.py`         | A domain we plug into. `observe()` returns state; you add domain methods tactics call. |
| `Tactic`  | `core/tactic.py`         | One reusable strategy. `is_applicable()` + `execute() → Outcome`. |
| `Outcome` | `core/outcome.py`        | The learning signal: `success` + `reward`. Bigger reward = better. |
| `Policy`  | `core/policy.py`         | Chooses the next tactic (explore/exploit) via an `Estimator`. |

Supporting cast: `Context` (the per-step snapshot handed to tactics), `Memory`
(records outcomes per tactic+situation; `InMemoryStore` or persistent `JsonStore`),
`Agent`/`RunResult` (`core/engine.py`, runs the loop).

### Learning upgrades (v0.2, both domain-free, both pluggable)

| Concern | File | What it does | Strategies |
|---------|------|--------------|------------|
| **Generalization across situations** | `core/estimator.py` | How a tactic's value *here* is judged. The policy asks the `Estimator`, so generalization is a swap, not a rewrite. | `ExactEstimator` (default; identical situations only) · `SimilarityEstimator` (borrows from *similar* situations via a domain-free feature kernel) |
| **Delayed credit assignment** | `core/credit.py` | Spreads a reward back over the moves that earned it. Controls *when* memory is written. | `ImmediateCredit` (default; online) · `DiscountedReturn(gamma)` (Monte-Carlo return across episodes) |

Each `Agent.pursue` call is one episode. Use `DiscountedReturn` when reward is
delayed (trading, multi-step fixes) and run many episodes; use a
`SimilarityEstimator` when situations vary continuously and cold-start hurts.

### The colony layer (v0.2) — `tactics.colony`, also domain-free

Many ants over a shared board, coordinating by pheromones. Built on the core loop.

| Concept | File | Job |
|---------|------|-----|
| `Blackboard` | `colony/blackboard.py` | Shared Tasks + Findings + **pheromones** (decay & reinforce). Thread-safe. |
| `Planner` | `colony/planner.py` | Goal → Tasks; re-runs each round to post follow-ups from Findings. |
| `Critic` | `colony/critic.py` | Verifies an Outcome before it's trusted. Learning integrity **and** the safety gate. |
| `Colony` | `colony/colony.py` | Each round: plan → claim top tasks (pheromone-biased) → run ants **in parallel** → critic verifies → learn + reinforce → evaporate. |

Important: the Critic gates **learning and task-completion**, not a side effect a
tactic already performed. For irreversible/outward-facing actions (deploy, send,
sell), write the tactic to *propose* and let the **approval gate** (below) commit.
Use `max_workers=1` for deterministic runs/tests.

### The trust layer (v0.3) — safety, control, observability, adaptation

All domain-free, all defaults backward-compatible (no Budget = no caps, gate
defaults to `AutoApprove`, a Journal is always attached).

| Concern | File | What it does |
|---------|------|--------------|
| **Failure isolation** | `core/engine.py`, `colony/colony.py` | A tactic that raises becomes `Outcome.failed` (a loss), logged — it never crashes the loop or the parallel swarm. |
| **Budgets / limits** | `core/budget.py` | Caps `max_cost` (sum of `Outcome.cost`), `max_seconds` (wall-clock), and `max_attempts_per_task` (colony gives up on a stuck task). The governor against runaway autonomy. |
| **Approval gate** | `core/approval.py` | `Proposal` + gates: `AutoApprove`, `DryRun` (review-only), `CallbackGate` (human hook), `PolicyGate` (auto-approve reversible/low-risk, escalate the rest). Tactics call `ctx.gate.submit(proposal, ctx)` to commit irreversible actions. |
| **Audit journal** | `core/journal.py` | Ordered, thread-safe event log (`step`, `verify`, `gate.commit/hold`, `error`…). `result.journal.explain()` answers "why did it do that?". |
| **Adaptive memory** | `core/memory.py` `RecencyStore` | Recency-weighted stats for non-stationary worlds (trading regimes, changing code). Old experience decays so the mean tracks what works *now*. |

**Reward vs cost vs risk** — three separate axes a tactic reports honestly:
`reward` = how well it served the goal (learning signal); `cost` = what it
consumed (drives the Budget); risk/reversibility = declared on the `Proposal` so
the gate can decide. Don't conflate them.

## How to add a new domain (a "playbook")

This is the path for focumate, trading, lead-finder, gift-cards. Always the same:

1. **Write one `Target`** in `src/tactics/playbooks/<domain>.py`. Implement
   `observe()` to read live state. Add domain methods tactics will use
   (`run_tests()`, `place_order()`, `send_email()`, `deploy()` …). Optionally
   override `features()` so the policy can learn situation-specific preferences.
2. **Write `Tactic`s** for that domain in the same module. Each `execute()` does
   real work through `ctx.target` and returns an `Outcome` whose `reward`
   reflects how well it served the goal (tests passing, P&L, replies, etc.).
3. **Define `Goal`s** with an honest `is_satisfied` predicate.
4. **Wire an `Agent`** with a `JsonStore` so learning persists between runs.
5. **Add tests** in `tests/` mirroring `tests/test_core.py`. No piece lands
   without a test.

Copy `examples/lead_finder_demo.py` as the canonical shape.

## Design rules (the "right win, not the quick win")

- **Core stays domain-free.** If you're tempted to import a domain into
  `core/`, you're holding it wrong — push it into a playbook.
- **Reward must mean something.** A tactic that returns a fake high reward
  poisons learning. Reward = the real measured result, or 0.
- **Add, don't entangle.** New tactics must not depend on other tactics. They
  compete only through outcomes in memory.
- **Tests are the contract.** Every behavior has a test. `python3 -m pytest`
  must be green before commit.
- **Persist learning.** Real playbooks use `JsonStore` so the system keeps
  getting better across runs.

## Commands

```bash
python3 -m pip install -e .            # install (editable)
python3 -m pytest                      # run tests (keep green — 43 tests)
python3 examples/lead_finder_demo.py   # see the single loop learn
python3 examples/swarm_demo.py         # see the colony swarm, verify, reinforce
python3 examples/safety_demo.py        # see budgets, the approval gate, the journal
```

## Status

- [x] Core loop, memory (in-memory + JSON), UCB + epsilon-greedy policies, tests.
- [x] Generalization across situations (`SimilarityEstimator`).
- [x] Delayed credit assignment (`DiscountedReturn`).
- [x] Colony layer: blackboard + pheromones, planner, critic, parallel swarm.
- [x] Trust layer: failure isolation, budgets, approval gate, journal, recency memory.
- [ ] Playbook: focumate (Rails + Swift) — harden, bug-hunt, contract-check.
- [ ] Playbook: trading — alerts, shift/buy/sell against a goal.
- [ ] Playbook: lead-finder — score bad sites, draft + send outreach.
- [ ] Playbook: gift-cards — production-readiness checks.
- [ ] Optional: LLM-backed tactics (Claude API) for the judgment-heavy work.

Build them one at a time. Each new playbook should leave this checklist and the
contracts above true.
