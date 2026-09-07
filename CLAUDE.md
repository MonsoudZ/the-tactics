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

### The LLM layer (v0.4) — `tactics.llm`, the judgment brain

For the work that needs a model, not a rule. Provider-agnostic and **offline-testable**.

| Concept | File | Job |
|---------|------|-----|
| `LLMClient` | `llm/client.py` | The one interface (`complete()`). `ClaudeClient` (lazy-imports `anthropic`, model `claude-opus-4-8`) for production; `ScriptedClient` for tests/demos (no key, no network). |
| `LLMTactic` | `llm/tactic.py` | A tactic whose `execute()` asks the model; returns a normal `Outcome` so it competes and learns like any other. Token usage → `Outcome.cost` (Budget caps token spend). |
| `LLMCritic` | `llm/critic.py` | An adversarial verifier — **defaults to reject when unsure, fail-closed on errors**. Gates what the colony learns and commits. |

Rules: never send `temperature`/`top_p` (Opus 4.x rejects them); request JSON via
`output_config.format`; `ClaudeClient` lazy-imports so the core installs/tests
without `anthropic` (`pip install 'tactics[llm]'` to enable). An unparseable LLM
result is a loss, not a crash. Subclass `LLMTactic.build_prompt` /
`LLMCritic.build_prompt` to feed the model real evidence.

### Verbal memory (v0.5) — lessons + the Scribe

`Memory` remembers numbers (which tactic earned what). **Lessons** remember words
(what was learned and why). Together they make learning compound across runs.

| Concept | File | Job |
|---------|------|-----|
| `Lesson` / `LessonStore` | `core/lessons.py` | One distilled insight + provenance (playbook, goal, tags, evidence). `InMemoryLessons` for tests; `JsonlLessons` for persistence (append-only JSONL — O(1) writes, greppable, hand-editable: delete a line to retract a lesson). |
| `Scribe` | `llm/scribe.py` | After a run, distills the journal + findings into at most a few strict, evidence-backed lessons. Fail-silent: an unparseable distillation writes *nothing* (a bad lesson pollutes every future prompt). |
| `LLMTactic(lessons=...)` | `llm/tactic.py` | Pass a `LessonStore` and relevant past lessons are prepended to every prompt — tactics start informed instead of cold. Lessons scoped to a *different* playbook are excluded; general (`playbook=None`) lessons apply everywhere. |

The loop: run finishes → `Scribe(client, store).distill(result, playbook=...)` →
future `LLMTactic`s wired to the same store recall what past runs learned.
Rules: the scribe records only specific, actionable, evidence-backed insights —
an empty list is a valid answer. Relevance is playbook/goal match + keyword
overlap + recency; keep lesson text short and concrete so matching works.

### The execution layer (v0.6) — the Agent SDK as a Target (`playbooks/agent_sdk.py`)

The **Claude Agent SDK** (`pip install claude-agent-sdk`) is Claude Code as a
library: tool loop, built-in Read/Write/Edit/Bash/Grep, subagents, context
compaction, permissions. It is the execution layer, and it is not worth
rebuilding. What it has no opinion about is which brief to send, whether to
believe the result, what the run cost, or what last week taught it — which is
precisely this framework. So don't compete with the harness; govern it.

| Seam | What joins | Why it matters |
|------|-----------|----------------|
| `AgentWorkspace(Target)` | `run_agent()` does the work; `verify()` measures it | Two separate methods on purpose — reward is measured, never self-reported. `runner` is injectable, so all 24 tests run with no SDK, key, or network. |
| `BriefTactic` | a brief *is* a tactic (system prompt + tool allowlist + subagent roster) | Brief shapes compete and the policy learns which wins where. `SingleAgentNarrow` · `WriteTestFirst` · `PlanThenPatch` · `ReviewedSwarm`. |
| `GateBridge` | `ctx.gate` → the SDK's **PreToolUse hook** | Every write, Bash command, and unknown tool is classified and decided by *our* gate and lands in *our* journal. `DryRun` = a colony that cannot write a byte. |
| `Outcome.cost` | `ResultMessage.total_cost_usd` | `Budget(max_cost=5.00)` is a real dollar ceiling on an autonomous swarm. |

**Never send the brief's roster as the SDK's `allowed_tools`, and never gate
through `can_use_tool`.** `allowed_tools` *grants* permission: a whole-tool entry
auto-approves that tool before any callback runs. The first live run did exactly
this and a `DryRun` colony wrote two files. `permission_mode "bypassPermissions"`
and settings-file allow rules shadow the callback the same way — so permission
goes through a PreToolUse hook (which sees every call), the roster is enforced by
`GateBridge`, and `bypassPermissions` is refused outright. Content blocks are
matched **structurally** (`name`+`input`), not by a `type` field the installed
dataclasses do not have — that read silently reported zero tool calls. Work is
measured against a **snapshot taken before the run**, because a tree that was
already dirty otherwise banks a win the agent didn't earn.

Rules: `classify_tool_call` is **fail-closed** — an unclassified tool is
irreversible + high risk, so a `PolicyGate` escalates it. Cost comes from
`total_cost_usd`, never from summing assistant `usage`: with subagents `usage`
counts only the top-level loop, so `ReviewedSwarm` would look artificially cheap
and the Budget would under-count the tactic most able to run away with the bill.
Reward is 1.0 only when the check passes *and* the diff is non-empty — a
confident summary over an empty diff is the exact failure this guards. Efficiency
lives on `cost`, not `reward`. `max_workers=1` is the default because parallel
ants would be parallel agents editing one working tree; give each its own git
worktree before raising it.

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
python3 -m pip install -e '.[llm]'     # install with the Claude-backed LLM layer
python3 -m pip install -e '.[agent-sdk]'  # install with the Claude Agent SDK execution layer
python3 -m pytest                      # run tests (keep green)
python3 examples/lead_finder_demo.py   # see the single loop learn
python3 examples/swarm_demo.py         # see the colony swarm, verify, reinforce
python3 examples/safety_demo.py        # see budgets, the approval gate, the journal
python3 examples/llm_demo.py           # LLM tactics + LLM critic (offline, scripted)
python3 examples/agent_sdk_demo.py     # the framework governing a Claude Code agent (offline)
```

## Known limitations (audited, accepted for now)

- **Recency + persistence don't combine yet.** `RecencyStore` is in-memory only;
  `JsonStore` doesn't decay. Trading (non-stationary *and* persistent) will want
  a persistent recency store — build it with the trading playbook.
- **`JsonStore` flushes on every `record`.** Atomic and safe, but O(n) per write;
  fine for thousands of entries, revisit for very large memories.
- **`ScriptedClient` isn't thread-safe** (its response index races). It's a
  test/demo helper — run LLM colony demos with `max_workers=1`. `ClaudeClient`
  (the production path) is fine in parallel.
- **`agent_sdk._sdk_runner` is now live-verified against SDK 0.2.152** (CLI 2.1.263),
  on a scratch clone, in all three gate postures: `DryRun` → 6 tool calls, 4 held,
  0 bytes written; `AutoApprove` → 15 calls, 2 files written, suite green, $0.20;
  `PolicyGate` → edits committed, all 8 Bash calls held, still verified green by
  *our* check rather than the agent's. That first run also found three real bugs
  (allowed-tools shadowing, `type`-field block parsing, absolute-dirtiness reward),
  each now covered by a regression test that fails against the pre-fix code.
  Still unexercised: `agents=` subagent rosters (`ReviewedSwarm`) and `max_turns`
  have never run live — `ReviewedSwarm` is the one whose cost accounting most
  depends on `total_cost_usd` being right.
- **The single `Agent` loop isolates tactic errors but not `observe()`/goal-predicate
  errors.** The `Colony` (the production path) isolates everything. Keep `Target.observe`
  and `Goal.is_satisfied` total/non-throwing.

## Status

- [x] Core loop, memory (in-memory + JSON), UCB + epsilon-greedy policies, tests.
- [x] Generalization across situations (`SimilarityEstimator`).
- [x] Delayed credit assignment (`DiscountedReturn`).
- [x] Colony layer: blackboard + pheromones, planner, critic, parallel swarm.
- [x] Trust layer: failure isolation, budgets, approval gate, journal, recency memory.
- [x] LLM layer: Claude-backed tactics + adversarial LLM critic (`tactics.llm`).
- [x] Verbal memory: `LessonStore` (JSONL-persistent) + `Scribe` distillation +
      lesson injection into `LLMTactic` prompts (`core/lessons.py`, `llm/scribe.py`).
- [~] Playbook: focumate (Rails + Swift) — prod-readiness audits built
      (`playbooks/focumate.py`: tests, RuboCop, Brakeman, bundler-audit, migrations,
      secret scan, committed-key check; Swift build/test/lint). Run against the repo
      when it's in session scope. Next: gated fix tactics + front/back contract check.
- [x] Playbook: repo-health — audit any repo (tests/lint/secrets) via shell-out
      (`playbooks/repo_health.py`); the reusable base the focumate audits build on.
- [x] Playbook: agent-sdk — the Claude Agent SDK as a governed Target: competing
      briefs, `can_use_tool` → approval gate, dollar-denominated Budget, verified
      reward (`playbooks/agent_sdk.py`). Next: git-worktree fan-out so briefs can
      run in parallel, and `JsonStore` + `Scribe` wired in so briefs compound.
- [ ] Playbook: trading — alerts, shift/buy/sell against a goal.
- [x] Playbook: lead-finder — score bad sites, draft + send outreach (`playbooks/lead_finder.py`, dry-run by default).
- [ ] Playbook: gift-cards — production-readiness checks.

Build them one at a time. Each new playbook should leave this checklist and the
contracts above true.
