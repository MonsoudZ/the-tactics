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
| `Target`  | `core/target.py`         | A domain we plug into. `observe()` returns state; you add domain methods tactics call. `session(task)`/`release()` optionally hand each parallel worker a private view (default: one shared world). |
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
| **Parallel exploration** | `core/policy.py` | Whether the ants of one round try *different* tactics. The policy is asked once per ant against one memory snapshot, so by default they all agree. | `WithoutReplacement(inner)` hides what a round already handed out, so a 3-ant round samples 3 tactics instead of one three times (`begin_round` scopes it; the Colony calls it) |
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
| `Critic` | `colony/critic.py` | Verifies an Outcome before it's trusted. Learning integrity **and** the safety gate. `Verdict.done` separates "trustworthy" from "finished", so a verified failure is learned from *and* retried. |
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
| journal attribution | PreToolUse `agent_type` | A swarm's audit trail says *which ant* asked — "code-reviewer subagent tool call: Bash". |
| `Target.session` | one **git worktree** per ant | `max_workers > 1` without it is refused: parallel agents in one tree make every diff unattributable. Work comes back as `target.patches`; the main repo is never written. |
| `JsonStore` + `JsonlLessons` | `<repo>/.tactics/` | `persist=True` keeps both halves of memory beside the code they describe: which brief wins here, and what past runs learned. |
| `ApplyBestPatch` | candidates → one landed change | Closes the last manual step: each candidate is re-tried against current HEAD, and only survivors compete. |

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
lives on `cost`, not `reward`.

**Fan-out (verified live: 3 agents, 62s, 3 *distinct* patches, main repo
untouched).** Set `AgentWorkspace(isolate=True)` and `max_workers > 1`; the
unsafe combination is refused, not warned about. `spread=True` (the default)
wraps the policy in `WithoutReplacement` so the round's ants try different
briefs — without it they all reach the same conclusion from the same memory
snapshot, and you buy throughput without information exactly when the briefs are
what you're comparing. Turn it off to spend a parallel round reducing variance on
the current best instead. Each ant gets a detached worktree at HEAD, verifies in
its *own* tree (so reward is attributable), and its work is lifted out as a
`Patch` before the worktree is destroyed. Landing one is a separate deliberate
act (`apply_patch`, `--3way`). Two consequences to know: the check runs *inside*
the worktree, so make it self-contained (`PYTHONPATH=$PWD/src …` — an editable
install silently measures the main tree); and since the main repo never changes,
a check-based goal never completes — that is what `work_queue_goal` is for.
`delivery_goal` is for **repair** (red → green); on a healthy repo it is
satisfied at round 0 and the colony stops having done nothing. A run the gate
held is rejected by the critic rather than learned from: it is evidence about
the gate, not about the brief.

**Compounding (`persist=True`, verified live).** Numeric memory (`JsonStore`)
records which brief wins in which situation; verbal memory (`JsonlLessons`) holds
what runs learned, and `BriefTactic._with_lessons` prepends the relevant ones to
every brief — the half a fresh agent process cannot have, however good the
harness is. Lessons go in the *brief*, never the system prompt: the system prompt
**is** the shape being measured, so varying it would make two runs of the same
tactic incomparable. `run_and_learn(colony, goal, client=...)` closes the loop —
run, then `BriefScribe` distills the journal *plus the brief scoreboard*, because
the durable lesson is almost always comparative. Verified across two processes:
run 1 wrote `SingleAgentNarrow 1.0`, run 2 read it back, explored the untried
brief, and added `WriteTestFirst 3 runs / 3.0` to the same file; briefs arrived
carrying a stored lesson. Note what UCB does here — it tries an unmeasured brief
before exploiting a proven one, so "it picked the winner" is only meaningful once
every brief has a record.

**Choosing among candidates (`ApplyBestPatch`, verified live).** A fan-out leaves
several answers to one goal in `target.patches`. The tactic re-tries every
candidate against a scratch checkout of the *current* HEAD — passing where it was
born is not the same as passing here, since HEAD may have moved or another patch
landed — and only survivors compete. The deterministic pick is the smallest
verified diff; an optional `judge` (any `LLMClient`) may re-order *those*, sees
them already ranked, and fails closed: an unparseable answer, or one naming a
candidate outside the verified set, falls back to the measured order and says so.
A model's opinion breaks ties between proven options; it is never the proof.
Landing goes through the gate (`reversible`, `risk="medium"`), so `DryRun` still
runs the whole selection and leaves a review artifact naming what *would* have
landed. Winning patch → `landed`, the rest → `discarded` (they are alternative
answers to the same goal; stacking a second one is not a merge). Note `git apply
--3way` **stages** what it applies — a landed change shows under `git diff
--cached`, not `git diff`.

**Subagents (verified live).** The PreToolUse hook fires *inside* subagents too,
carrying `agent_type` — so a brief cannot delegate its way around the gate, and
delegation itself (`Agent`, older name `Task`) is correctly classified read-only:
what the delegate then does is gated call by call. Token counts come from
`model_usage`, never `usage`, for the same reason cost does — a live swarm run
reported `usage` out:191 against an actual 3512. A call can emit more than one
`ResultMessage`; each later one carries the running total for the whole call, so
take the last and never sum. `BriefSpec.agents` without a delegation tool in
`allowed_tools` is refused before spending: that exact mismatch (roster said
`Task`, CLI wanted `Agent`) made `ReviewedSwarm` run solo while still scoring
1.0 — a wrong label on real statistics, which is worse than a failure.

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
  `ReviewedSwarm` is now live too, and found three more (delegation tool named
  `Agent` not `Task`, `usage` undercounting subagent tokens ~18x, multiple result
  messages) — all covered by tests. Under `DryRun` with a delegate-first brief:
  21 hook firings, 6 of them inside the subagent, 15 denied, 0 bytes written.
  Parallel fan-out is live too (3 agents in their own worktrees, 48s, 3 patches,
  2 distinct, main repo clean, one patch applied back with `--3way`), and found
  three more (a check-based goal satisfied at round 0, the journal entry written
  before the measurement, gate-held runs polluting a brief's statistics).
  Compounding is live too (`persist=True` across two processes; lessons reaching
  a real agent's brief), and so is per-round spread (3 ants, 3 different briefs,
  3 distinct patches, against memory that favoured one of them).
  **The scribe is live too, and settling why it wrote nothing for six runs took
  three bug fixes, none of them in the scribe.** A colony *completed* any task
  whose failure the critic verified — trustworthy was conflated with finished —
  so it stopped after one round and failures could never recur. The journal
  recorded that a check failed but never why, so a cause that repeated every
  round read as three anonymous zeroes. And `persist=True` did not wire the
  lesson store into caller-supplied tactics, so the scribe was not even called.
  With all three fixed, on a repo red in a way agents cannot fix (a test importing
  an uninstallable package), three briefs failed identically across three rounds
  and the scribe wrote two lessons: it named the recurring `ModuleNotFoundError`,
  diagnosed it as environmental rather than logical, and observed that switching
  briefs without addressing it wastes rounds. The conservatism was never the
  problem — it had been shown a tally, never a failure.

  **Recall works. This lesson changed nothing — measured, 20 trials.** A fresh
  process loads both halves off disk and the brief the real agent receives opens
  with "What past runs on this repository learned:", naming the recurring
  `ModuleNotFoundError`. That is deterministic. Whether it *changes what the
  agent does* was then run properly: same repo, same brief, same tactic, 10
  trials per arm, the only difference being whether the store was wired in
  (verified: 968-char brief vs 24-char). Result — declared the dependency 0/10
  vs 0/10; attempted an install 0/10 vs 1/10, and that one was `pip show`, an
  inspection; deleted the failing test 0/10 vs 0/10. File-level behaviour was
  identical in 20/20 (`pytest.ini`-or-`pyproject.toml` plus the `calc.py` fix),
  cost within 3%, Fisher p = 1.0 on every outcome.

  Read it carefully, though: this lesson advised installing a package that
  *cannot* be installed here, so an agent ignoring it may simply be right. The
  result is about one non-actionable lesson, not about lesson injection. And
  0/10 has a 95% upper bound near 0.28, so an effect up to ~28% would hide. The
  experiment that would actually settle the mechanism's worth needs a task
  agents fail at roughly half the time unaided, and a lesson that names the step
  they miss — measured on reward, not on behaviour. That one has not been run.
  One incidental finding worth keeping: across 20 trials under a check-based
  reward, nothing ever deleted the failing test to go green.

  Patch selection is live too (3 candidates from 3 briefs, all re-verified,
  smallest landed, suite green on the result; a real LLM judge returning a
  reasoned choice, and the fail-closed fallback firing for real when the judge
  errored). Still unexercised: `max_turns`, and `max_workers` above 3.
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
      briefs, PreToolUse hook → approval gate, dollar-denominated Budget, verified
      reward, git-worktree fan-out for parallel ants (`playbooks/agent_sdk.py`).
      `persist=True` compounds both halves of memory across runs, parallel rounds
      spread across briefs, and `ApplyBestPatch` chooses among the candidates.
      Verified end to end on a repo whose failures repeat: the scribe records
      the recurring cause and future briefs start carrying it.
- [ ] Playbook: trading — alerts, shift/buy/sell against a goal.
- [x] Playbook: lead-finder — score bad sites, draft + send outreach (`playbooks/lead_finder.py`, dry-run by default).
- [ ] Playbook: gift-cards — production-readiness checks.

Build them one at a time. Each new playbook should leave this checklist and the
contracts above true.
