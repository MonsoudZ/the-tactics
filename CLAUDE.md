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
| **Adaptive memory** | `core/memory.py` `RecencyStore` · `TimeDecayStore` | Recency-weighted stats for non-stationary worlds. Old experience decays so the mean tracks what works *now* — by observation count, or by wall-clock half-life when the world moves while you are not looking. |

**Reward vs cost vs risk** — three separate axes a tactic reports honestly:
`reward` = how well it served the goal (learning signal); `cost` = what it
consumed (drives the Budget); risk/reversibility = declared on the `Proposal` so
the gate can decide. Don't conflate them.

### The LLM layer (v0.4) — `tactics.llm`, the judgment brain

For the work that needs a model, not a rule. Provider-agnostic and **offline-testable**.

| Concept | File | Job |
|---------|------|-----|
| `LLMClient` | `llm/client.py` | The one interface (`complete()`). **`SdkClient` needs no API key** — it goes through the Claude Agent SDK, the same authentication the agents already use; `ClaudeClient` (lazy-imports `anthropic`, model `claude-opus-4-8`) for the API path; `ScriptedClient` for tests/demos (no key, no network). |
| `LLMTactic` | `llm/tactic.py` | A tactic whose `execute()` asks the model; returns a normal `Outcome` so it competes and learns like any other. Token usage → `Outcome.cost` (Budget caps token spend). |
| `LLMCritic` | `llm/critic.py` | An adversarial verifier — **defaults to reject when unsure, fail-closed on errors**. Gates what the colony learns and commits. |

**Prefer `SdkClient` over `ClaudeClient`.** The agents in the agent-sdk playbook
authenticate through the Claude Code CLI and need no API key, so requiring one
for the Scribe put the two halves of memory on *different credentials* — the
verbal half reported "no ANTHROPIC_API_KEY" in exactly the environments where
the execution half ran fine. `SdkClient` closes that: same SDK, same auth,
nothing billed to the API. Two differences to know. It denies every tool call
through a PreToolUse hook, because a client whose job is answering questions has
no business touching the filesystem and the SDK's loop would otherwise be free
to — denial rather than declining to grant, since a grant elsewhere shadows the
callback. And `schema` becomes an *instruction* rather than a constraint (the
SDK has no `output_config.format`), parsed leniently by `extract_json`; callers
already treat an unparseable answer as a failed judgment, which is the right
posture for a shape the model was merely asked to honour. It is not cheaper per
token — the CLI sends a substantial system prompt, and a one-word live reply
cost $0.08 on 902 input tokens — but it needs no separate credential. `cli.py`
picks it first and falls back to the API path only when the SDK is absent.

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

### The trading playbook (v0.7) — `playbooks/trading.py`, money on the line

The domain that makes every safety rule load-bearing at once, and the first one
where a mistake does not come back.

| Decision | Why it is that way |
|----------|--------------------|
| A tactic returns an **intention**; `TradingTactic.execute` is the only thing that turns one into an order, through `ctx.gate` | A subclass cannot fill around the gate even by accident. Orders propose as `reversible=False, risk="high"` — it is money. |
| **Reward is benchmark-relative**: the account's return minus an equal-weight buy-and-hold of the same symbols | Raw equity change is mostly the market. In a rising market every tactic looks brilliant, and the policy learns only that bulls are nice. Excess return measures the *decision*. It also lets `StandAside` earn a positive score for holding cash through a decline — a move raw P&L cannot express. |
| **The posture follows the broker**: simulated fills auto-approve, anything else defaults to `DryRun` | A backtest that needs a human per order is not a backtest; a live venue that trades without one is a liability. The safe default is the one for the case nobody thought about. |
| `RiskLimits` wraps the posture rather than replacing it | Two questions, kept apart: *is this within the limits we set* and *is it approved*. A breach is refused even under `AutoApprove`; `max_drawdown` measures against the high-water mark, so a strategy that already lost too much stops rather than trading back. |
| **The `Agent`, not the `Colony`** | There is no git worktree for a brokerage account. Two ants on one account interleave into a position neither chose. Decisions are serial. |
| `SimilarityEstimator` by default | Regime features change most bars, so under exact matching almost every step is unseen, UCB explores instead of comparing, and the first tactic in the list wins forever. Measured: 290 identical choices in a 290-step run. |
| `walk_forward`, not one long run | `DiscountedReturn` writes nothing until an episode ends — it cannot know a return before then. A single long `pursue` therefore consults an empty memory every step and learns nothing *while* it runs. The episode is the unit of learning, so the harness makes episodes and shares memory across them. Walking forward is also the honest evaluation: each episode is scored on bars the policy had not seen. |

`PaperBroker` is a test instrument, not a market simulator: no partial fills, no
queue position, no gaps, no slippage beyond the fee. A strategy that only looks
good here has been measured against a kinder world than the real one. **No live
broker has ever been run against this code**, and the `Broker` protocol is the
only place one would go.

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
A patch is captured **before the check runs**: the check's own artifacts
(`__pycache__`, coverage data, build output) are its side effects, not the
agent's work, and a patch carrying them is bigger, unattributable, and often
will not apply. An agent that runs the suite itself can still pull artifacts in,
which is why a repo without a `.gitignore` gets a warning rather than a fix.

Reward is 1.0 only when the check passes *and* the diff is non-empty — a
confident summary over an empty diff is the exact failure this guards. Efficiency
lives on `cost`, not `reward`.

**On a green repo that is not enough, and `proves_itself` is the answer.** When
the check already passes, "it passes after the run" means only that nothing
broke — and for feature work the agent writes the test that grades its own work,
so *reward is measured, never self-reported* quietly stops being true exactly
where the tool gets used most. (The `--version` patch landed on that basis. It
was fine. That was luck.) So each candidate's **test files alone** are applied to
a clean checkout of HEAD and the check is run: it must **fail**. A test that goes
red without the implementation was measuring the implementation; one that stays
green proved nothing, and the report says so. Failing is the good outcome here,
which inverts `PatchTrial.passes` in a way easy to get backwards — the test for
a vacuous patch fails if the result is hardcoded to success. Which files are
tests is a path heuristic (`TEST_PATH`), so a repo naming them otherwise gets an
honest "nothing to prove it" rather than a confident wrong verdict. The CLI
baselines the check before starting so it can say which situation you are in.
Verified against two real agent-written patches: both proved themselves.

**The front door (`cli.py`, verified live).** `tactics <repo> "<task>"` is the
only thing standing between "a framework you wire up" and "a brain you point at
a repo" — everything it does was already possible, in about thirty lines you had
to get right. It always isolates (worktrees), defaults to a `PolicyGate` that
escalates anything irreversible (prompting at a terminal, refusing without one),
never writes the repo without `--apply`, and warns about the two setups that
silently measure the wrong thing: a `src/` layout with no pythonpath, and a repo
with no `.gitignore`. Verified end to end on a red repo: 2 agents, 1 verified
patch, landed, suite green.

The report says what the *check* found, not what the agent claimed: a live run
listed a candidate patch adding two unused helpers as though it worked, because
the output named the patch and never named the verdict. So the `verify` journal
events are printed per ant. `--show-diff` prints each candidate in full and does
not truncate — the worktree that produced it is gone by then.

The same failure had two more faces, both found by asking what a first-time user
sees. **A run that never started looked exactly like one that found nothing to
do**: with the SDK not installed, `1 run(s), $0.00 spent, 0 file(s)` and
`nothing to apply` — the error was in the journal the whole time and simply
never printed. It prints now, and when *every* run errored the report says so
plainly, names the fix where it can recognise one (`pip install
'tactics[agent-sdk]'`, or authenticate), and exits 2 rather than 1, because
nothing ran at all is not the same answer as ran and found nothing (and
"nothing ran" is decided by spend, not by the fact that every run errored:
running out of turns costs money and is a result). And **the
critic said "check re-run confirms the fix" whenever the check passed**, empty
diff included — so on an already-green repo every ant confirmed a fix that did
not exist. A pass over an empty diff now says exactly that. The reward was
always right; only the sentence lied.

**`.tactics/` ignores the half of itself nobody should commit.** Two different
things live there: patches are run artifacts (one per agent, per run, forever)
and memory + lessons are what past runs on this repository learned, which a team
may well want in git so everyone's agents start informed. So the CLI writes
`.tactics/.gitignore` with `patches/` — from *inside* the directory, so a
.gitignore the user maintains is never touched — and says once that the
remaining choice is theirs. The note stops when they have made either decision,
committing it or ignoring it, rather than nagging forever. Verified live:
`git add .tactics` stages the memory and not the patch.

**The default posture's prompt is serialized** (`_PROMPT`). `_ask` is reached
from each ant's own thread, so with `--agents` above one and no lock several
agents read the same stdin at once: prompts interleave and a "y" meant for one
is delivered to whichever thread is reading — an approval given for the wrong
action, which is the one thing a gate must never do. It also names what is being
asked for (`agent tool call: Bash  rm -rf build`), since the tool name alone is
not a question anyone can answer, and refuses on EOF rather than guessing.
Verified live: a real agent's `rm -rf build` escalated, printed the command, and
was refused for want of a terminal.

**Patches are archived before the worktree is destroyed** (verified live: a run
`kill -9`'d mid-flight printed no report at all, and its patch was on disk,
applied with plain `git apply --3way`, and turned the repo green). Ordering is
the whole feature: until `release` runs the only copy is a worktree about to be
deleted, and after it the only copy is a list in memory that a killed run never
returns. `AgentWorkspace(patch_dir=...)` names the directory; the CLI gives each
run its own (`<repo>/.tactics/patches/<stamp>`, or a temp dir under
`--no-persist` — that flag means *do not write your repo*, not *throw the work
away*), never clobbers an existing file, and is fail-soft: an unwritable archive
costs the copy, never the patch, and `saved_to` stays empty so the report can
say so rather than imply a file exists. One directory per run forever is a slow
leak, so the CLI keeps the last `--keep-runs` (default 20) and says out loud
what it deleted — these are files the previous report named by path, and a
retention nobody announces is one nobody knows about. It prunes only the archive
it manages: a `--patch-dir` you named is never deleted from, which is how "keep
everything" is spelled.

**And the leavings are reaped** (verified live). A killed run's worktrees are
both registered and on disk, which is exactly the case `git worktree prune`
will not touch, so they accumulate one full checkout at a time. Two halves:
`SIGTERM` is caught and turned into `SystemExit` so the existing `cleanup()`
still runs (Ctrl-C already unwound; `timeout`, a CI cancellation and a plain
`kill` did not), and `reap_abandoned_worktrees()` runs at startup for the case
nothing can catch. The reaper never touches a worktree the user added
themselves (only paths under a `tactics-worktrees-*` root count) and never one
a live run still holds, which is an advisory `flock` on the root rather than a
PID file: the kernel drops it however the process dies. Where locks are
unavailable it reaps nothing, because deleting a live run's tree is far worse
than leaving a dead one's. Deliberately out of scope: a root belonging to a
*different* repository, which that repo's next run owns, and an empty root left
by a run that died before its first worktree — sweeping those would mean
deleting directories this repo never registered.

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
the durable lesson is almost always comparative — and, since the 450-trial
result below, it is told to **name the cause rather than prescribe a procedure**.
`lesson_budget` caps how much recalled text precedes the task; that is a guard
against an append-only store growing forever, not a fix for the length effect
(the lesson that caused it would have fit inside the budget). Verified across two processes:
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

This is the path for focumate, trading, lead-finder, code-review. Always the same:

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
python3 examples/trading_demo.py       # gated orders, risk limits, walk-forward (offline, no broker)
tactics <repo> "<task>" --check "..."  # the front door: point the brain at any repo
```

## Known limitations (audited, accepted for now)

- **Recency, persistence and time-decay all combine.** `_JsonBacked` keeps
  persistence orthogonal to how stats update, so the four stores are two
  independent choices: `RecencyStore`/`TimeDecayStore` for *what recent means*,
  and the `Json*` variants for whether it survives a restart. Decay by
  observation is right for a backtest (replaying five years in three seconds
  should fade nothing); decay by clock is right for anything live, and fades on
  *read* as well as write so an idle store stops handing the policy stale
  confidence. `JsonTimeDecayStore` round-trips its timestamps, because decay
  that only runs while the process is alive has no opinion about the week it was
  dead. `half_life` has no default — the right value is domain-specific and
  guessing it mis-weights everything the policy reads.
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
  *cannot* be installed here, so an agent ignoring it may simply be right.

  **The calibrated experiment was then run properly: 450 trials, three arms,
  n=150 each, interleaved.** The repo (a module-level registry that leaks between
  tests, where the obvious fix breaks a second test) sits at a 49% unaided pass
  rate with `max_turns=5` — the headroom the earlier repos lacked. Full writeup
  and raw data in `docs/experiments/`.

  | arm | passed | ran out of turns | wrong fix | $/trial |
  |-----|--------|------------------|-----------|---------|
  | no lesson | 74/150 (0.49) | 4 | 72 | 0.048 |
  | scribe's lessons | **105/150 (0.70)** | 10 | 35 | 0.056 |
  | hand-written mechanism lesson | 88/150 (0.59) | **45** | 17 | 0.041 |

  **Lessons work: +20.7pp over control, p = 0.0004.** The mechanism is legible in
  the failure modes — both lessons cut wrong fixes sharply (72 → 35, p = 1.2e-5;
  72 → 17, p < 1e-6), which is the diagnostic value. But the hand-written lesson,
  which states the answer outright and reduces wrong fixes most, pushes 45/150
  runs into turn exhaustion with nothing written (p < 1e-4 against either other
  arm), and the two effects nearly cancel: +9.3pp, p = 0.13. The scribe's shorter
  lessons capture most of the diagnostic benefit without spending the budget to
  get it.

  So the naive expectation — a more detailed lesson is a better lesson — is the
  one thing the data contradicts. Lesson length trades against the turn budget.
  Note also that the earlier n=20 pilot of this same experiment showed +15pp at
  p=0.50 and concluded nothing; it was underpowered exactly as predicted, and an
  n=20 result on a 20pp effect is not evidence of absence.

  One incidental finding across all 530 trials in this playbook: under a
  check-based reward, nothing ever deleted a failing test to go green.

  Patch selection is live too (3 candidates from 3 briefs, all re-verified,
  smallest landed, suite green on the result; a real LLM judge returning a
  reasoned choice, and the fail-closed fallback firing for real when the judge
  errored).

  **`max_turns` and four agents are live now too**, on a fresh build of the
  calibrated registry repo. One variable: same repo, same brief, same model,
  `--max-turns 1` → 0 files and `Reached maximum number of turns (1)`;
  `--max-turns 10` → a verified fix, for a third of a cent more ($0.094 vs
  $0.096 — the failed run costs about what the successful one does, so a turn
  budget set too low buys nothing and saves nothing). Four agents is the whole
  roster at once: 76s, four *distinct* patches, all four verified, main repo
  clean and no worktrees left. All four wrote snapshot-and-restore rather than
  the `HANDLERS.clear()` that breaks `test_plugins` — at 10 turns they had room
  to find the trap, which is the same length effect the 450-trial experiment
  measured from the other side.

  The `--max-turns 1` run found a real bug in the report, of exactly the kind
  the previous three were: it called turn exhaustion "a setup problem, not a
  result" and exited 2. The agent *ran*; it spent $0.09 and hit a ceiling the
  user set. Spend is now the line between the two — a run that cost money
  reached the model, so its failure is an answer about this task, and only a
  run that cost nothing gets called a broken environment. The turn limit also
  gets its own hint naming the current value.
  **Pointed at its own repository, it found two more.** The `src/` warning's own
  remedy could not be followed — `--check 'PYTHONPATH=$PWD/src pytest -q'` needs
  a shell, and the check is `shlex.split` and run without one, so `argv[0]` was
  the assignment and every check died with "command not found". (The trap it
  warns about is real here: in a fresh worktree plain `pytest` imports the main
  tree's `src`.) And with three agents all hitting `--max-turns 15` *after*
  writing patches the check then verified, the report announced "every agent
  failed part-way through its run" directly above three passing verdicts — an
  error is how a run ended, not a verdict on what it produced. The run's own
  `--version` patch was landed from the patch archive, re-verified against a
  HEAD that had moved two commits since: the first agent-authored change in
  this repository, and the archive doing exactly what it exists for.

### Finding the work (v0.8) — `playbooks/survey.py`

Everything else here waits to be told what to do. This finds it, and it can
because **cleanup is more measurable than feature work, not less.** Feature work
self-grades: the agent writes the test that judges it. A refactor cannot — the
tests already exist, so the reward is *a number moving while the suite you
already had stays green*, with nothing self-reported in it. Every `Candidate`
therefore carries a `measure(tree)` and a `target`, re-measured in the ant's own
worktree exactly as the check command is. "Make it cleaner" is inexpressible
here, which is the point: a proposal that cannot fail cannot be scored, cannot
be rejected, and teaches the policy nothing.

Three finders run by default because they read the same in every language: a
long file is long (`oversized`, target as a *ratio* — "under 400" means
different things at 700 and 3000 lines), a repeated block is repeated
(`duplication`, exact-match sliding windows, three copies minimum because two
are often coincidence), a TODO is a TODO (`unfinished`).

**Every refinement in it came from being wrong on a real repo, and the failure
modes are the lesson.** Surveying this repo, `unfinished` matched its own regex
and its own docstring prose, and counted `raise NotImplementedError` — which is
just how Python spells an abstract method. Markers now have to sit in a comment
and that pattern is gone. Then surveying a real Rails API, `unreferenced`
returned **188 candidates: every controller, job and serializer in the app**,
because `resources :friends` never spells `FriendsController`. Acting on it
would have deleted a production API. So it is out of the default set entirely —
`survey(kinds=["unreferenced"])` asks for it — and it skips framework autoload
directories and dispatched class names. Its accuracy depends on whether a
codebase wires itself by name or by convention, which this module cannot detect
for you. That same Rails survey then offered `db/schema.rb` as its largest file:
generated, and rewritten by the next migration, so a file whose own header says
a tool wrote it is not work for a person.

The pattern worth keeping: a finder that cries wolf gets ignored, so narrow
beats clever, and a heuristic must be *named* as one where it can be wrong.

### The report (v0.9) — `playbooks/report.py`, the front of the whole thing

Point it at a repo and get back what the thing does, feature by feature, what is
missing, and what could go. Everything downstream — tasks, agents, verification
— should hang off this.

**Every line traces to a file, and that constraint is the design.** A report
that invents a feature is worse than no report: it reads exactly like one that
was careful, and it will be believed. So `inventory()` only extracts. Routes
come from the *booted* router, because `resources :tasks` expands to seven
routes and a regex over routes.rb reports one; if the app cannot boot, the
routes are empty and the report says so rather than listing plausible ones.
Features are the routes — that is what an application *does* — and models,
services, jobs and policies attach to them, with whatever attaches to nothing
listed as support code. A gap is an *absence the repo's own conventions make
visible* (an endpoint no spec mentions, an action opting out of the
authorization check its siblings use, a model with no policy in a Pundit app),
never "this code is bad": absence is checkable, taste is not.

`describe()` is the one part that asks a model, is kept separate for exactly
that reason, is given only the extracted facts, and is rendered as *narration*
so a reader always knows which half they are reading. A model that cannot
answer leaves the feature blank — silence beats invention.

**Both false-positive classes came from the first real run** and are regression-
tested. It reported 136 "features" for an app with 25 controllers, one per
service file — a filing system pretending to be an understanding. And it
collapsed `:id` to nothing when matching routes against specs, producing
`/lists//tasks//complete`, which no spec contains, so eight thoroughly tested
endpoints were reported untested. 65 gaps became 27.

**What is not there yet** is the third section, and the one where inventing is
most tempting. The rule that keeps it honest: a missing feature is *the
repository contradicting itself* — an action its siblings all offer, a column it
stores and never returns. Two finders survive that bar. `incomplete resource`
compares a controller against the CRUD its siblings do (only where three of five
are already present, so a single-purpose controller is not badgered into being a
resource). `stored but never returned` diffs the schema against the serializers,
skipping plumbing *and Devise's own columns*, which otherwise bury the one that
matters under six of their own. A third finder — `has_many :x` with no route
serving x — was **written and deleted**: association names and route names need
not correspond (`recent_searches` is served at `/searches/recent`,
`calendar_events` at `/calendar`), so it reported eighteen absences of which
roughly none were real. That mapping is not mechanically derivable, so it
belongs in narration. Anything beyond these is taste, and taste is labelled.

On a real app it found a **fully built feature with no API surface**: three
streak columns, a 120-line StreakService and a job enqueued on every task
completion, and not one mention of "streak" in any serializer or controller. No
client could ever see it.

**It found a real production bug on its first honest run.** The gap list said
`POST /api/v1/tasks/batch` was served with no spec mentioning it — the only
mutation endpoint in the app with zero coverage. Probing it: every request
returns 400 `Unsupported action: batch`, because `batch_params` permits
`:action` while Rails' router has already set `params[:action] = "batch"` and
path params win the merge, so `TaskBatchService` never receives the caller's
`complete`/`delete`/`move`. Bulk actions had been broken for every client, and
the absence of a test is exactly why nobody knew. The report did not guess; it
noticed an absence, and the absence was load-bearing.

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
- [x] Playbook: trading — gated orders, benchmark-relative reward, risk limits,
      walk-forward episodes, persistent recency memory (`playbooks/trading.py`).
      Paper only: the `Broker` seam has never been pointed at a live venue, and
      the defaults refuse to trade one without you saying so explicitly.
- [x] Playbook: lead-finder — score bad sites, draft + send outreach (`playbooks/lead_finder.py`, dry-run by default).

Build them one at a time. Each new playbook should leave this checklist and the
contracts above true.

A playbook earns its own module when it brings a *domain* — new state to observe,
new actions to gate, a new way to measure reward. A list of shell commands over
an existing Target is configuration, not a playbook: `focumate.py` is 74 lines of
exactly that on top of `repo_health.py`, and that is the right size for it.
