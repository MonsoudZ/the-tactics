# the-tactics

A goal-driven agent framework whose **tactics learn from outcomes** and **snap
together**. You plug it into anything — a code repository, a Claude Code agent, a
trading account, a lead list — and it pursues a goal, learns what works from real
results, and shifts toward it.

It's all one loop:

```
Goal → pick a Tactic → act on a Target → measure the Outcome → learn → repeat
```

## Why one loop covers everything

| Use case            | Goal                   | Tactics                              | Outcome (the reward) |
|---------------------|------------------------|--------------------------------------|----------------------|
| **Agent SDK**       | land a working change  | competing briefs for a Claude Code agent | tests pass + a real diff |
| **Trading**         | beat buy-and-hold      | momentum, fade, take-profit, cut-loss, stand aside | excess return vs benchmark |
| **Repo health**     | production-ready       | tests, lint, secret scan, gated fixes | checks passing / vulns fixed |
| **Lead finder**     | win clients            | score bad sites, draft outreach      | replies / deals closed |

Same engine. Each domain is just a new **playbook** (a `Target` + its `Tactic`s).
The core never changes.

## The pieces

- **`Goal`** — what you want, plus how to tell you got there.
- **`Target`** — the domain you plug in. The one class you write per use case.
- **`Tactic`** — a single reusable strategy that does its job and reports a result.
- **`Outcome`** — `success` + a `reward` the system maximizes. This is the learning signal.
- **`Policy`** — decides which tactic to try next, balancing what works against
  what's untried (UCB or epsilon-greedy). Wrap it in `WithoutReplacement` and the
  parallel workers of one round try *different* tactics instead of all agreeing.
- **`Memory`** — records every outcome per tactic and situation. `JsonStore` to
  keep learning across runs, `RecencyStore` to let stale experience fade,
  `JsonRecencyStore` when you need both, and `TimeDecayStore` when "recent"
  means wall-clock time rather than observation count.
- **`Agent`** — runs the loop.

## The colony (the "lots of little ants" part)

The single loop above is one ant. The **colony layer** (`tactics.colony`) runs a
swarm — and it's all domain-free:

- **`Blackboard`** — shared memory where ants post tasks and findings and leave
  **pheromones**: trails that fade over time and get reinforced when work pays off
  (this is *stigmergy* — coordination with no boss).
- **`Planner`** — turns a goal into tasks, and posts follow-ups as findings come in.
- **`Critic`** — verifies an outcome *before* the colony trusts it. Keeps fake
  wins out of memory, and is the safety gate for risky actions. `Verdict.done`
  separates "trustworthy" from "finished", so a verified failure is learned from
  *and* retried rather than quietly marked complete.
- **`Colony`** — each round: plan → claim the highest-value tasks (pheromone-biased)
  → run ants **in parallel** → verify → learn → reinforce → let trails evaporate.
  A `Target` may hand each ant a private view of the domain (`Target.session`), so
  parallel workers don't collide.

Two brain upgrades, both pluggable:

- **Generalization across situations** (`SimilarityEstimator`) — a tactic that
  worked in one situation informs *similar* ones, instead of starting cold.
- **Delayed credit assignment** (`DiscountedReturn`) — when the payoff comes
  later (a buy that sets up a sell), the moves that earned it get the credit.
  Note it writes nothing until an episode ends, so the *episode* is the unit of
  learning: run many short ones, not one long one.

## Trust layer (so it's safe to point at real systems)

Autonomy you can actually turn loose — all domain-free, all off by default:

- **Failure isolation** — a tactic that throws becomes a logged loss, never a
  crash that takes down the swarm.
- **Budgets** (`Budget`) — cap total cost, wall-clock time, and retries per task.
  The governor against runaway loops or spend.
- **Approval gate** (`Proposal` + `AutoApprove`/`DryRun`/`CallbackGate`/`PolicyGate`)
  — irreversible actions (deploy, send, sell) must pass a gate before they fire.
  Swap the gate to go from review-only to human-approved to fully autonomous.
- **Audit journal** (`Journal`) — every decision recorded; `result.journal.explain()`
  answers "why did it do that?".
- **Adaptive memory** (`RecencyStore`, `TimeDecayStore`) — for worlds that change
  (markets, code), old experience decays so it tracks what works *now*: by
  observation count, or by wall-clock half-life when the world keeps moving while
  the process is idle. Both persist (`JsonRecencyStore`, `JsonTimeDecayStore`).

## The judgment brain (`tactics.llm`)

For the work that needs a model, not a rule — "is this a *real* bug?", "is this
email good?". Provider-agnostic and offline-testable:

- **`LLMTactic`** — a tactic whose `execute()` asks Claude and returns a normal
  `Outcome`, so it competes and learns like any other. Token usage becomes
  `Outcome.cost`, so a `Budget` caps token spend.
- **`LLMCritic`** — an adversarial verifier that **defaults to rejecting when
  unsure** and fails closed on errors, so fake or shaky work never gets trusted.
- **`ClaudeClient`** for production; **`ScriptedClient`** for tests and demos —
  no API key, no network. Enable with `pip install 'tactics[llm]'`.

## Verbal memory: lessons and the scribe

`Memory` remembers numbers (which tactic earned what). **Lessons** remember words.
After a run, a `Scribe` distills the journal into a few short, evidence-backed
`Lesson`s; every future model-backed tactic wired to the same store gets the
relevant ones prepended to its prompt. An unparseable distillation writes
*nothing* — a fabricated lesson would pollute every prompt that recalls it.

This one is measured rather than asserted. On a calibrated repair task (49% pass
rate unaided), **450 trials, 150 per arm**:

| arm | passed | ran out of turns | wrong fix |
|-----|--------|------------------|-----------|
| no lesson | 74/150 (0.49) | 4 | 72 |
| the scribe's lessons | **105/150 (0.70)** | 10 | 35 |
| a hand-written mechanism lesson | 88/150 (0.59) | **45** | 17 |

Lessons work: **+20.7pp, p = 0.0004**. But note the third row — the most detailed
lesson cut wrong fixes hardest *and still lost*, because it pushed 45 runs into
turn exhaustion with nothing written. Lesson length trades against the turn
budget, which is the opposite of the obvious expectation. Raw data and the
writeup are in [`docs/experiments/`](docs/experiments/).

## Governing a Claude Code agent (`playbooks/agent_sdk.py`)

The **Claude Agent SDK** is the execution layer — tool loop, built-in file and
shell tools, subagents, context compaction. It has no opinion about *which* brief
to send, whether to believe the result, what it cost, or what last week taught
it. That's this framework. So don't compete with the harness; govern it.

- Competing **briefs** (`SingleAgentNarrow`, `WriteTestFirst`, `PlanThenPatch`,
  `ReviewedSwarm`) — the policy learns which shape wins where.
- The gate rides the SDK's **PreToolUse hook**, so every write, Bash command and
  unknown tool is decided by *your* gate and lands in *your* journal. `DryRun`
  becomes a colony that cannot write a byte — including inside subagents.
- **`Budget(max_cost=5.00)`** is a real dollar ceiling, read from the run's
  reported cost.
- **Fan-out**: each ant gets its own git worktree, verifies in its own tree, and
  its work comes back as a `Patch`. `ApplyBestPatch` re-verifies every candidate
  against current HEAD and lands the winner through the gate.

## Trading (`playbooks/trading.py`) — paper only

The domain where a mistake doesn't come back, so the safety rules are
load-bearing. A tactic returns an *intention*; the base class is the only thing
that turns one into an order, and only through the gate. Reward is the account's
return **minus an equal-weight buy-and-hold** — raw equity change is mostly the
market, and would score every tactic well in a bull run.

**No live broker has ever been run against this code.** `PaperBroker` is a test
instrument, not a market simulator: no partial fills, no queue position, no gaps.
The posture follows the broker — simulated fills auto-approve, anything else
defaults to `DryRun`, and `RiskLimits` refuses a breach even under `AutoApprove`.

## Quickstart

```bash
python3 -m pip install -e .
python3 -m pytest                      # 285 tests, all green
python3 examples/lead_finder_demo.py   # one ant learns the winning email
python3 examples/swarm_demo.py         # a colony hardens a service in parallel
python3 examples/safety_demo.py        # budgets, approval gate, audit journal
python3 examples/llm_demo.py           # LLM tactics + LLM critic (offline, scripted)
python3 examples/agent_sdk_demo.py     # governing a Claude Code agent (offline)
python3 examples/trading_demo.py       # gated orders, risk limits, walk-forward
```

The lead demo gives three cold-email tactics hidden reply-rates the agent can't
see. It discovers the best one purely from outcomes:

```
free_audit_offer   chosen 193x   learned reply-rate 32.6%   ← the real winner
generic_followup   chosen  58x   learned reply-rate 12.1%
blunt_pitch        chosen  49x   learned reply-rate  8.2%
```

The swarm demo runs ants in parallel; the critic filters flaky work:

```
Goal 'hardened': satisfied after 7 rounds — 12 tasks done, 0 failed
Verified fixes: 12   rejected attempts (caught by critic): 10
```

## Build a new playbook in ~30 lines

```python
from tactics import Agent, Goal, JsonStore, Outcome, Tactic, Target

class Service(Target):                    # the one class you write per domain
    name = "service"
    def observe(self) -> dict:
        return {"tests_passing": run_tests(), "tls_ok": check_tls()}

class RepairTests(Tactic):
    def is_applicable(self, ctx): return not ctx.get("tests_passing")
    def execute(self, ctx) -> Outcome:
        fixed = repair(ctx.target)        # real work, through the target
        return Outcome(success=fixed, reward=1.0 if fixed else 0.0)

goal = Goal("launch", is_satisfied=lambda ctx: ctx.get("tests_passing") and ctx.get("tls_ok"))
agent = Agent(Service(), [RepairTests()], memory=JsonStore(".tactics/service.json"))
print(agent.pursue(goal).summary())
```

See **`CLAUDE.md`** for the architecture rules and the step-by-step recipe for
adding a domain. See `src/tactics/core/` for the (heavily commented) contracts.

## Use the brain in your other projects

`tactics` is a normal pip-installable package, so any project can depend on it:

```bash
# from GitHub (pin a branch or tag)
python3 -m pip install "git+https://github.com/MonsoudZ/the-tactics.git@main"
python3 -m pip install "tactics[llm] @ git+https://github.com/MonsoudZ/the-tactics.git@main"
python3 -m pip install "tactics[agent-sdk] @ git+https://github.com/MonsoudZ/the-tactics.git@main"
```

Two ways to organize playbooks:

- **Central** — keep every playbook in this repo under `src/tactics/playbooks/`
  and run them pointed at your other projects. One toolbox, one place to learn.
- **Per-project** — each repo installs `tactics` and writes its own `Target` +
  tactics locally. The core stays domain-free; each repo brings its domain.

The brain is **Python**, but it can drive a project in *any* language: your
`Target` reaches the domain however it needs to — shell out (`subprocess` to run
Rails tests or a Ruby script), call an HTTP API, read/write files. The Ruby
lead-finder is exactly this: a Ruby tool produces `leads.csv`, and the Python
brain (`playbooks/lead_finder.py`) picks it up downstream.

**Secrets:** never hardcode keys. Read them from the environment
(`os.environ["ANTHROPIC_API_KEY"]`) and keep a gitignored `.env` (this repo's
`.gitignore` already excludes `.env`, `*.key`, `leads.csv`, etc.).

## Layout

```
src/tactics/core/        the domain-free engine (Goal, Target, Tactic, Outcome,
                         Policy, Estimator, CreditAssigner, Memory, Lessons, Agent)
                         plus the trust layer (Budget, ApprovalGate, Journal)
src/tactics/colony/      the swarm layer (Blackboard, Planner, Critic, Colony)
src/tactics/llm/         the judgment brain (LLMTactic, LLMCritic, Scribe, clients)
src/tactics/playbooks/   one module per real use case (add yours here)
examples/                runnable demos, all offline
docs/experiments/        measured results, with the raw data behind them
tests/                   pytest suite — keep it green
```

## What's verified, and what isn't

Worth stating plainly, because "the tests pass" turned out to be a weak claim:

- **Verified against the real thing.** The agent-sdk playbook has been run live
  against the Claude Agent SDK in every gate posture — including a `DryRun` swarm
  that made 21 tool calls, 6 of them inside a subagent, and wrote zero bytes.
  Parallel fan-out, patch selection, lesson recall and the scribe are all live-verified.
- **Found by running it, not by testing it.** Seventeen real bugs in that
  playbook were found by live runs while the suite was green — including a gate
  with a silent bypass, an 18x token undercount, and a colony that marked failed
  work complete. Keep the injectable seams *and* point them at the real thing
  periodically; neither alone is enough.
- **Not verified.** No live broker has ever been connected, and `max_workers`
  above 3 is unexercised. Time-decayed memory is tested against an injected
  clock rather than against real elapsed time.

## Status

Core engine, colony, trust layer, LLM layer and verbal memory are built, tested
(285 tests), and proven to learn. Six playbooks ship: agent-sdk, trading,
repo-health, focumate, lead-finder and code-review. Progress and the full
architecture notes are in `CLAUDE.md`.
