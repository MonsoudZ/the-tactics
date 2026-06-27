# the-tactics

A goal-driven agent framework whose **tactics learn from outcomes** and **snap
together**. You plug it into anything — a Rails backend, a trading account, a
lead list, a website — and it pursues a goal, learns what works from real
results, and shifts toward it.

It's all one loop:

```
Goal → pick a Tactic → act on a Target → measure the Outcome → learn → repeat
```

## Why one loop covers everything

| Use case            | Goal                   | Tactics                              | Outcome (the reward) |
|---------------------|------------------------|--------------------------------------|----------------------|
| **focumate** (Rails+Swift) | production-ready  | security audit, bug hunt, contract-check | vulns fixed / tests passing |
| **Trading**         | hit target return      | alert, shift, buy, sell rules        | realized P&L vs goal |
| **Lead finder**     | win clients            | score bad sites, draft outreach      | replies / deals closed |
| **Gift cards**      | public-ready site      | harden, fix, launch checks           | works in production |

Same engine. Each domain is just a new **playbook** (a `Target` + its `Tactic`s).
The core never changes.

## The pieces

- **`Goal`** — what you want, plus how to tell you got there.
- **`Target`** — the domain you plug in. The one class you write per use case.
- **`Tactic`** — a single reusable strategy that does its job and reports a result.
- **`Outcome`** — `success` + a `reward` the system maximizes. This is the learning signal.
- **`Policy`** — decides which tactic to try next, balancing what works against what's untried (UCB or epsilon-greedy).
- **`Memory`** — records every outcome per tactic and situation. Use `JsonStore` to keep learning across runs.
- **`Agent`** — runs the loop.

## Quickstart

```bash
python3 -m pip install -e .
python3 -m pytest                      # 14 tests, all green
python3 examples/lead_finder_demo.py   # watch it learn the winning email
```

The demo gives three cold-email tactics hidden reply-rates the agent can't see.
It discovers the best one purely from outcomes:

```
free_audit_offer   chosen 193x   learned reply-rate 32.6%   ← the real winner
generic_followup   chosen  58x   learned reply-rate 12.1%
blunt_pitch        chosen  49x   learned reply-rate  8.2%
```

## Build a new playbook in ~30 lines

```python
from tactics import Agent, Goal, JsonStore, Outcome, Tactic, Target

class GiftCardSite(Target):
    name = "giftcards"
    def observe(self) -> dict:
        return {"tests_passing": run_test_suite(), "https": tls_ok()}

class FixFailingTests(Tactic):
    def is_applicable(self, ctx): return not ctx.get("tests_passing")
    def execute(self, ctx) -> Outcome:
        fixed = repair_until_green()
        return Outcome(success=fixed, reward=1.0 if fixed else 0.0)

goal = Goal("launch", is_satisfied=lambda ctx: ctx.get("tests_passing") and ctx.get("https"))
agent = Agent(GiftCardSite(), [FixFailingTests()], memory=JsonStore(".tactics/giftcards.json"))
print(agent.pursue(goal).summary())
```

See **`CLAUDE.md`** for the architecture rules and the step-by-step recipe for
adding a domain. See `src/tactics/core/` for the (heavily commented) contracts.

## Layout

```
src/tactics/core/        the domain-free engine (Goal, Target, Tactic, Outcome, Policy, Memory, Agent)
src/tactics/playbooks/   one module per real use case (add yours here)
examples/                runnable demos
tests/                   pytest suite — keep it green
```

## Status

Core engine is built, tested, and proven to learn. Domain playbooks
(focumate, trading, lead-finder, gift-cards) are next — built one at a time, each
slotting into the same loop. Progress tracked in `CLAUDE.md`.
