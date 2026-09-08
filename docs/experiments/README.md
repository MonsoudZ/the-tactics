# Does a lesson change what the agent does?

The question the verbal-memory half of the framework rests on, run properly.

## Setup

`midrepo` (rebuildable from `run_experiment.py`'s sibling notes): a module-level
handler registry that leaks between tests. `tests/test_isolation.py` fails
because `test_dispatch.py` left a handler behind — and the obvious fix, clearing
the registry in an autouse fixture, then breaks `tests/test_plugins.py`, whose
built-ins register at import time. Unaided pass rate ~49% at `max_turns=5`,
which is the headroom the question needs: a task agents always solve or never
solve cannot show an effect either way.

One variable. Same repo, same brief, same tactic (`SingleAgentNarrow`), same
model, `max_turns=5`, `AutoApprove`. Arms differ only in the lesson store wired
into the brief, and every trial records `lessons_in_brief` as proof the
manipulation applied.

| arm | store |
|-----|-------|
| `without` | none (control) |
| `with` | the two lessons the scribe wrote from real failures on this repo |
| `oracle` | one hand-authored mechanism lesson — an upper bound, not a result |

450 trials, 150 per arm, **interleaved** through one shuffled queue rather than
run in blocks: a 20-trial pilot swung 2/10 then 6/10 on the same arm, so
anything drifting over the run would otherwise be confounded with the arm.

## Result

| arm | passed | rate | 95% CI | ran out of turns | wrong fix | $/trial |
|-----|--------|------|--------|------------------|-----------|---------|
| no lesson | 74/150 | 0.49 | [0.41, 0.57] | 4 | 72 | 0.048 |
| scribe's lessons | 105/150 | **0.70** | [0.62, 0.77] | 10 | 35 | 0.056 |
| hand-written mechanism | 88/150 | 0.59 | [0.51, 0.66] | **45** | 17 | 0.041 |

* scribe's lessons vs control: **+20.7pp, p = 0.0004**
* hand-written vs control: +9.3pp, p = 0.13
* wrong fixes vs control: 35 vs 72 (p = 1.2e-5) and 17 vs 72 (p < 1e-6)
* ran out of turns, hand-written vs either other arm: p < 1e-4

**Lessons work, and the mechanism is legible.** Both lessons roughly halve or
better the rate of wrong fixes — that is the diagnostic value, and it is large.
The hand-written one reduces wrong fixes *most* (72 → 17) because it states the
answer outright, but it pushes 45/150 runs into turn exhaustion with nothing
written, and the two effects very nearly cancel. The scribe's shorter lessons
capture most of the diagnostic benefit without spending the budget to get it.

The naive expectation — a more detailed lesson is a better lesson — is the one
thing here the data contradicts.

## Caveats

One repo, one task shape, one model, one turn budget. The turn budget is the
moderator that decides whether a detailed lesson pays, and it was fixed at 5.

There is real drift within the run: turn exhaustion rose in the second half for
both lesson arms (hand-written 14/72 → 31/78, p = 0.008; scribe's 0/72 → 10/78,
p = 0.002), most likely load from six concurrent agents over 28 minutes.
Interleaving keeps this from biasing the arm comparison in expectation, but the
size of the hand-written arm's penalty is partly a property of how tight the
budget effectively was, not of the lesson alone.

The `with` arm is a bundle of two lessons, one of which advises raising the turn
budget — advice the agent cannot act on. Its effect is therefore a lower bound
on what the mechanism lesson alone would do.

## Reproducing

```bash
python run_experiment.py <repo> <python> <n-per-arm> <max_turns> results.jsonl
python analyse.py results.jsonl
```

`lesson_efficacy_450.jsonl` is the raw data behind every number above.
