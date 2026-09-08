"""150 trials per arm, interleaved.

Arms are round-robined through one shuffled queue rather than run in blocks: the
n=20 pilot swung 2/10 then 6/10 on the same arm, so anything that drifts over
the run (load, model routing) would otherwise be confounded with the arm.
"""
import dataclasses, json, pathlib, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

from tactics import AutoApprove, Context, Goal, Journal
from tactics.colony.blackboard import Task
from tactics.playbooks.agent_sdk import AgentWorkspace, SingleAgentNarrow, brief_lessons

REPO, PY_BIN, N, TURNS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
OUT = pathlib.Path(sys.argv[5])
ORACLE = pathlib.Path(__file__).parent / "oracle_lessons"
TASK = "make the test suite pass"

STORES = {"without": None, "with": brief_lessons(REPO), "oracle": brief_lessons(str(ORACLE))}
EXPECT = {"without": 0, "with": 2, "oracle": 1}
for arm, store in STORES.items():
    got = len(list(store.entries())) if store else 0
    if got != EXPECT[arm]:
        raise SystemExit(f"ABORT: arm {arm!r} store holds {got} lessons, expected {EXPECT[arm]}")
print(f"stores verified: { {a: (len(list(s.entries())) if s else 0) for a, s in STORES.items()} }",
      flush=True)

ws = AgentWorkspace(REPO, check=[PY_BIN, "-m", "pytest", "-q"], isolate=True)
write_lock, wt_lock = threading.Lock(), threading.Lock()
done = [0]
t_start = time.time()


def trial(job):
    arm, i = job
    session = ws.session(Task(id=f"{arm}{i}", description=TASK))
    ctx = Context(target=session, goal=Goal(name="delivery"), data={}, features={},
                  task=Task(id=f"{arm}{i}", description=TASK),
                  gate=AutoApprove(), journal=Journal())
    t0 = time.time()
    try:
        tactic = SingleAgentNarrow(lessons=STORES[arm])
        tactic.spec = dataclasses.replace(tactic.spec, max_turns=TURNS)
        outcome = tactic.execute(ctx)
        patch = session.capture_patch()
    finally:
        with wt_lock:  # git worktree metadata is not safe to mutate concurrently
            ws.run(["git", "worktree", "remove", "--force", session.path])

    row = {
        "arm": arm, "i": i, "seconds": round(time.time() - t0),
        "success": bool(outcome.success),
        "cost": outcome.metrics.get("cost_usd", 0.0),
        "out_of_turns": "no work done" in outcome.notes,
        "untouched": "untouched" in outcome.notes,
        "wrong_fix": "check failed" in outcome.notes,
        "files": [f for f in patch.files if "pycache" not in f],
        "lessons_in_brief": next((e.data["count"] for e in ctx.journal.events
                                  if e.kind == "lessons.recalled"), None),
        "t": round(time.time() - t_start),
    }
    with write_lock:
        with OUT.open("a") as fh:      # append: a crash costs the trial, not the run
            fh.write(json.dumps(row) + "\n")
        done[0] += 1
        if done[0] % 25 == 0:
            el = time.time() - t_start
            print(f"  {done[0]}/{N*3} in {el/60:.1f}m "
                  f"(eta {el/done[0]*(N*3-done[0])/60:.0f}m)", flush=True)
    return row


jobs = [(arm, i) for arm in STORES for i in range(N)]
random.Random(0).shuffle(jobs)
try:
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(trial, jobs))
finally:
    ws.cleanup()
print(f"\nDONE {done[0]} trials in {(time.time()-t_start)/60:.1f} min", flush=True)
