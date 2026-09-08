import json, math, pathlib, sys
from collections import Counter

rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
arms = {}
for r in rows:
    arms.setdefault(r["arm"], []).append(r)
ORDER = ["without", "with", "oracle"]
LABEL = {"without": "no lesson", "with": "scribe's lessons", "oracle": "hand-written mechanism"}

def fisher(a1, a0, b1, b0):
    C = math.comb; n = a1+a0+b1+b0; r1, r2 = a1+a0, b1+b0; c1 = a1+b1
    p = lambda x: C(r1, x)*C(r2, c1-x)/C(n, c1)
    obs = p(a1); lo = max(0, c1-r2); hi = min(r1, c1)
    return sum(p(x) for x in range(lo, hi+1) if p(x) <= obs*(1+1e-9))

def wilson(k, n):
    if not n: return (0., 0.)
    z, ph = 1.96, k/n; d = 1 + z*z/n
    c = (ph + z*z/(2*n))/d
    h = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/d
    return max(0., c-h), min(1., c+h)

print(f"  {len(rows)} trials   ${sum(r['cost'] for r in rows):.2f}\n")
print(f"  {'arm':<24} {'passed':>10}  {'rate':>6}  95% CI         {'out of turns':>13} {'$/trial':>8}")
for a in ORDER:
    rs = arms.get(a, []);  n = len(rs)
    if not n: continue
    k = sum(r["success"] for r in rs); lo, hi = wilson(k, n)
    oot = sum(r["out_of_turns"] for r in rs)
    print(f"  {LABEL[a]:<24} {k:>4}/{n:<5} {k/n:>6.2f}  [{lo:.2f}, {hi:.2f}]  "
          f"{oot:>8}/{n:<4} {sum(r['cost'] for r in rs)/n:>8.4f}")

print("\n  SUCCESS RATE, pairwise (Fisher exact, two-sided)")
for a, b in (("without","with"), ("without","oracle"), ("with","oracle")):
    if a not in arms or b not in arms: continue
    ka, na = sum(r["success"] for r in arms[a]), len(arms[a])
    kb, nb = sum(r["success"] for r in arms[b]), len(arms[b])
    d = kb/nb - ka/na
    print(f"    {LABEL[a]:<24} vs {LABEL[b]:<24} {d:+.3f}   p = {fisher(ka,na-ka,kb,nb-kb):.4f}")

print("\n  RAN OUT OF TURNS, pairwise")
for a, b in (("without","with"), ("without","oracle"), ("with","oracle")):
    if a not in arms or b not in arms: continue
    ka, na = sum(r["out_of_turns"] for r in arms[a]), len(arms[a])
    kb, nb = sum(r["out_of_turns"] for r in arms[b]), len(arms[b])
    print(f"    {LABEL[a]:<24} vs {LABEL[b]:<24} {kb/nb-ka/na:+.3f}   p = {fisher(ka,na-ka,kb,nb-kb):.4f}")

print("\n  DRIFT CHECK — first half vs second half of the run, by arm")
half = max(r["t"] for r in rows)/2
for a in ORDER:
    rs = arms.get(a, [])
    if not rs: continue
    e = [r for r in rs if r["t"] <= half]; l = [r for r in rs if r["t"] > half]
    f = lambda xs: f"{sum(x['success'] for x in xs)}/{len(xs)}" if xs else "-"
    print(f"    {LABEL[a]:<24} early {f(e):>8}   late {f(l):>8}")

print("\n  FAILURE MODES")
for a in ORDER:
    rs = arms.get(a, [])
    if not rs: continue
    m = Counter("out of turns" if r["out_of_turns"] else "untouched" if r["untouched"]
                else "wrong fix" if r["wrong_fix"] else "other"
                for r in rs if not r["success"])
    print(f"    {LABEL[a]:<24} {dict(m)}")
