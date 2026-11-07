"""EXPLORATORY (2026-11-07 ~07:00, after the six held-out domains were scored): candidate producer x listener
scores on all eleven settings' evaluation days. Anything chosen here must be re-tested on data collected later.
    python scripts/www_method_explore.py → outputs/interaction_experiment/www_eval/method_explore.json
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py")
BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF); BF.ECAP = 10**9
from collections import Counter

def yld(cells, days, f):
    tot = 0
    for d in days:
        keys = [k for k in cells if k[1] == d]
        sel = sorted(keys, key=lambda k: (-f(cells[k], k), k[0]))[:3]
        tot += len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
    return tot

def rankmix(cells, days, fs, ws):
    R = {}
    for d in days:
        keys = [k for k in cells if k[1] == d]
        if not keys: continue
        sc = sum(w * rankdata([f(cells[k]) for k in keys]) for f, w in zip(fs, ws))
        R.update(dict(zip(keys, sc)))
    return lambda c, k: R.get(k, -1)

SS = BF.MECH.settings(False)
names = [n for n in SS if not n.endswith("_terms")]
res = {}
for n in names:
    S = SS[n]; cells = BF.prepare(S, {}, {})
    for c in cells.values():
        au = Counter(S["info"][s]["author_id"] for s in c["posts"])
        ae = Counter()
        for s, e in zip(c["posts"], c["e"]):
            ae[S["info"][s]["author_id"]] += e
        c["Aeng"] = sum(1 for a in ae if ae[a] > 0)            # authors with any reply
        c["Acap"] = sum(min(v, 5) for v in ae.values())          # replies capped per author
        c["sqrtA"] = sum(np.sqrt(v) for v in ae.values())       # concave per author
        c["Elog"] = sum(np.log1p(v) for v in c["e"])            # concave per post
    ev = S["eval"]
    E = lambda c, k=None: c["E"]; A = lambda c, k=None: c["A"]
    r = {"posts": yld(cells, ev, lambda c, k: c["V"]), "authors": yld(cells, ev, lambda c, k: c["A"]),
         "edges": yld(cells, ev, lambda c, k: c["E"]), "oracle": yld(cells, ev, lambda c, k: len(c["T"])),
         "A_engaged": yld(cells, ev, lambda c, k: c["Aeng"]),
         "author_cap5": yld(cells, ev, lambda c, k: c["Acap"]),
         "author_sqrt": yld(cells, ev, lambda c, k: c["sqrtA"]),
         "post_log": yld(cells, ev, lambda c, k: c["Elog"]),
         "habitual": yld(cells, ev, lambda c, k: c["hist_y"])}
    for b in [0.25, 0.5, 1.0]:
        r[f"A*rate^{b}"] = yld(cells, ev, lambda c, k, b=b: c["A"] * (1 + c["E"] / c["V"]) ** b)
    for w in [0.5]:
        r["rank(A)+rank(E)"] = yld(cells, ev, rankmix(cells, ev, [lambda c: c["A"], lambda c: c["E"]], [1, 1]))
    res[n] = r
    o = r["oracle"]
    print(n, {k: round(v / o, 3) for k, v in r.items()}, flush=True)
Path("outputs/interaction_experiment/www_eval/method_explore.json").write_text(json.dumps(res, indent=1))
