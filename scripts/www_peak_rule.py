import importlib.util, sys
import numpy as np
from collections import defaultdict
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
SS = BF.MECH.settings(True)
GRID = [1, 2, 3, 5, 10, 20, 10 ** 9]
for n in ("cyber_main", "cyber_fresh", "nba", "crypto", "ai"):
    S = SS[n]
    cells = BF.prepare(S, {}, {})
    for c in cells.values():
        es = np.sort(c["e"])[::-1]
        c["peak"] = {m: es[:m].sum() for m in GRID}
        best = defaultdict(float)
        for s, e in zip(c["posts"], c["e"]):
            a = S["info"][s]["author_id"]; best[a] = max(best[a], e)
        bs = np.sort(np.array(list(best.values())))[::-1]
        c["peakA"] = {m: bs[:m].sum() for m in GRID}
    def yld(f, days):
        tot = 0
        for d in days:
            T = [t for (t, dd) in cells if dd == d]
            sel = sorted(T, key=lambda t: (-f(cells[(t, d)]), t))[:3]
            tot += len(set().union(*[cells[(t, d)]["T"] for t in sel]))
        return tot
    dev, ev = list(S["dev"]), list(S["eval"])
    out = {}
    for key in ("peak", "peakA"):
        dv = {m: yld(lambda c, m=m: c[key][m], dev) for m in GRID}
        mstar = max(GRID, key=lambda m: (dv[m], -m))
        out[key] = (mstar if mstar < 10 ** 9 else "all", yld(lambda c: c[key][mstar], ev), {(m if m < 10**9 else 'all'): yld(lambda c, m=m: c[key][m], ev) for m in GRID})
    print(n, "edges", yld(lambda c: c["e"].sum(), ev), "authors", yld(lambda c: c["A"], ev), out, flush=True)
