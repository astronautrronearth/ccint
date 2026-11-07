import importlib.util, sys
import numpy as np
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
EMBV, TYPES = {}, {}
SS = BF.MECH.settings("--known-only" in sys.argv)
for n in [x for x in SS if not x.endswith("_terms")]:
    S = SS[n]
    cells = BF.prepare(S, EMBV, TYPES)
    ev = list(S["eval"])
    def yld(f):
        tot = 0
        for d in ev:
            T = [t for (t, dd) in cells if dd == d]
            sel = sorted(T, key=lambda t: (-f(cells[(t, d)]), t))[:3]
            tot += len(set().union(*[cells[(t, d)]["T"] for t in sel]))
        return tot
    print(n, {"edges": yld(lambda c: c["e"].sum()), "edges_cap20": yld(lambda c: np.minimum(c["e"], 20).sum()),
              "edges_cap5": yld(lambda c: np.minimum(c["e"], 5).sum()),
              "responding_posts": yld(lambda c: (c["e"] > 0).sum()), "posts": yld(lambda c: c["V"]),
              "authors": yld(lambda c: c["A"]), "oracle": yld(lambda c: len(c["T"]))}, flush=True)
