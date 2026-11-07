"""Where is the remaining gap to the oracle? Per setting: share of (oracle - edges) from the single worst day and from
the worst topic-day, development settings."""
import importlib.util, sys
import numpy as np
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
SS = BF.MECH.settings("--known-only" in sys.argv)
for n in [x for x in SS if not x.endswith("_terms")]:
    S = SS[n]; cells = BF.prepare(S, {}, {}); gaps = []
    for d in sorted(S["eval"]):
        T = [t for (t, dd) in cells if dd == d]
        U = lambda sel: len(set().union(*[cells[(t, d)]["T"] for t in sel]))
        O = sorted(T, key=lambda t: -len(cells[(t, d)]["T"]))[:3]; E = sorted(T, key=lambda t: -cells[(t, d)]["e"].sum())[:3]
        gaps.append(U(O) - U(E))
    g = np.array(gaps); tot = g.sum()
    print(n, "gap", int(tot), "worst day share", f"{g.max() / tot:.0%}" if tot else "-", "top2 days share", f"{np.sort(g)[-2:].sum() / tot:.0%}" if tot else "-",
          "days with gap<=3", int((g <= 3).sum()), "of", len(g), flush=True)
