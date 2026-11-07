import importlib.util, sys
import numpy as np
from collections import defaultdict
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
SS = BF.MECH.settings(True)
for n in ("cyber_main", "cyber_fresh", "nba", "crypto", "ai"):
    S = SS[n]
    cells = BF.prepare(S, {}, {})
    for c in cells.values():
        best = defaultdict(float)
        for s, e in zip(c["posts"], c["e"]):
            a = S["info"][s]["author_id"]
            best[a] = max(best[a], e)
        c["LWP"] = sum(1 + v for v in best.values())
        c["LWP_log"] = sum(1 + np.log1p(v) for v in best.values())
        c["auth_resp"] = sum(v > 0 for v in best.values())
        es = np.sort(c["e"])[::-1]
        c["top5"] = es[:5].sum()
    ev = list(S["eval"])
    def yld(f):
        tot = 0
        for d in ev:
            T = [t for (t, dd) in cells if dd == d]
            sel = sorted(T, key=lambda t: (-f(cells[(t, d)]), t))[:3]
            tot += len(set().union(*[cells[(t, d)]["T"] for t in sel]))
        return tot
    print(n, {"posts": yld(lambda c: c["V"]), "authors": yld(lambda c: c["A"]), "edges": yld(lambda c: c["e"].sum()),
              "LWP": yld(lambda c: c["LWP"]), "LWP_log": yld(lambda c: c["LWP_log"]), "authors_responded": yld(lambda c: c["auth_resp"]),
              "top5_posts": yld(lambda c: c["top5"]), "oracle": yld(lambda c: len(c["T"]))}, flush=True)
