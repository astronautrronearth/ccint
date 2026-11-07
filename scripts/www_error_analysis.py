"""Error analysis on the five DEVELOPMENT settings (known; not the six held-out domains): which topics does ranking
by engagement miss relative to the oracle, and what free / count features distinguish them.
    python scripts/www_error_analysis.py → outputs/interaction_experiment/www_eval/error_analysis.json
"""
import importlib.util, json, sys
import numpy as np
from collections import defaultdict
from pathlib import Path
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
SS = BF.MECH.settings(True)
out = {}
for n in ("cyber_main", "cyber_fresh", "nba", "crypto", "ai"):
    S = SS[n]
    cells = BF.prepare(S, {}, {})
    ev = list(S["eval"])
    miss, wrong, lost = defaultdict(list), defaultdict(list), 0
    for d in ev:
        T = [t for (t, dd) in cells if dd == d]
        E = sorted(T, key=lambda t: (-cells[(t, d)]["e"].sum(), t))[:3]
        O = sorted(T, key=lambda t: (-len(cells[(t, d)]["T"]), t))[:3]
        for grp, U in (("missed_by_edges", set(O) - set(E)), ("wrongly_picked", set(E) - set(O))):
            for t in U:
                c = cells[(t, d)]
                es = np.sort(c["e"])[::-1]
                D = miss if grp == "missed_by_edges" else wrong
                D["yield"].append(len(c["T"]))
                D["edges"].append(c["e"].sum())
                D["top1_share"].append(es[0] / max(1, es.sum()))
                D["posts"].append(c["V"])
                D["authors"].append(c["A"])
                D["hist_yield"].append(c["hist_y"])
                D["responding_share"].append(float((c["e"] > 0).mean()))
                D["burst"].append(c["burst"])
        lost += len(set().union(*[cells[(t, d)]["T"] for t in O])) - len(set().union(*[cells[(t, d)]["T"] for t in E]))
    out[n] = {"lost_vs_oracle": lost, **{g: {k: float(np.median(v)) for k, v in D.items()} | {"n": len(D["yield"])}
                                         for g, D in (("missed_by_edges", miss), ("wrongly_picked", wrong))}}
    print(n, json.dumps(out[n]), flush=True)
Path("outputs/interaction_experiment/www_eval/error_analysis.json").write_text(json.dumps(out, indent=1))
