"""Paired per-day bootstrap CIs (exploratory, 2026-11-07, after RESULTS_DOSE3) and pooled cost-to-steal.
Reads manipulation_adaptive.json (per-day yields; attacks A1-A13) and cost.json.
Per tier: % of oracle (summed over settings), clean and worst over the one-sided families {A2, A9a, A10, A11}
= A1-A11, and worst over all incl. A12/A13.
Bootstrap: resample evaluation days with replacement within each setting, 2000 draws, same days for all counts.
    python scripts/www_story_ci.py → outputs/interaction_experiment/www_eval/story_ci.json
"""
import json
import numpy as np
M = json.load(open("outputs/interaction_experiment/www_eval/manipulation_adaptive.json"))
C = json.load(open("outputs/interaction_experiment/www_eval/cost.json"))
G = {"dev": ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"], "held": ["climate", "film", "games", "soccer", "space", "stocks"],
     "test": ["music", "books", "f1", "tennis", "health", "food"]}
MS = ["edges", "posts", "authors", "conv_resp", "clipA_q90", "AxR_b75", "anchA_q90", "anchR_q90", "anchAR_q90"]
ONE = [a for a in M["music"] if a != "clean" and not a.startswith(("A12", "A13"))]   # A1-A11 (one- or two-sided families)
ALL = [a for a in M["music"] if a != "clean"]
rng = np.random.default_rng(20261107)
out = {}
for g, names in G.items():
    days = {n: sorted(M[n]["clean"]["_days"]["oracle"]) for n in names}
    def pct(m, a, idx):
        num = sum(sum(M[n][a]["_days"][m][days[n][i]] for i in idx[n]) for n in names)
        den = sum(sum(M[n][a]["_days"]["oracle"][days[n][i]] for i in idx[n]) for n in names)
        return 100 * num / den
    base = {n: list(range(len(days[n]))) for n in names}
    def stats(idx):
        r = {}
        for m in MS:
            r[m] = {"clean": pct(m, "clean", idx), "worst_one": min(pct(m, a, idx) for a in ONE),
                    "worst_all": min(pct(m, a, idx) for a in ALL)}
        return r
    pt = stats(base)
    B = []
    for _ in range(2000):
        idx = {n: list(rng.integers(0, len(days[n]), len(days[n]))) for n in names}
        B.append(stats(idx))
    res = {}
    for m in MS:
        res[m] = {k: round(pt[m][k], 1) for k in pt[m]}
        if m != "anchAR_q90":
            for k in ("clean", "worst_one", "worst_all"):
                dd = np.array([b["anchAR_q90"][k] - b[m][k] for b in B])
                res[m][f"M3_minus_{k}"] = [round(pt["anchAR_q90"][k] - pt[m][k], 1), round(float(np.quantile(dd, .025)), 1), round(float(np.quantile(dd, .975)), 1)]
    cost = {}
    for m in ["edges", "posts", "authors", "anchA", "anchR", "anchAR"]:
        x = np.concatenate([C[n]["_raw"][m] for n in names]); e = np.concatenate([C[n]["_raw"]["edges"] for n in names])
        cl = np.concatenate([C[n]["_clone"][m] for n in names])
        cost[m] = {"median": float(np.median(x)), "sum_ratio_to_edges": float(x.sum() / e.sum()), "sum_ratio_to_clone": float(x.sum() / cl.sum())}
    out[g] = {"yield": res, "cost": cost}
    print(g)
    for m in MS:
        print("  ", m.ljust(11), res[m])
    for m, r in cost.items():
        print("   cost", m.ljust(7), {k: round(v, 2) for k, v in r.items()})
json.dump(out, open("outputs/interaction_experiment/www_eval/story_ci.json", "w"), indent=1)
