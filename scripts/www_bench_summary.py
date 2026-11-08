"""Readings of PREREG_BENCH.md (2026-11-08). Reads bench_attack{,_spread,_near}.json (scripts/www_bench_attack.py).
Sanity: clean ranker/gate = ranker.json / gate_explore.json; counts = manipulation_adaptive.json (same seeds).
Per tier and file: clean, worst_one (A1-A11), worst_all (A1-A13) as % of the oracle summed over the tier; success
rate of the strongest attack (pooled over the tier's attacked days); paired per-day bootstrap (2000 draws) of
ranker - M3 and gate - M3; H3 Kendall tau of method orderings between tiers.
    python scripts/www_bench_summary.py → outputs/interaction_experiment/www_eval/bench_summary.json
"""
import json
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau

P = "outputs/interaction_experiment/www_eval/"
G = {"dev": ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"],
     "held": ["climate", "film", "games", "soccer", "space", "stocks"],
     "test": ["music", "books", "f1", "tennis", "health", "food"]}
MS = ["edges", "posts", "authors", "conv_resp", "clipA_q90", "anchR_q90", "anchA_q90", "anchAR_q90", "ranker", "gate"]
F = {"top": "bench_attack.json", "spread": "bench_attack_spread.json", "near": "bench_attack_near.json"}


def sanity(M):
    rk = json.load(open(P + "ranker.json"))
    ga = json.load(open(P + "gate_explore.json"))
    ma = json.load(open(P + "manipulation_adaptive.json"))
    bad = []
    for n in sum(G.values(), []):
        if M[n]["clean"]["ranker"] != rk[n]["ranker"]:
            bad.append((n, "ranker", M[n]["clean"]["ranker"], rk[n]["ranker"]))
        if M[n]["clean"]["gate"] != ga[n]["gate"]:
            bad.append((n, "gate", M[n]["clean"]["gate"], ga[n]["gate"]))
        for a in ma[n]:
            for m in MS[:8]:
                if abs(M[n][a][m] - ma[n][a][m]) > 1e-6:
                    bad.append((n, a, m, M[n][a][m], ma[n][a][m]))
    return bad


def tier(M, names, rng):
    ONE = [a for a in M[names[0]] if a != "clean" and not a.startswith(("A12", "A13"))]
    ALL = [a for a in M[names[0]] if a != "clean"]
    days = {n: sorted(M[n]["clean"]["_days"]["oracle"]) for n in names}

    def pct(m, a, idx):
        num = sum(sum(M[n][a]["_days"][m][days[n][i]] for i in idx[n]) for n in names)
        den = sum(sum(M[n][a]["_days"]["oracle"][days[n][i]] for i in idx[n]) for n in names)
        return 100 * num / den

    def stats(idx):
        return {m: {"clean": pct(m, "clean", idx), "worst_one": min(pct(m, a, idx) for a in ONE),
                    "worst_all": min(pct(m, a, idx) for a in ALL)} for m in MS}

    base = {n: list(range(len(days[n]))) for n in names}
    pt = stats(base)
    w = np.array([len(days[n]) for n in names], float)

    def succ(m, attacks):
        best, arg = -1, None
        for a in attacks:
            s = [M[n][a]["_succ"][m] for n in names]
            if any(x is None for x in s):
                continue
            v = float(np.dot(s, w) / w.sum())
            if v > best:
                best, arg = v, a
        return round(100 * best, 1), arg

    res = {m: {k: round(v, 1) for k, v in pt[m].items()} for m in MS}
    for m in MS:
        res[m]["succ_one"], res[m]["succ_one_attack"] = succ(m, ONE)
        res[m]["succ_all"], res[m]["succ_all_attack"] = succ(m, ALL)
        res[m]["worst_one_attack"] = min(ONE, key=lambda a: pct(m, a, base))
    B = []
    for _ in range(2000):
        idx = {n: list(rng.integers(0, len(days[n]), len(days[n]))) for n in names}
        B.append(stats(idx))
    for x in ("ranker", "gate", "edges"):
        for y in ("anchAR_q90", "edges"):
            if x == y:
                continue
            for k in ("clean", "worst_one", "worst_all"):
                d = np.array([b[x][k] - b[y][k] for b in B])
                res[x][f"minus_{y}_{k}"] = [round(pt[x][k] - pt[y][k], 1), round(float(np.quantile(d, .025)), 1),
                                            round(float(np.quantile(d, .975)), 1)]
    return res


def main():
    out = {}
    for tag, f in F.items():
        M = json.load(open(P + f))
        bad = sanity(M) if tag != "near" else [b for b in sanity(M) if b[1] in ("ranker", "gate")]
        if tag == "near":   # counts in the near file are compared with manipulation_adaptive_near.json
            man = json.load(open(P + "manipulation_adaptive_near.json"))
            bad += [(n, a, m) for n in sum(G.values(), []) for a in man[n] for m in MS[:8]
                    if abs(M[n][a][m] - man[n][a][m]) > 1e-6]
        print(tag, "sanity mismatches:", len(bad), bad[:5], flush=True)
        rng = np.random.default_rng(20261108)
        out[tag] = {"sanity_mismatches": len(bad), **{g: tier(M, names, rng) for g, names in G.items()}}
        for g in G:
            print(" ", tag, g)
            for m in MS:
                r = out[tag][g][m]
                print("    ", m.ljust(11), r["clean"], r["worst_one"], r["worst_all"], "succ", r["succ_one"], r["succ_all"],
                      r["worst_one_attack"], flush=True)
    # H3: Kendall tau of method orderings between tiers (primary file)
    h3 = {}
    for k in ("clean", "worst_one", "worst_all"):
        for a, b in combinations(G, 2):
            x = [out["top"][a][m][k] for m in MS]
            y = [out["top"][b][m][k] for m in MS]
            t = kendalltau(x, y)
            h3[f"{k}_{a}_vs_{b}"] = [round(float(t.statistic), 2), round(float(t.pvalue), 3)]
        for g in G:
            h3[f"best_{k}_{g}"] = max(MS, key=lambda m: out["top"][g][m][k])
    out["H3"] = h3
    print(json.dumps(h3, indent=0))
    json.dump(out, open(P + "bench_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
