"""Synthetic engagement manipulation (stress test for listener-side counts), paper 2.

Motivation: reply counts are the near-optimal signal on clean data but are the easiest to inflate (coordinated reply
accounts). The producer dial applied to the listener side gives a robust count: the effective number of responders
D_q = (sum_j p_j^q)^(1/(1-q)) over the accounts j replying to a unit's posts before the cutoff, p_j = j's share of
those replies (q = 2: inverse Simpson; q -> 1: exp Shannon; q = 0: distinct responders).

Attack: on each evaluation day, one unit among those whose true next-day yield is in the lower half is chosen at
random and receives R fake reply edges from B fake accounts (evenly split), created before the cutoff. Fake accounts
are never counted as genuine new responders, so the target is unchanged. Grid: B in {1, 3, 10}, R in {0, 50, 200,
1000}; 20 random draws per cell.
Rankings: edges (all replies+quotes), distinct responders, effective responders D_1 and D_2, authors, posts, oracle.

    python scripts/www_manipulation.py [--known-only] → outputs/interaction_experiment/www_eval/manipulation{_known}.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("mech", "scripts/www_dose2_mech.py")
MECH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MECH)
MON = MECH.MON
GRID_B = [1, 3, 10, 100, 300]        # 100, 300: Sybil attacks (many new accounts, few replies each)
GRID_R = [0, 50, 200, 1000]
DRAWS = 20


def cells_of(S):
    by = {}
    for s in S["seeds"]:
        r = S["info"][s]
        by.setdefault((r["topic"], r["published_at"].date()), []).append(s)
    cells = {}
    for (t, d), R in by.items():
        c = MON.cut(d)
        c2 = c + timedelta(days=1)
        before, after, resp = set(), set(), Counter()
        ne = 0
        for s in R:
            a = S["info"][s]["author_id"]
            for e in S["eb"][s]:
                ts = e["event_created_at"]
                if ts is None:
                    continue
                ext = e["source_actor_id"] != a
                if ts < c:
                    ne += 1
                    if ext:
                        before.add(e["source_actor_id"])
                        resp[e["source_actor_id"]] += 1
                elif ts < c2 and ext:
                    after.add(e["source_actor_id"])
        cells[(t, d)] = {"V": len(R), "A": len({S["info"][s]["author_id"] for s in R}), "E": ne, "resp": resp,
                         "T": after - before}
    # established responders: accounts that replied to any unit of the setting on an earlier day (before d's start)
    first_seen = {}
    for (t, d), c in sorted(cells.items(), key=lambda kv: kv[0][1]):
        for j in c["resp"]:
            first_seen.setdefault(j, d)
    for (t, d), c in cells.items():
        c["est"] = Counter({j: k for j, k in c["resp"].items() if first_seen.get(j, d) < d})
    return cells


def hill(counts, q):
    v = np.array([x for x in counts if x > 0], float)
    if not len(v):
        return 0.0
    p = v / v.sum()
    if q == 0:
        return float(len(v))
    if q == 1:
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p ** q).sum() ** (1 / (1 - q)))


def scores(c, fake):
    counts = list(c["resp"].values()) + fake
    est = list(c["est"].values())          # fake accounts are new: never established
    return {"edges": c["E"] + sum(fake), "responders": len(c["resp"]) + len(fake),
            "eff_resp_D1": hill(counts, 1), "eff_resp_D2": hill(counts, 2),
            "est_resp": float(len(est)), "est_eff_D1": hill(est, 1),
            "authors": c["A"], "posts": c["V"], "oracle": len(c["T"])}


def run(S):
    cells = cells_of(S)
    ev = sorted(S["eval"])
    pols = ["edges", "responders", "eff_resp_D1", "eff_resp_D2", "est_resp", "est_eff_D1", "authors", "posts", "oracle"]
    out = {}
    for B in GRID_B:
        for R in GRID_R:
            if R == 0 and B > 1:
                continue
            rng = np.random.default_rng([20261107, B, R])
            tot = {p: [] for p in pols}
            for _ in range(DRAWS if R else 1):
                acc = {p: 0 for p in pols}
                for d in ev:
                    T = [t for (t, dd) in cells if dd == d]
                    if not T:
                        continue
                    low = sorted(T, key=lambda t: len(cells[(t, d)]["T"]))[: max(1, len(T) // 2)]
                    victim = low[rng.integers(len(low))] if R else None
                    fake = [max(1, R // B)] * B if R else []
                    sc = {t: scores(cells[(t, d)], fake if t == victim else []) for t in T}
                    for p in pols:
                        sel = sorted(T, key=lambda t: (-sc[t][p], t))[:3]
                        acc[p] += len(set().union(*[cells[(t, d)]["T"] for t in sel]))
                for p in pols:
                    tot[p].append(acc[p])
            out[f"B{B}_R{R}"] = {p: float(np.mean(v)) for p, v in tot.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-only", action="store_true")
    a = ap.parse_args()
    SS = MECH.settings(a.known_only)
    res = {}
    for n in [x for x in SS if not x.endswith("_terms")]:
        res[n] = run(SS[n])
        print(n, {k: {p: round(v) for p, v in r.items()} for k, r in res[n].items()}, flush=True)
    tag = "_known" if a.known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/manipulation{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
