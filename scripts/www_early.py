"""Early-decision task (exploratory on development settings, 2026-11-07 ~03:55): the monitor decides at hour h of
day d (UTC), using posts published in [d 00:00, d+h) and edges created before d+h; the target is new external
responders in [d+h, d+h+24h) to those posts, not seen before d+h. h in {2, 6, 12, 24} (24 = the standard task).
Signals: posts, authors, edges, peak-5, habitual (mean past yield of the unit at the same h), oracle; plus the
leave-one-setting-out ridge ranker over {edges, peak5, hist_y, authors, posts} (raw target).

    python scripts/www_early.py [--known-only] → outputs/interaction_experiment/www_eval/early{_known}.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("mech", "scripts/www_dose2_mech.py")
MECH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MECH)
DEV = ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"]
HOURS = [2, 6, 12, 24]
FE = ["edges", "peak5", "hist_y", "authors", "posts"]


def build(S, h):
    by = defaultdict(list)
    for s in S["seeds"]:
        r = S["info"][s]
        d = r["published_at"].date()
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        if r["published_at"] < start + timedelta(hours=h):
            by[(r["topic"], d)].append(s)
    cells = {}
    for (t, d), R in by.items():
        c = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(hours=h)
        c2 = c + timedelta(days=1)
        before, after, ep = set(), set(), []
        for s in R:
            a = S["info"][s]["author_id"]
            k = 0
            for e in S["eb"][s]:
                ts = e["event_created_at"]
                if ts is None:
                    continue
                ext = e["source_actor_id"] != a
                if ts < c:
                    k += 1
                    if ext:
                        before.add(e["source_actor_id"])
                elif ts < c2 and ext:
                    after.add(e["source_actor_id"])
            ep.append(k)
        cnt = Counter(S["info"][s]["author_id"] for s in R)
        cells[(t, d)] = {"V": len(R), "A": len(cnt), "e": np.array(ep, float), "T": after - before}
    hist = defaultdict(dict)
    for (t, d), c in cells.items():
        hist[t][d] = c
    for (t, d), c in cells.items():
        prev = [hist[t][x] for x in hist[t] if x < d]
        c["hist_y"] = float(np.mean([len(p["T"]) for p in prev])) if prev else 0.0
        es = np.sort(c["e"])[::-1]
        c["f"] = {"edges": np.log1p(es.sum()), "peak5": np.log1p(es[:5].sum()), "hist_y": np.log1p(c["hist_y"]),
                  "authors": np.log1p(c["A"]), "posts": np.log1p(c["V"])}
    return cells


def yld(cells, days, f):
    tot = 0
    for d in days:
        keys = [k for k in cells if k[1] == d]
        sel = sorted(keys, key=lambda k: (-f(cells[k], k), k[0]))[:3]
        tot += len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
    return tot


def zs(cells, days):
    Z = {}
    for d in days:
        keys = [k for k in cells if k[1] == d]
        if not keys:
            continue
        M = np.array([[cells[k]["f"][f] for f in FE] for k in keys])
        M = (M - M.mean(0)) / (M.std(0) + 1e-6)
        Z.update(dict(zip(keys, M)))
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-only", action="store_true")
    a = ap.parse_args()
    SS = MECH.settings(a.known_only)
    names = [n for n in SS if not n.endswith("_terms")]
    res = {}
    for h in HOURS:
        C = {n: build(SS[n], h) for n in names}
        folds = [["cyber_main", "cyber_fresh"], ["nba"], ["crypto"], ["ai"]]
        W = {}
        for held in folds:
            X, y = [], []
            for n in DEV:
                if n in held:
                    continue
                Z = zs(C[n], SS[n]["dev"])
                for d in SS[n]["dev"]:
                    keys = [k for k in Z if k[1] == d]
                    if len(keys) < 2:
                        continue
                    t = np.array([len(C[n][k]["T"]) for k in keys], float)
                    X += [Z[k] for k in keys]
                    y += list(t - t.mean())
            X, y = np.array(X), np.array(y)
            w = np.linalg.solve(X.T @ X + np.eye(len(FE)), X.T @ y)
            for n in held:
                W[n] = w
        Xa, ya = [], []
        for n in DEV:
            Z = zs(C[n], SS[n]["dev"])
            for d in SS[n]["dev"]:
                keys = [k for k in Z if k[1] == d]
                if len(keys) < 2:
                    continue
                t = np.array([len(C[n][k]["T"]) for k in keys], float)
                Xa += [Z[k] for k in keys]
                ya += list(t - t.mean())
        wall = np.linalg.solve(np.array(Xa).T @ np.array(Xa) + np.eye(len(FE)), np.array(Xa).T @ np.array(ya))
        res[h] = {}
        for n in names:
            c, ev = C[n], SS[n]["eval"]
            w = W.get(n, wall)
            Z = zs(c, ev)
            r = {"posts": yld(c, ev, lambda x, k: x["V"]), "authors": yld(c, ev, lambda x, k: x["A"]),
                 "edges": yld(c, ev, lambda x, k: x["e"].sum()),
                 "peak5": yld(c, ev, lambda x, k: np.sort(x["e"])[::-1][:5].sum()),
                 "habitual": yld(c, ev, lambda x, k: x["hist_y"]),
                 "ranker": yld(c, ev, lambda x, k: float(Z[k] @ w) if k in Z else -1e9),
                 "oracle": yld(c, ev, lambda x, k: len(x["T"]))}
            res[h][n] = r
            print(h, n, r, flush=True)
    tag = "_known" if a.known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/early{tag}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
