"""Audit + baselines (2026-11-07, after the user's check): (1) score the discussion target as the UNION of new
external responders over the k selected topics (search-term units overlap, so summing per-topic counts double
counts); (2) add standard baselines: random topics, burst z-score of today's posts against the topic's previous
days, yesterday's new responders (needs fetching), a ridge model on free features fitted on development days, and
the same model with engagement features.

    python scripts/www_baselines_union.py [--known-only] → outputs/interaction_experiment/www_eval/baselines_union{_known}.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MECH = _mod("mech", "scripts/www_dose2_mech.py")
MON = MECH.MON


def sets_and_features(S):
    """Per (unit, day): target responder set (new external responders in the 24 h after the cutoff, not seen before
    the cutoff on that unit's posts) and the scores of every policy."""
    by = defaultdict(list)
    for s in S["seeds"]:
        r = S["info"][s]
        by[(r["topic"], r["published_at"].date())].append(s)
    cells = {}
    for (t, d), R in by.items():
        c = MON.cut(d)
        c2 = c + timedelta(days=1)
        before, after, ne = set(), set(), 0
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
                elif ts < c2 and ext:
                    after.add(e["source_actor_id"])
        cnt = Counter(S["info"][s]["author_id"] for s in R)
        cells[(t, d)] = {"posts": len(R), "authors": len(cnt), "edges": ne, "responders": len(before),
                         "T": after - before}
    return cells


def evaluate(cells, days, scorer, k=3, reps=20, seed=0):
    rng = np.random.default_rng([20261107, seed])
    tot_u, tot_s, per = 0.0, 0.0, []
    for d in days:
        T = [t for (t, dd) in cells if dd == d]
        if not T:
            per.append(0.0)
            continue
        sc = np.array([scorer(t, d) for t in T], float)
        vu, vs = [], []
        for _ in range(reps):
            idx = np.argsort(-(sc + rng.random(len(T)) * 1e-6))[:k]
            sel = [T[i] for i in idx]
            vu.append(len(set().union(*[cells[(t, d)]["T"] for t in sel])))
            vs.append(sum(len(cells[(t, d)]["T"]) for t in sel))
        tot_u += np.mean(vu)
        tot_s += np.mean(vs)
        per.append(float(np.mean(vu)))
    return tot_u, tot_s, per


def run(name, S):
    cells = sets_and_features(S)
    days_all = sorted({d for _, d in cells})
    hist = defaultdict(dict)
    for (t, d), c in cells.items():
        hist[t][d] = c
    dev, ev = list(S["dev"]), list(S["eval"])

    def prev_days(t, d):
        return [hist[t][x] for x in hist[t] if x < d]

    def burst(t, d):
        p = [c["posts"] for c in prev_days(t, d)]
        x = cells[(t, d)]["posts"]
        if len(p) < 2:
            return 0.0
        return (x - np.mean(p)) / (np.std(p) + 1.0)

    def yday(t, d):
        c = hist[t].get(d - timedelta(days=1))
        return len(c["T"]) if c else 0.0

    def feats(t, d, eng):
        c = cells[(t, d)]
        pd_ = prev_days(t, d)
        hist_y = np.mean([len(x["T"]) for x in pd_]) if pd_ else 0.0
        f = [np.log1p(c["posts"]), np.log1p(c["authors"]), burst(t, d), np.log1p(hist_y)]
        if eng:
            f += [np.log1p(c["edges"]), np.log1p(c["responders"]), np.log1p(yday(t, d))]
        return f

    def ridge(eng):
        X, y = [], []
        for (t, d), c in cells.items():
            if d in set(dev):
                X.append(feats(t, d, eng))
                y.append(np.log1p(len(c["T"])))
        X, y = np.array(X), np.array(y)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        w = np.linalg.solve(Xs.T @ Xs + 1.0 * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
        return lambda t, d: float(((np.array(feats(t, d, eng)) - mu) / sd) @ w)
    pols = {"random": lambda t, d: 0.0, "posts": lambda t, d: cells[(t, d)]["posts"],
            "authors": lambda t, d: cells[(t, d)]["authors"], "burst_z": burst,
            "edges": lambda t, d: cells[(t, d)]["edges"], "responders": lambda t, d: cells[(t, d)]["responders"],
            "yesterday_yield": yday, "free_model": ridge(False), "free_plus_engagement": ridge(True),
            "oracle": lambda t, d: len(cells[(t, d)]["T"])}
    out = {}
    for i, (p, f) in enumerate(pols.items()):
        u, s_, per = evaluate(cells, ev, f, seed=i)
        out[p] = {"union": round(u, 1), "sum": round(s_, 1), "per_day": per}
    base = out["posts"]["union"]
    for p in out:
        out[p]["vs_posts_union"] = round(out[p]["union"] / base - 1, 4) if base else None
    rb = np.random.default_rng(20261107)
    idx = [rb.integers(0, len(ev), len(ev)) for _ in range(2000)]
    for p in ("edges", "free_model", "free_plus_engagement", "authors", "burst_z"):
        a, b = np.array(out[p]["per_day"]), np.array(out["posts"]["per_day"])
        out[p]["vs_posts_ci"] = [float(np.percentile([a[i].sum() / max(1e-9, b[i].sum()) - 1 for i in idx], q)) for q in (2.5, 97.5)]
    fm = np.array(out["free_model"]["per_day"])
    for p in ("edges", "free_plus_engagement"):
        a = np.array(out[p]["per_day"])
        out[p]["vs_free_model"] = round(a.sum() / max(1e-9, fm.sum()) - 1, 4)
        out[p]["vs_free_model_ci"] = [float(np.percentile([a[i].sum() / max(1e-9, fm[i].sum()) - 1 for i in idx], q)) for q in (2.5, 97.5)]
    print(name, {p: (v["union"], v["sum"], v.get("vs_posts_union")) for p, v in out.items()}, flush=True)
    return out


def main(known_only):
    S = MECH.settings(known_only)
    res = {n: run(n, s) for n, s in S.items()}
    tag = "_known" if known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/baselines_union{tag}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main("--known-only" in sys.argv)
