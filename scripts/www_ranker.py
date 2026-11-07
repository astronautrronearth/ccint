"""Cross-domain producer-listener ranker (paper 2 method). Features motivated by the error analysis on the five
development settings (scripts/www_error_analysis.py): edges favour large topics with many lukewarm posts and
already-large audiences; missed topics have one hot thread and room for new responders.

Per (topic, day), features standardised within the setting-day: log1p of edges, posts, authors, prior responders,
peak-5 engagement (sum over the 5 most-engaged posts), past mean yield; top-1 share of engagement; burst z.
Model: ridge on log1p(next-day new responders), demeaned within setting-day, trained on the development days of the
training settings (pooled). Evaluation: k = 3 union yield on evaluation days.
  --loso      leave-one-setting-out over the five development settings (the two cyber windows held out together)
  default     train on all five development settings, freeze, evaluate on every setting (incl. the six held-out)

    python scripts/www_ranker.py --loso | python scripts/www_ranker.py → outputs/interaction_experiment/www_eval/ranker{_loso}.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py")
BF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
DEV_SETTINGS = ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"]
FEATS = ["edges", "peak5", "hist_y"]   # frozen candidate (PREREG_DOSE2_AMEND9); full list in www_ranker_variants.py
ALL_FEATS = ["edges", "posts", "authors", "prior_resp", "peak5", "hist_y", "top1", "burst"]
LAM = 1.0


def feats(c):
    es = np.sort(c["e"])[::-1]
    return {"edges": np.log1p(es.sum()), "posts": np.log1p(c["V"]), "authors": np.log1p(c["A"]),
            "prior_resp": np.log1p(c.get("prior_resp", 0)), "peak5": np.log1p(es[:5].sum()),
            "hist_y": np.log1p(c["hist_y"]), "top1": es[0] / max(1.0, es.sum()), "burst": c["burst"]}


def build(S):
    cells = BF.prepare(S, {}, {})
    # prior responders before the cutoff (distinct external accounts), recomputed here
    for (t, d), c in cells.items():
        cut = BF.MON.cut(d)
        pr = set()
        for s in c["posts"]:
            a = S["info"][s]["author_id"]
            for e in S["eb"][s]:
                if e["event_created_at"] is not None and e["event_created_at"] < cut and e["source_actor_id"] != a:
                    pr.add(e["source_actor_id"])
        c["prior_resp"] = len(pr)
    rows = {}
    for (t, d), c in cells.items():
        rows[(t, d)] = feats(c)
    return cells, rows


def standardise(rows, days):
    """Standardise the frozen FEATS within each day."""
    Z = {}
    for d in days:
        keys = [k for k in rows if k[1] == d]
        if not keys:
            continue
        M = np.array([[rows[k][f] for f in FEATS] for k in keys], float)
        mu, sd = M.mean(0), M.std(0) + 1e-6
        for k, v in zip(keys, (M - mu) / sd):
            Z[k] = v
    return Z


def train(data, names):
    X, y = [], []
    for n in names:
        cells, rows, S = data[n]
        Z = standardise(rows, S["dev"])
        for d in S["dev"]:
            keys = [k for k in Z if k[1] == d]
            if len(keys) < 2:
                continue
            t = np.array([len(cells[k]["T"]) for k in keys], float)   # raw target (AMEND9)
            X += [Z[k] for k in keys]
            y += list(t - t.mean())
    X, y = np.array(X), np.array(y)
    return np.linalg.solve(X.T @ X + LAM * np.eye(X.shape[1]), X.T @ y)


def evaluate(cells, rows, days, w):
    Z = standardise(rows, days)
    tot = 0
    for d in days:
        keys = [k for k in Z if k[1] == d]
        sel = sorted(keys, key=lambda k: (-(Z[k] @ w), k[0]))[:3]
        tot += len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
    return tot


def baseline(cells, days, f):
    tot = 0
    for d in days:
        keys = [k for k in cells if k[1] == d]
        sel = sorted(keys, key=lambda k: (-f(cells[k]), k[0]))[:3]
        tot += len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso", action="store_true")
    a = ap.parse_args()
    SS = BF.MECH.settings(a.loso)
    names = [n for n in SS if not n.endswith("_terms")]
    data = {n: (*build(SS[n]), SS[n]) for n in names}
    res = {}
    if a.loso:
        folds = [["cyber_main", "cyber_fresh"], ["nba"], ["crypto"], ["ai"]]
        for held in folds:
            w = train(data, [n for n in DEV_SETTINGS if n not in held])
            for n in held:
                cells, rows, S = data[n]
                res[n] = {"ranker": evaluate(cells, rows, S["eval"], w), "weights": dict(zip(FEATS, map(float, w)))}
    else:
        w = train(data, DEV_SETTINGS)
        for n in names:
            cells, rows, S = data[n]
            res[n] = {"ranker": evaluate(cells, rows, S["eval"], w)}
        res["_weights"] = dict(zip(FEATS, map(float, w)))
    for n in names:
        if n not in res:
            continue
        cells, rows, S = data[n]
        ev = S["eval"]
        res[n].update({"edges": baseline(cells, ev, lambda c: c["e"].sum()), "posts": baseline(cells, ev, lambda c: c["V"]),
                       "authors": baseline(cells, ev, lambda c: c["A"]), "oracle": baseline(cells, ev, lambda c: len(c["T"]))})
        print(n, {k: v for k, v in res[n].items() if k != "weights"}, flush=True)
    tag = "_loso" if a.loso else ""
    Path(f"outputs/interaction_experiment/www_eval/ranker{tag}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
