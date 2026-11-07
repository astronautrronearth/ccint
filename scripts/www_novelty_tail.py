"""Exploratory features on the development settings (2026-11-07 ~03:45): semantic novelty of a unit's day against
its own past (bge-small embeddings), and the shape of the within-unit engagement distribution. Evaluated by
leave-one-setting-out ranker (features standardised within day; ridge on raw target) and as single-signal rankers.
    python scripts/www_novelty_tail.py → outputs/interaction_experiment/www_eval/novelty_tail_known.json
"""
import importlib.util, json, sys
import numpy as np
from datetime import timedelta
from pathlib import Path
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
EMBV, _ = BF.load_side()
SS = BF.MECH.settings(True)
DEV = ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"]
FE = ["edges", "peak5", "hist_y", "novelty", "novelty_max", "gini", "hot_share", "sem_spread"]


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0: return 0.0
    n = len(x); return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


data = {}
for n in DEV:
    S = SS[n]; cells = BF.prepare(S, {}, {})
    hist = {}
    for (t, d), c in cells.items():
        V = np.array([EMBV.get(BF.pid(s), np.zeros(384, np.float32)) for s in c["posts"]], np.float32)
        c["emb"] = V; c["cent"] = V.mean(0) / (np.linalg.norm(V.mean(0)) + 1e-9)
        hist.setdefault(t, {})[d] = c
    rows = {}
    for (t, d), c in cells.items():
        prev = [hist[t][x] for x in hist[t] if x < d]
        es = np.sort(c["e"])[::-1]
        if prev:
            pc = np.mean([p["cent"] for p in prev[-3:]], 0); pc /= np.linalg.norm(pc) + 1e-9
            sims = c["emb"] @ pc
            nov, novmax = float(1 - c["cent"] @ pc), float(1 - np.percentile(sims, 10))
        else:
            nov, novmax = 0.0, 0.0
        spread = float(1 - np.linalg.norm(c["emb"].mean(0)))
        rows[(t, d)] = {"edges": np.log1p(es.sum()), "peak5": np.log1p(es[:5].sum()), "hist_y": np.log1p(c["hist_y"]),
                        "novelty": nov, "novelty_max": novmax, "gini": gini(c["e"]),
                        "hot_share": float((c["e"] >= 5).mean()), "sem_spread": spread}
    data[n] = (cells, rows, S)


def Z(rows, days, feats):
    out = {}
    for d in days:
        keys = [k for k in rows if k[1] == d]
        if not keys: continue
        M = np.array([[rows[k][f] for f in feats] for k in keys]); M = (M - M.mean(0)) / (M.std(0) + 1e-6)
        out.update(dict(zip(keys, M)))
    return out


def ev_single(f):
    tot = 0
    for n in DEV:
        cells, rows, S = data[n]
        for d in S["eval"]:
            keys = [k for k in rows if k[1] == d]
            sel = sorted(keys, key=lambda k: (-rows[k][f], k[0]))[:3]
            tot += len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
    return tot


def loso(feats, lam=1.0):
    folds = [["cyber_main", "cyber_fresh"], ["nba"], ["crypto"], ["ai"]]
    per, tot = {}, 0
    for held in folds:
        X, y = [], []
        for n in DEV:
            if n in held: continue
            cells, rows, S = data[n]; z = Z(rows, S["dev"], feats)
            for d in S["dev"]:
                keys = [k for k in z if k[1] == d]
                if len(keys) < 2: continue
                t = np.array([len(cells[k]["T"]) for k in keys], float)
                X += [z[k] for k in keys]; y += list(t - t.mean())
        X, y = np.array(X), np.array(y); w = np.linalg.solve(X.T @ X + lam * np.eye(len(feats)), X.T @ y)
        for n in held:
            cells, rows, S = data[n]; z = Z(rows, S["eval"], feats); r = 0
            for d in S["eval"]:
                keys = [k for k in z if k[1] == d]
                sel = sorted(keys, key=lambda k: (-(z[k] @ w), k[0]))[:3]
                r += len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
            per[n] = r; tot += r
    return tot, per, dict(zip(feats, map(float, w)))


res = {"single": {f: ev_single(f) for f in FE}}
print("single", res["single"], flush=True)
for feats in (["edges"], ["edges", "novelty"], ["edges", "novelty_max"], ["edges", "gini"], ["edges", "hot_share"],
              ["edges", "sem_spread"], ["edges", "peak5", "hist_y"], ["edges", "peak5", "hist_y", "novelty"],
              ["edges", "peak5", "hist_y", "novelty", "gini", "hot_share", "sem_spread"]):
    t, per, w = loso(feats)
    res["+".join(feats)] = {"total": t, "per": per, "w": w}
    print("+".join(feats), t, per, flush=True)
Path("outputs/interaction_experiment/www_eval/novelty_tail_known.json").write_text(json.dumps(res, indent=1))
