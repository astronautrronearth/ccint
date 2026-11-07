"""EXPLORATORY (2026-11-07 ~08:00): day-level signal gating. The best single count changes from day to day
(per-day best-of-signals reaches ~0.97 of the oracle vs ~0.88 for edges). A gate predicts, from today's observable
cross-topic statistics, the relative yield of each signal; trained on development days of the five development
settings, applied frozen to evaluation days of all settings (dev settings: leave-one-setting-out).
Day features: engagement concentration (top topic share of edges; top post share), Spearman agreement between
signals, number of topics, total posts/edges (log), share of today's edges from posts of the top-3 authors.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py")
BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF); BF.ECAP = 10**9
SIG = {"posts": lambda c: c["V"], "authors": lambda c: c["A"], "edges": lambda c: c["E"],
       "habitual": lambda c: c["hist_y"], "peak5": lambda c: np.sort(c["e"])[::-1][:5].sum()}
NS = list(SIG)
DEV = ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"]
FOLDS = [["cyber_main", "cyber_fresh"], ["nba"], ["crypto"], ["ai"]]

def dy(cells, keys, f):
    sel = sorted(keys, key=lambda k: (-f(cells[k]), k[0]))[:3]
    return len(set().union(*[cells[k]["T"] for k in sel]))

def dayfeat(cells, keys):
    M = np.array([[f(cells[k]) for f in SIG.values()] for k in keys], float)
    E = M[:, 2]; allE = np.concatenate([cells[k]["e"] for k in keys])
    ag = []
    for i in range(len(NS)):
        for j in range(i + 1, len(NS)):
            r = spearmanr(M[:, i], M[:, j]).correlation
            ag.append(0.0 if np.isnan(r) else r)
    return np.array([E.max() / max(1, E.sum()), allE.max() / max(1, allE.sum()), np.log(len(keys)),
                     np.log1p(M[:, 0].sum()), np.log1p(E.sum()), *ag, 1.0])

SS = BF.MECH.settings(False)
names = [x for x in SS if not x.endswith("_terms")]
D = {}
for n in names:
    S = SS[n]; cells = BF.prepare(S, {}, {})
    rows = {}
    for d in sorted(set(S["dev"]) | set(S["eval"])):
        keys = [k for k in cells if k[1] == d]
        if len(keys) < 4: continue
        y = np.array([dy(cells, keys, f) for f in SIG.values()], float)
        o = dy(cells, keys, lambda c: len(c["T"]))
        rows[d] = (dayfeat(cells, keys), y, o)
    D[n] = (rows, S)
    print("built", n, flush=True)

def fit(train):
    X, Y = [], []
    for n in train:
        rows, S = D[n]
        for d in S["dev"]:
            if d in rows:
                x, y, o = rows[d]
                X.append(x); Y.append(y / max(1, y.max()))
    X, Y = np.array(X), np.array(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6; sd[-1] = 1; mu[-1] = 0
    Xs = (X - mu) / sd
    W = np.linalg.solve(Xs.T @ Xs + 5 * np.eye(X.shape[1]), Xs.T @ Y)
    return lambda x: ((x - mu) / sd) @ W

res = {}
for n in names:
    held = next((f for f in FOLDS if n in f), None)
    g = fit([m for m in DEV if not held or m not in held])
    rows, S = D[n]
    out = {s: 0.0 for s in NS + ["gate", "best_per_day", "oracle"]}
    for d in S["eval"]:
        if d not in rows: continue
        x, y, o = rows[d]
        for i, s in enumerate(NS): out[s] += y[i]
        out["gate"] += y[int(np.argmax(g(x)))]
        out["best_per_day"] += y.max(); out["oracle"] += o
    res[n] = out
    print(n, {k: round(v / out["oracle"], 3) for k, v in out.items()}, flush=True)
H = ["climate", "film", "games", "soccer", "space", "stocks"]
for grp, L in [("dev", DEV), ("held", H)]:
    print(grp, {k: round(np.mean([res[n][k] / res[n]["oracle"] for n in L]), 3) for k in res[L[0]]})
Path("outputs/interaction_experiment/www_eval/gate_explore.json").write_text(json.dumps(res, indent=1))
