"""Breakout prediction (exploratory, development settings): a unit-day is a breakout if next-day new responders >= 3x
the unit's past mean and >= 10. AUC of single signals and of a leave-one-setting-out logistic model over within-day
standardised features (edges, peak5, top1 share, authors, posts, burst, author growth vs yesterday, novelty)."""
import importlib.util, json, sys
import numpy as np
from datetime import timedelta
from pathlib import Path
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
from scipy.stats import rankdata


def roc_auc_score(y, s):
    y = np.asarray(y); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class LogisticRegression:
    def __init__(self, max_iter=1000, C=0.5):
        self.it, self.l2 = max_iter, 1.0 / C
    def fit(self, X, Y):
        w = np.zeros(X.shape[1] + 1); Xb = np.hstack([X, np.ones((len(X), 1))])
        for _ in range(self.it):
            p = 1 / (1 + np.exp(-Xb @ w)); g = Xb.T @ (p - Y) / len(Y) + self.l2 * np.r_[w[:-1], 0] / len(Y)
            w -= 0.5 * g
        self.w = w; return self
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.hstack([X, np.ones((len(X), 1))]) @ self.w)); return np.c_[1 - p, p]
EMBV, _ = BF.load_side()
KNOWN = "--known-only" in sys.argv
SS = BF.MECH.settings(KNOWN)
DEV = ["cyber_main", "cyber_fresh", "nba", "crypto", "ai"]
ALLN = [x for x in SS if not x.endswith("_terms")]
F = ["edges", "peak5", "top1", "authors", "posts", "burst", "auth_growth", "novelty"]
data = {}
for n in ALLN:
    S = SS[n]; cells = BF.prepare(S, {}, {}); hist = {}
    for (t, d), c in cells.items():
        V = np.array([EMBV.get(BF.pid(s), np.zeros(384, np.float32)) for s in c["posts"]]); m = V.mean(0)
        c["cent"] = m / (np.linalg.norm(m) + 1e-9); hist.setdefault(t, {})[d] = c
    rows = []
    for (t, d), c in cells.items():
        if d not in set(S["dev"]) | set(S["eval"]): continue
        prev = [hist[t][x] for x in sorted(hist[t]) if x < d]
        if not prev: continue
        y = len(c["T"]); base = np.mean([len(p["T"]) for p in prev])
        es = np.sort(c["e"])[::-1]; yd = hist[t].get(d - timedelta(days=1))
        pc = np.mean([p["cent"] for p in prev[-3:]], 0); pc /= np.linalg.norm(pc) + 1e-9
        rows.append({"d": d, "split": "eval" if d in set(S["eval"]) else "dev", "y": int(y >= 3 * base and y >= 10),
                     "edges": np.log1p(es.sum()), "peak5": np.log1p(es[:5].sum()), "top1": es[0] / max(1, es.sum()),
                     "authors": np.log1p(c["A"]), "posts": np.log1p(c["V"]), "burst": c["burst"],
                     "auth_growth": np.log1p(c["A"]) - np.log1p(yd["A"] if yd else 0), "novelty": float(1 - c["cent"] @ pc)})
    # within-day standardisation
    for d in {r["d"] for r in rows}:
        R = [r for r in rows if r["d"] == d]
        for f in F:
            v = np.array([r[f] for r in R]); mu, sd = v.mean(), v.std() + 1e-6
            for r in R: r["z_" + f] = (r[f] - mu) / sd
    data[n] = rows
res = {}
for n in ALLN:
    R = [r for r in data[n] if r["split"] == "eval"]; ys = [r["y"] for r in R]
    res[n] = {"n": len(R), "breakouts": int(sum(ys))}
    if 0 < sum(ys) < len(ys):
        res[n]["auc"] = {f: round(float(roc_auc_score(ys, [r[f] for r in R])), 3) for f in F}
        tr = [r for m in DEV if m != n and not (n.startswith("cyber") and m.startswith("cyber")) for r in data[m]]
        X = np.array([[r["z_" + f] for f in F] for r in tr]); Y = np.array([r["y"] for r in tr])
        if 0 < Y.sum() < len(Y):
            clf = LogisticRegression(max_iter=1000, C=0.5).fit(X, Y)
            p = clf.predict_proba(np.array([[r["z_" + f] for f in F] for r in R]))[:, 1]
            res[n]["auc_model"] = round(float(roc_auc_score(ys, p)), 3)
    print(n, res[n], flush=True)
Path("outputs/interaction_experiment/www_eval/breakout" + ("_known" if KNOWN else "") + ".json").write_text(json.dumps(res, indent=1, default=str))
