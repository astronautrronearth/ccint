import importlib.util, sys
import numpy as np
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("rk", "scripts/www_ranker.py"); RK = importlib.util.module_from_spec(spec); spec.loader.exec_module(RK)
SS = RK.BF.MECH.settings(True)
data = {n: (*RK.build(SS[n]), SS[n]) for n in RK.DEV_SETTINGS}
ALL = list(RK.FEATS)
folds = [["cyber_main", "cyber_fresh"], ["nba"], ["crypto"], ["ai"]]
def run(feats, target, lam):
    idx = [ALL.index(f) for f in feats]
    def train(names):
        X, y = [], []
        for n in names:
            cells, rows, S = data[n]
            Z = RK.standardise(rows, S["dev"])
            for d in S["dev"]:
                keys = [k for k in Z if k[1] == d]
                if len(keys) < 2: continue
                t = np.array([len(cells[k]["T"]) for k in keys], float)
                t = np.log1p(t) if target == "log" else (t if target == "raw" else np.sqrt(t))
                X += [Z[k][idx] for k in keys]; y += list(t - t.mean())
        X, y = np.array(X), np.array(y)
        return np.linalg.solve(X.T @ X + lam * np.eye(len(idx)), X.T @ y)
    tot_r, tot_e = 0, 0
    per = {}
    for held in folds:
        w = train([n for n in RK.DEV_SETTINGS if n not in held])
        for n in held:
            cells, rows, S = data[n]
            Z = RK.standardise(rows, S["eval"])
            r = 0
            for d in S["eval"]:
                keys = [k for k in Z if k[1] == d]
                sel = sorted(keys, key=lambda k: (-(Z[k][idx] @ w), k[0]))[:3]
                r += len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
            e = RK.baseline(cells, S["eval"], lambda c: c["e"].sum())
            per[n] = (r, e); tot_r += r; tot_e += e
    return tot_r, tot_e, per
for feats in (ALL, ["edges", "peak5", "prior_resp", "posts"], ["edges", "peak5", "top1", "posts"], ["edges", "peak5"], ["edges", "peak5", "hist_y"]):
    for target in ("log", "sqrt", "raw"):
        for lam in (1.0, 100.0):
            r, e, per = run(feats, target, lam)
            print("+".join(feats), target, lam, "ranker", r, "edges", e, per, flush=True)
