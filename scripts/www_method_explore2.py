"""EXPLORATORY (2026-11-07 ~07:30): self-calibrating selection. Each day, using only past days of the same domain
(outcomes of day d-1 are known at day d's cutoff), (i) follow-the-leader among signals (all past / last 7 days),
(ii) a ridge ranker refit daily on the domain's own past days. Signals: posts, authors, edges, habitual, peak5.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py")
BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF); BF.ECAP = 10**9
SIG = {"posts": lambda c: c["V"], "authors": lambda c: c["A"], "edges": lambda c: c["E"],
       "habitual": lambda c: c["hist_y"], "peak5": lambda c: np.sort(c["e"])[::-1][:5].sum()}
def feat(c): return np.log1p([c["V"], c["A"], c["E"], c["hist_y"], np.sort(c["e"])[::-1][:5].sum()])
def dy(cells, keys, sc):
    sel = sorted(keys, key=lambda k: (-sc[k], k[0]))[:3]
    return len(set().union(*[cells[k]["T"] for k in sel])) if sel else 0
SS = BF.MECH.settings(False)
res = {}
for n in [x for x in SS if not x.endswith("_terms")]:
    S = SS[n]; cells = BF.prepare(S, {}, {})
    days = sorted(set(S["hist"]) | set(S["dev"]) | set(S["eval"])); ev = set(S["eval"])
    past = {s: [] for s in SIG}; XY = []
    out = {k: 0 for k in ["oracle", "ftl_all", "ftl_7", "ridge_self", "rankavg_all"] + list(SIG)}
    for d in days:
        keys = [k for k in cells if k[1] == d]
        if not keys: continue
        sc = {s: {k: f(cells[k]) for k in keys} for s, f in SIG.items()}
        ys = {s: dy(cells, keys, sc[s]) for s in SIG}
        mx = max(ys.values()) or 1
        if d in ev:
            for s in SIG: out[s] += ys[s]
            out["oracle"] += dy(cells, keys, {k: len(cells[k]["T"]) for k in keys})
            for tag, W in [("ftl_all", None), ("ftl_7", 7)]:
                best = max(SIG, key=lambda s: (sum(past[s][-W:] if W else past[s]), s == "edges"))
                out[tag] += ys[best]
            if len(XY) > 30:
                X = np.array([x for x, _ in XY]); y = np.array([y for _, y in XY])
                w = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ y)
                Z = np.array([feat(cells[k]) for k in keys]); Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
                out["ridge_self"] += dy(cells, keys, dict(zip(keys, Z @ w)))
            else:
                out["ridge_self"] += ys["edges"]
            rk = sum(rankdata([sc[s][k] for k in keys]) for s in SIG)
            out["rankavg_all"] += dy(cells, keys, dict(zip(keys, rk)))
        for s in SIG: past[s].append(ys[s] / mx)
        if len(keys) >= 2:
            Z = np.array([feat(cells[k]) for k in keys]); Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
            t = np.array([len(cells[k]["T"]) for k in keys], float)
            XY += list(zip(Z, t - t.mean()))
    res[n] = out; o = out["oracle"]
    print(n, {k: round(v / o, 3) for k, v in out.items()}, flush=True)
Path("outputs/interaction_experiment/www_eval/method_explore2.json").write_text(json.dumps(res, indent=1))
