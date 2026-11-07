"""PREREG_DOSE2 amendment 4, E1-E5 (E6 joins content types when available).

    python scripts/www_dose2_theory.py [--known-only] → outputs/interaction_experiment/www_eval/dose2_theory{_known}.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path[:0] = ["src", "scripts"]

GRID = [round(x, 2) for x in np.linspace(0, 1, 11)]
BGRID = [round(float(x), 3) for x in np.logspace(np.log10(0.05), np.log10(2), 12)]
EMB = Path("outputs/interaction_experiment/embed")
PT = Path("outputs/interaction_experiment/ptype")
MANIF = {"cyber_main": "www_scale", "cyber_fresh": "www_contrast", "nba": "www_contrast", "crypto": "www_contrast2",
         "ai": "www_contrast2"}


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_emb():
    E = {}
    for f in EMB.glob("*.npz"):
        z = np.load(f)
        for i, v in zip(z["ids"], z["emb"]):
            E[int(i)] = v.astype(np.float32)
    return E


def load_types():
    T = {}
    for f in PT.glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            T[r["post_id"]] = r["type"]
    return T


def pid(s):
    return s[0] if isinstance(s, tuple) else s


def clusters(vecs, thr=0.9):
    n = len(vecs)
    if n <= 1:
        return n, 1.0 if n else 0.0
    X = np.stack(vecs)
    Z = X @ X.T
    par = list(range(n))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    ii, jj = np.where(np.triu(Z >= thr, 1))
    for a, b in zip(ii, jj):
        par[f(a)] = f(b)
    M = len({f(a) for a in range(n)})
    lc = 1.0 / float(np.clip(Z, 0, None).mean())       # Leinster-Cobbold order 2, uniform weights over posts
    return M, lc


def main(known_only):
    MECH = _mod("mech", "scripts/www_dose2_mech.py")
    MON = MECH.MON
    S = MECH.settings(known_only)
    EMBV, TYPES = load_emb(), load_types()
    extra = [f"H{g}" for g in GRID] + [f"B{b}" for b in BGRID] + ["SEM_M", "SEM_AM", "SEM_LC"]
    MON.POLICIES[:] = [p for p in MON.POLICIES if p[:1] not in ("H", "B", "S") or p in ("P1_posts",)] + extra
    R, rows_e3 = {}, []
    for name, s in S.items():
        cells, by = MON.build(s["seeds"], s["info"], s["eb"], s["th"], s["feeds"])
        MON.add_persistent(cells, s["dev"])
        topics = sorted({t for t, _ in cells})
        conv = {}
        for key, L in by.items():
            cnt = Counter(s["info"][x]["author_id"] for x in L)
            c = cells[key]
            for g in GRID:
                c[f"H{g}"] = float(sum(v ** g for v in cnt.values())) if g > 0 else float(len(cnt))
            for b in BGRID:
                c[f"B{b}"] = float(sum(1 - np.exp(-b * v) for v in cnt.values()))
            vecs = [EMBV[pid(x)] for x in L if pid(x) in EMBV]
            M, lc = clusters(vecs)
            c["SEM_M"], c["SEM_LC"] = float(M), float(lc)
            c["SEM_AM"] = float(len(cnt)) * (M / max(1, len(vecs)))
            ty = [TYPES.get(pid(x)) for x in L]
            ty = [t for t in ty if t]
            conv[key] = (sum(t in ("question", "opinion") for t in ty) / len(ty)) if ty else None

        def tot(pol, days):
            return sum(MON.select_value(cells, topics, d, pol, 3, 20) for d in days)
        dev = {p: tot(p, s["dev"]) for p in extra[:len(GRID) + len(BGRID)]}
        gstar = max(GRID, key=lambda g: (dev[f"H{g}"], -g))
        bstar = max(BGRID, key=lambda b: (dev[f"B{b}"], -b))
        ev = {p: tot(p, s["eval"]) for p in ["P1_posts", "P2_edges", "P3_authors", "P9_oracle", f"H{gstar}", f"B{bstar}",
                                              "SEM_M", "SEM_AM", "SEM_LC"]}
        ev_grid = {g: tot(f"H{g}", s["eval"]) for g in GRID}
        # E3 rows: per evaluation topic-day
        for (t, d), c in cells.items():
            if d in set(s["eval"]) and c["P1_posts"] > 0:
                rows_e3.append((name, str(d), np.log1p(c["target"]), np.log1p(c[f"H{gstar}"]),
                                np.log1p(c["P2_edges"]) - np.log1p(c[f"H{gstar}"]), conv.get((t, d))))
        # E4 top-k boundary discordance on history+development days
        pre = list(s["hist"]) + list(s["dev"])
        shares = []
        for d in pre:
            T = sorted([t for t in topics if (t, d) in cells], key=lambda t: -cells[(t, d)]["P1_posts"])[:6]
            if len(T) < 2:
                continue
            neff = {}
            for t in T:
                cnt = Counter(s["info"][x]["author_id"] for x in by[(t, d)])
                n = sum(cnt.values())
                neff[t] = 1 / sum((v / n) ** 2 for v in cnt.values())
            pairs = [(a, b) for i, a in enumerate(T) for b in T[i + 1:]]
            dis = sum((cells[(a, d)]["P1_posts"] - cells[(b, d)]["P1_posts"]) * (neff[a] - neff[b]) < 0 for a, b in pairs)
            shares.append(dis / len(pairs))
        R[name] = {"gamma_star": gstar, "b_star": bstar, "dev": dev, "eval": ev, "eval_by_gamma": ev_grid,
                   "C_k": float(np.mean(shares)) if shares else None,
                   "conv_share": float(np.nanmean([v for v in conv.values() if v is not None])) if any(v is not None for v in conv.values()) else None}
        print(name, {k: v for k, v in R[name].items() if k not in ("dev", "eval_by_gamma")}, flush=True)
    # E1 pooled gamma (summed development yield over settings)
    pooled = max(GRID, key=lambda g: (sum(R[n]["dev"][f"H{g}"] for n in R), -g))
    out = {"settings": R, "pooled_gamma": pooled,
           "pooled_eval": {n: R[n]["eval_by_gamma"][pooled] for n in R}}
    # E3 regression with setting-day fixed effects
    def fit(rows, with_conv=False):
        by = defaultdict(list)
        for r in rows:
            if with_conv and r[5] is None:
                continue
            by[(r[0], r[1])].append(r)
        X, y = [], []
        for L in by.values():
            if len(L) < 2:
                continue
            A = np.array([[r[3], r[4]] + ([np.log(0.01 + r[5])] if with_conv else []) for r in L])
            X.append(A - A.mean(0))
            yy = np.array([r[2] for r in L])
            y.append(yy - yy.mean())
        X, y = np.vstack(X), np.concatenate(y)
        return np.linalg.lstsq(X, y, rcond=None)[0]
    keys = sorted({(r[0], r[1]) for r in rows_e3})
    grp = defaultdict(list)
    for r in rows_e3:
        grp[(r[0], r[1])].append(r)
    rng = np.random.default_rng(20261107)
    bs = [fit([r for k in [keys[i] for i in rng.integers(0, len(keys), len(keys))] for r in grp[k]])[1] for _ in range(500)]
    beta = fit(rows_e3)
    out["E3"] = {"coef_producer": float(beta[0]), "coef_residual": float(beta[1]),
                 "coef_residual_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(rows_e3)}
    if any(r[5] is not None for r in rows_e3):
        b2 = fit(rows_e3, with_conv=True)
        out["E3"]["with_conv"] = {"coef_residual": float(b2[1]), "coef_conv": float(b2[2])}
    # E4 / E6 cross-setting associations
    names = list(R)
    gain = [R[n]["eval"]["P2_edges"] / R[n]["eval"]["P1_posts"] - 1 for n in names]
    resid_gain = [R[n]["eval"]["P2_edges"] / R[n]["eval"]["P3_authors"] - 1 for n in names]
    if all(R[n]["C_k"] is not None for n in names):
        out["E4"] = {"rho_Ck_gain": float(spearmanr([R[n]["C_k"] for n in names], gain).statistic)}
    if all(R[n]["conv_share"] is not None for n in names):
        out["E6"] = {"rho_conv_resid_gain": float(spearmanr([R[n]["conv_share"] for n in names], resid_gain).statistic)}
    print(json.dumps({k: v for k, v in out.items() if k != "settings"}, indent=1, default=str))
    tag = "_known" if known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/dose2_theory{tag}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main("--known-only" in sys.argv)
