"""Budgeted acquisition of engagement for top-k topic monitoring (paper 2 method; design from the 2026-11-07 reviews,
docs/revision/PAPER2_1107/{astra_tech,fable_tech}.md). Offline replay: all neighbourhoods are cached; a policy pays one
unit per root post it "fetches" and only then sees that post's reply+quote edges created before the cutoff.

Score model (fitted on development days only): ridge of log(1 + next-day new responders) on free features
[log posts, log authors, burst z, log mean past yield] plus log(1 + E_t), E_t = the topic's engagement (edges before
the cutoff). At decision time E_t is replaced by an estimate from the fetched posts.

Policies at budget B = fraction of the day's posts:
  free        B = 0, E_t from the free prior (ridge of log1p E on free features)
  uniform     B * V posts uniformly over all of the day's posts; E_t = V_t * mean(e) over fetched posts of t
              (prior if none)
  stratified  the budget split evenly over topics (capped at V_t)
  boundary    1 post per topic, then repeatedly fetch one more post from the selected topic with the widest interval
              and from the outside topic with the highest upper bound (LUCB), until the budget is spent
  boundary_cv as boundary with a per-post control variate h(x) (ridge on bge embeddings, content type, author's past
              mean engagement; fitted on history + development posts): E_t = sum_t h + V_t * mean(e - h)
  full        B = 1 (true E_t)
Yield: union of new external responders over the k = 3 selected topics, summed over evaluation days.

    python scripts/www_budget_fetch.py [--known-only] [--seeds 10] → outputs/interaction_experiment/www_eval/budget_fetch{_known}.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
ECAP = 20
EMB = Path("outputs/interaction_experiment/embed")
PT = Path("outputs/interaction_experiment/ptype")
TYPES_L = ["question", "opinion", "news", "automated", "promotion", "other"]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MECH = _mod("mech", "scripts/www_dose2_mech.py")
MON = MECH.MON


def load_side():
    E, T = {}, {}
    for f in EMB.glob("*.npz"):
        z = np.load(f)
        for i, v in zip(z["ids"], z["emb"]):
            E[int(i)] = v.astype(np.float32)
    for f in PT.glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            T[r["post_id"]] = r["type"]
    return E, T


def pid(s):
    return s[0] if isinstance(s, tuple) else s


def ridge_fit(X, y, lam=1.0):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
    b = y.mean()
    return lambda x: float(((np.asarray(x, float) - mu) / sd) @ w + b)


def prepare(S, EMBV, TYPES):
    """Per (topic, day): post list with engagement e_p (edges before cutoff, capped), target responder set, features."""
    by = defaultdict(list)
    for s in S["seeds"]:
        r = S["info"][s]
        by[(r["topic"], r["published_at"].date())].append(s)
    cells = {}
    for (t, d), R in by.items():
        c = MON.cut(d)
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
            ep.append(min(k, ECAP))
        cnt = Counter(S["info"][s]["author_id"] for s in R)
        cells[(t, d)] = {"posts": R, "e": np.array(ep, float), "V": len(R), "A": len(cnt), "E": float(sum(ep)),
                         "T": after - before}
    hist = defaultdict(dict)
    for (t, d), c in cells.items():
        hist[t][d] = c
    for (t, d), c in cells.items():
        prev = [hist[t][x] for x in hist[t] if x < d]
        p = [x["V"] for x in prev]
        c["burst"] = (c["V"] - np.mean(p)) / (np.std(p) + 1.0) if len(p) >= 2 else 0.0
        c["hist_y"] = np.mean([len(x["T"]) for x in prev]) if prev else 0.0
        c["free"] = [np.log1p(c["V"]), np.log1p(c["A"]), c["burst"], np.log1p(c["hist_y"])]
    # author past mean engagement from history days only
    hist_days = set(S["hist"])
    ae = defaultdict(list)
    for (t, d), c in cells.items():
        if d in hist_days:
            for s, e in zip(c["posts"], c["e"]):
                ae[S["info"][s]["author_id"]].append(e)
    am = {a: float(np.mean(v)) for a, v in ae.items()}
    gm = float(np.mean([e for v in ae.values() for e in v])) if ae else 0.0
    for c in cells.values():
        X = []
        for s in c["posts"]:
            v = EMBV.get(pid(s))
            v = v if v is not None else np.zeros(384, np.float32)
            ty = TYPES.get(pid(s))
            oh = [1.0 if ty == x else 0.0 for x in TYPES_L]
            a = S["info"][s]["author_id"]
            X.append(np.concatenate([v, oh, [np.log1p(am.get(a, gm)), 1.0 if a in am else 0.0]]))
        c["X"] = np.array(X, np.float32)
    return cells


def fit_models(cells, S):
    dev, train_post_days = set(S["dev"]), set(S["hist"]) | set(S["dev"])
    Xs, ys, Xp, yp = [], [], [], []
    for (t, d), c in cells.items():
        if d in dev:
            Xs.append(c["free"] + [np.log1p(c["E"])])
            ys.append(np.log1p(len(c["T"])))
            Xp.append(c["free"])
            yp.append(np.log1p(c["E"]))
    score = ridge_fit(Xs, ys)
    prior = ridge_fit(Xp, yp)
    # per-post control variate on history + development posts (ridge on 392 features, closed form)
    X = np.vstack([c["X"] for (t, d), c in cells.items() if d in train_post_days])
    y = np.concatenate([np.log1p(c["e"]) for (t, d), c in cells.items() if d in train_post_days])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + 10.0 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    b = y.mean()
    h = lambda Xc: np.expm1(np.clip(((Xc - mu) / sd) @ w + b, 0, None))
    return score, prior, h


def choose(scores, k=3):
    return sorted(scores, key=lambda t: -scores[t])[:k]


def run_day(cells, d, score, prior, h, policy, B, rng, k=3, z=1.0):
    T = [t for (t, dd) in cells if dd == d]
    if not T:
        return 0.0, 0
    C = {t: cells[(t, d)] for t in T}
    Vtot = sum(c["V"] for c in C.values())
    budget = int(round(B * Vtot))
    pE = {t: max(0.0, float(np.expm1(prior(C[t]["free"])))) for t in T}
    hsum = {t: float(h(C[t]["X"]).sum()) if policy == "boundary_cv" else 0.0 for t in T}
    hp = {t: h(C[t]["X"]) if policy == "boundary_cv" else None for t in T}
    seen = {t: [] for t in T}
    order = {t: list(rng.permutation(C[t]["V"])) for t in T}

    def fetch(t):
        if order[t]:
            seen[t].append(order[t].pop())
            return 1
        return 0

    def est(t):
        c, n = C[t], len(seen[t])
        if policy == "full":
            return c["E"], 0.0
        if n == 0:
            return pE[t], pE[t]
        e = c["e"][seen[t]]
        if policy == "boundary_cv":
            r = e - hp[t][seen[t]]
            m = hsum[t] + c["V"] * r.mean()
            sdv = r.std(ddof=1) if n > 1 else max(1.0, abs(r.mean()))
        else:
            m = c["V"] * e.mean()
            sdv = e.std(ddof=1) if n > 1 else max(1.0, e.mean())
        se = c["V"] * sdv / np.sqrt(n) * np.sqrt(max(0.0, 1 - n / c["V"]))
        return max(0.0, m), se

    def q(t, Eh):
        return score(C[t]["free"] + [np.log1p(max(0.0, Eh))])
    used = 0
    if policy in ("uniform",):
        allp = [(t, i) for t in T for i in range(C[t]["V"])]
        for j in rng.permutation(len(allp))[:budget]:
            t, i = allp[j]
            seen[t].append(i)
        used = min(budget, len(allp))
    elif policy == "stratified":
        m = budget // max(1, len(T))
        for t in T:
            for _ in range(min(m, C[t]["V"])):
                used += fetch(t)
    elif policy in ("boundary", "boundary_cv"):
        for t in sorted(T, key=lambda t: -C[t]["V"]):
            if used >= budget:
                break
            used += fetch(t)
        while used < budget:
            mE = {t: est(t) for t in T}
            sc = {t: q(t, mE[t][0]) for t in T}
            top = choose(sc, k)
            out = [t for t in T if t not in top and order[t]]
            inn = [t for t in top if order[t]]
            if not out and not inn:
                break
            lo = {t: q(t, mE[t][0] - z * mE[t][1]) for t in inn}
            hi = {t: q(t, mE[t][0] + z * mE[t][1]) for t in out}
            if inn and used < budget:
                used += fetch(min(inn, key=lambda t: lo[t]))
            if out and used < budget:
                used += fetch(max(out, key=lambda t: hi[t]))
    sc = {t: q(t, est(t)[0]) for t in T}
    sel = choose(sc, k)
    run_day.last_sel = set(sel)
    return float(len(set().union(*[C[t]["T"] for t in sel]))), used


def run_setting(name, S, EMBV, TYPES, seeds):
    cells = prepare(S, EMBV, TYPES)
    score, prior, h = fit_models(cells, S)
    ev = list(S["eval"])
    Vday = {d: sum(c["V"] for (t, dd), c in cells.items() if dd == d) for d in ev}
    res = {"posts_eval": sum(Vday.values())}
    full_sel = {}
    for pol in ("free", "full"):
        rng = np.random.default_rng(1)
        y = 0.0
        for d in ev:
            y += run_day(cells, d, score, prior, h, pol, 0.0, rng)[0]
            if pol == "full":
                full_sel[d] = run_day.last_sel
            else:
                full_sel.setdefault("_free", {})[d] = run_day.last_sel
        res[pol] = {"yield": y, "cost": 0 if pol == "free" else res["posts_eval"]}
    fs = full_sel.pop("_free")
    res["free"]["overlap_with_full"] = float(np.mean([len(fs[d] & full_sel[d]) / 3 for d in ev]))
    # oracle and edges-only references
    res["oracle"] = sum(len(set().union(*[cells[(t, d)]["T"] for t in sorted([t for (t, dd) in cells if dd == d],
                        key=lambda t: -len(cells[(t, d)]["T"]))[:3]])) for d in ev)
    for pol in ("uniform", "stratified", "boundary", "boundary_cv"):
        res[pol] = {}
        for B in BUDGETS:
            ys, cs, ov = [], [], []
            for sd_ in range(seeds):
                rng = np.random.default_rng([20261107, sd_, int(B * 1000), len(pol)])
                yy, cc = 0.0, 0
                for d in ev:
                    o = run_day(cells, d, score, prior, h, pol, B, rng)
                    yy += o[0]
                    cc += o[1]
                    ov.append(len(run_day.last_sel & full_sel[d]) / 3)
                ys.append(yy)
                cs.append(cc)
            res[pol][str(B)] = {"yield": float(np.mean(ys)), "yield_sd": float(np.std(ys)), "cost": float(np.mean(cs)),
                                "overlap_with_full": float(np.mean(ov))}
    gap = res["full"]["yield"] - res["free"]["yield"]
    res["gap_full_minus_free"] = gap
    for pol in ("uniform", "stratified", "boundary", "boundary_cv"):
        rec = {B: ((v["yield"] - res["free"]["yield"]) / gap if gap > 0 else None) for B, v in res[pol].items()}
        res[pol + "_recovered"] = rec
    print(name, {k: (v if not isinstance(v, dict) else {kk: (round(vv["yield"], 1) if isinstance(vv, dict) else vv) for kk, vv in v.items()})
                 for k, v in res.items() if not k.endswith("_recovered")}, flush=True)
    print(name, "overlap", {"free": round(res["free"]["overlap_with_full"], 2), **{p: {B: round(v["overlap_with_full"], 2) for B, v in res[p].items()} for p in ("uniform", "stratified", "boundary", "boundary_cv")}}, flush=True)
    print(name, "recovered", {p: {B: (round(x, 2) if x is not None else None) for B, x in res[p + "_recovered"].items()}
                              for p in ("uniform", "stratified", "boundary", "boundary_cv")}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    EMBV, TYPES = load_side()
    S = MECH.settings(a.known_only)
    names = [n for n in S if not n.endswith("_terms")]
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    res = {n: run_setting(n, S[n], EMBV, TYPES, a.seeds) for n in names}
    tag = "_known" if a.known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/budget_fetch{tag}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
