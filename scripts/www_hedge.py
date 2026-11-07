"""Online expert mixture (Hedge) for daily top-k topic selection (paper 2 method).

Experts rank a day's topics by one signal each: posts, distinct authors, reply+quote edges (counts at the cutoff),
peak-5 engagement, past mean yield (habitual audience), yesterday's new responders. At the cutoff of day d the target
of day d-1 is fully observed, so every expert's counterfactual union yield on d-1 is known. Weights
w_j ∝ exp(eta * sum of past normalised yields), yields normalised by the day's best expert (in [0, 1]); the
decision ranks topics by the weighted mean of the experts' within-day percentile ranks and picks the top k.
History and development days warm-start the weights (their outcomes are past at every evaluation decision);
eta = sqrt(8 ln N / T) with T the number of warm-start + evaluation days.
Guarantee (Hedge, Freund & Schapire 1997): for normalised yields the regret against the best single expert in
hindsight is at most sqrt(T ln N / 2) days of normalised yield, per setting, whichever expert that is.

    python scripts/www_hedge.py [--known-only] → outputs/interaction_experiment/www_eval/hedge{_known}.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("bf", "scripts/www_budget_fetch.py")
BF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BF)
BF.ECAP = 10 ** 9
K = 3


def experts(cells, hist):
    def yday(t, d):
        c = hist[t].get(d - timedelta(days=1))
        return len(c["T"]) if c else 0.0
    return {"posts": lambda t, d: cells[(t, d)]["V"], "authors": lambda t, d: cells[(t, d)]["A"],
            "edges": lambda t, d: cells[(t, d)]["e"].sum(),
            "peak5": lambda t, d: np.sort(cells[(t, d)]["e"])[::-1][:5].sum(),
            "habitual": lambda t, d: cells[(t, d)]["hist_y"], "yesterday": yday}


def day_yield(cells, d, sel):
    return len(set().union(*[cells[(t, d)]["T"] for t in sel])) if sel else 0


def top(T, score):
    return sorted(T, key=lambda t: (-score[t], t))[:K]


def run(S):
    cells = BF.prepare(S, {}, {})
    hist = {}
    for (t, d), c in cells.items():
        hist.setdefault(t, {})[d] = c
    E = experts(cells, hist)
    names = list(E)
    warm = sorted(set(S["hist"]) | set(S["dev"]))
    ev = sorted(S["eval"])
    Tn = len(warm) + len(ev)
    eta = np.sqrt(8 * np.log(len(names)) / Tn)
    cum = np.zeros(len(names))
    out = {"hedge": 0.0, **{n: 0.0 for n in names}, "oracle": 0.0, "weights": []}

    def expert_day(d):
        T = [t for (t, dd) in cells if dd == d]
        sc = {n: {t: E[n](t, d) for t in T} for n in names}
        y = np.array([day_yield(cells, d, top(T, sc[n])) for n in names], float)
        return T, sc, y
    for d in warm:
        T, sc, y = expert_day(d)
        if T and y.max() > 0:
            cum += y / y.max()
    for d in ev:
        T, sc, y = expert_day(d)
        if not T:
            continue
        w = np.exp(eta * (cum - cum.max()))
        w /= w.sum()
        pr = {n: dict(zip(T, rankdata([sc[n][t] for t in T]) / len(T))) for n in names}
        mix = {t: sum(w[j] * pr[n][t] for j, n in enumerate(names)) for t in T}
        out["hedge"] += day_yield(cells, d, top(T, mix))
        for j, n in enumerate(names):
            out[n] += y[j]
        out["oracle"] += day_yield(cells, d, top(T, {t: len(cells[(t, d)]["T"]) for t in T}))
        out["weights"].append(dict(zip(names, map(float, w))))
        if y.max() > 0:
            cum += y / y.max()          # today's outcome becomes known at tomorrow's cutoff
    out["final_weights"] = out["weights"][-1] if out["weights"] else None
    out["best_expert"] = max(names, key=lambda n: out[n])
    out["eta"] = float(eta)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-only", action="store_true")
    a = ap.parse_args()
    SS = BF.MECH.settings(a.known_only)
    res = {}
    for n in [x for x in SS if not x.endswith("_terms")]:
        r = run(SS[n])
        res[n] = r
        print(n, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k not in ("weights",)}, flush=True)
    tag = "_known" if a.known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/hedge{tag}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
