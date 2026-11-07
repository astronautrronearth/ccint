"""Cost to steal a slot (exploratory, 2026-11-07, after RESULTS_DOSE3): attack-agnostic robustness.

For every evaluation day and every topic in the lower half by true yield (the victims of the attack simulation),
the minimum number of fake actions that lifts the victim's score strictly above the third-best other topic's score,
when the attacker knows the count and its constants. Action costs (one action = one post or one reply by an
account the attacker controls; fresh accounts are free to create):
  +1 R (reply/quote edge)                       1 action (a reply)
  +1 A, +1 V (a new author with one post)       1 action (a post)
  +1 C_recv (a depth-1 replier answered by another account):
      answer an unanswered real depth-1 reply   1 action, also +1 R (at most |d1u| such)
      staged pair (fake reply + fake answer)    2 actions, also +2 R
Counts: edges R; posts V; authors A; anchA = min(R, kA(1+A)); anchR = min(R, kC(1+C_recv));
M3 anchAR_q90 = min(R, kA(1+A), kC(1+C_recv)); kappas as in www_manipulation.py (q90 on history days).
Ceiling: any score computed from the interaction graph without account history is matched by cloning the
third-best topic's graph with fresh accounts, so no such count can force more than clone = V_3 + R_3 actions
(posts and replies of the count's third-best topic); ratio_to_clone = cost / clone.
Output: per setting the median and mean cost and the cost in multiples of the edges cost.

    python scripts/www_cost.py → outputs/interaction_experiment/www_eval/cost.json
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("man", "scripts/www_manipulation.py")
MAN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MAN)
COUNTS = ["edges", "posts", "authors", "anchA", "anchR", "anchAR"]


def score(c, m, kA, kC):
    R, A, C = c["E"], c["A"], len(c["recv"])
    return {"edges": R, "posts": c["V"], "authors": A, "anchA": min(R, kA * (1 + A)),
            "anchR": min(R, kC * (1 + C)), "anchAR": min(R, kA * (1 + A), kC * (1 + C))}[m]


def above(s):
    """smallest integer strictly above s"""
    return math.floor(s) + 1


def cost(c, m, s, kA, kC):
    R, A, V, C, U = c["E"], c["A"], c["V"], len(c["recv"]), len(c["d1u"])
    if m == "edges":
        return max(0, above(s) - R)
    if m == "posts":
        return max(0, above(s) - V)
    if m == "authors":
        return max(0, above(s) - A)
    need_R = above(s) - R
    need_A = max(0, math.ceil(s / kA - 1 + 1e-9) - A) if m in ("anchA", "anchAR") else 0
    need_C = max(0, math.ceil(s / kC - 1 + 1e-9) - C) if m in ("anchR", "anchAR") else 0
    if m in ("anchA", "anchAR") and kA * (1 + A + need_A) <= s:
        need_A += 1
    if m in ("anchR", "anchAR") and kC * (1 + C + need_C) <= s:
        need_C += 1
    cheap = min(need_C, U)
    actC = cheap + 2 * (need_C - cheap)
    gotR = cheap + 2 * (need_C - cheap)
    return need_A + actC + max(0, need_R - gotR)


def run(S):
    cells = MAN.cells_of(S)
    hc = [c for (t, d), c in cells.items() if d in set(S["hist"])]
    kC = float(np.quantile([c["E"] / (1 + len(c["recv"])) for c in hc], 0.9)) if hc else 4.0
    kA = float(np.quantile([c["E"] / (1 + c["A"]) for c in hc], 0.9)) if hc else 4.0
    out = {m: [] for m in COUNTS}
    clone = {m: [] for m in COUNTS}
    for d in sorted(S["eval"]):
        T = [t for (t, dd) in cells if dd == d]
        if len(T) < 4:
            continue
        low = sorted(T, key=lambda t: len(cells[(t, d)]["T"]))[: max(1, len(T) // 2)]
        for v in low:
            for m in COUNTS:
                others = sorted(((score(cells[(t, d)], m, kA, kC), t) for t in T if t != v), reverse=True)
                s3, t3 = others[2] if len(others) >= 3 else others[-1]
                out[m].append(cost(cells[(v, d)], m, s3, kA, kC))
                clone[m].append(cells[(t3, d)]["V"] + cells[(t3, d)]["E"])
    res = {m: {"median": float(np.median(x)), "mean": float(np.mean(x)), "n": len(x)} for m, x in out.items()}
    e = np.array(out["edges"], float)
    for m in COUNTS:
        x = np.array(out[m], float)
        cl = np.array(clone[m], float)
        res[m]["median_ratio_to_clone"] = float(np.median(x / np.maximum(1, cl)))
        res[m]["sum_ratio_to_clone"] = float(x.sum() / cl.sum())
        res[m]["median_ratio_to_edges"] = float(np.median(x[e > 0] / e[e > 0])) if (e > 0).any() else None
    res["_raw"] = out
    res["_clone"] = clone
    res["_kappa"] = {"kA": kA, "kC": kC}
    return res


def main():
    SS = MAN.MECH.settings(False)
    res = {}
    for n in [x for x in SS if not x.endswith("_terms")]:
        res[n] = run(SS[n])
        print(n, {m: (round(r["median"]), round(r["median_ratio_to_edges"] or 0, 2), round(r["sum_ratio_to_clone"], 2))
                  for m, r in res[n].items() if not m.startswith("_")}, flush=True)
    Path("outputs/interaction_experiment/www_eval/cost.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
