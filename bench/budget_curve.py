"""Budget curves (exploratory, 2026-11-08): promotion rate vs attack budget per family and method.

For each family and budget R (0 = control: victim drawn, nothing injected). The attack rng is seeded by family, not
by budget, so victims are the same across budgets (except reply_to_replies, whose draw of answered replies consumes
random numbers in proportion to R). Reported per tier:
  promotion   share of attacked days on which the victim was NOT in the method's clean top k and is in the attacked
              top k (conditional on exclusion before the attack)
  coverage    attacked yield as % of the best possible pick
  actions     actual actions per attacked day (fake replies + fake posts; staged answers count as replies)
  rel_budget  actions / median replies of the day's k-th topic by replies (the decision boundary's volume)
    python bench/budget_curve.py outputs/ccbench_v1 → outputs/interaction_experiment/www_eval/ccbench_budget_v2.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ccbench as cb  # noqa: E402

GRID = [0, 10, 20, 40, 80, 160, 320]
FAM = {"sybil": "new", "reply_to_replies": "r2r_new", "post_and_reply": "post_reply", "staged_all": "all_staged",
       "post_flood": "post_flood", "slow_history": "history"}


def accounts(kind, R):
    if kind == "all_staged":
        return max(1, round(R / 1.5))
    return max(1, R)


def actions(kind, B, R):
    per = max(1, R // B) if R else 0
    return {"new": B * per, "r2r_new": B * per, "post_reply": B * per + B, "all_staged": B * per + B + B,
            "post_flood": B, "history": R}[kind]


PLACE = "top"


def main(data):
    S = cb.load(data)
    ms = [m for m in cb.extended_methods() if m.name in ("replies", "authors", "responders", "cap_answered",
                                                          "cross_anchored", "per_post_cap", "history_only", "ranker",
                                                          "gate", "gbm", "ranker+cross")]
    out = {"grid": GRID, "tiers": {}}
    res = {}
    # custom loop: we need the clean top k per method and day to condition on exclusion
    for n, s in S.items():
        held = next((x for x in cb.FOLDS if n in x), [])
        for m in ms:
            m.calibrate({d: cb.day_views(s, d) for d in s.hist})
            m.fit(cb._train_set(S, [x for x in cb.DEV if x not in held and x in S]))
        clean_top = {}
        for d in sorted(s.eval):
            v = cb.day_views(s, d)
            T = list(v)
            for m in ms:
                sc = m.score(v)
                clean_top[(m.name, d)] = set(sorted(T, key=lambda t: (-sc[t], t))[:3])
        res[n] = {}
        for f, kind in FAM.items():
            for R in GRID:
                rng = np.random.default_rng([20261108, len(kind), sum(map(ord, f))])
                B = accounts(kind, R) if R else 1
                k_, B_, R_ = (kind, B, R if kind != "post_flood" else 0)
                if kind == "post_flood":
                    B_ = R if R else 0
                prom = {m.name: [] for m in ms}
                yld = {m.name: 0.0 for m in ms}
                orc = 0
                for _ in range(cb.DRAWS):
                    state = {"pod_victim": None, "pod_ids": None, "pod_seen": set()}
                    for d in sorted(s.eval):
                        T = [t for (t, dd) in s.cells if dd == d]
                        if len(T) < 4:
                            continue
                        if R == 0:
                            low = sorted(T, key=lambda t: len(s.cells[(t, d)].T))[: max(1, len(T) // 2)]
                            vic, views = low[rng.integers(len(low))], cb.day_views(s, d)
                        else:
                            vic, views = cb._attacked_views(s, d, T, k_, B_, R_, rng, state, "low", PLACE)
                        orc += cb.best_union(s, d, T, 3)
                        for m in ms:
                            sc = m.score(views)
                            sel = sorted(T, key=lambda t: (-sc[t], t))[:3]
                            yld[m.name] += len(set().union(*[s.cells[(t, d)].T for t in sel]))
                            if vic not in clean_top[(m.name, d)]:
                                prom[m.name].append(vic in sel)
                res[n][(f, R)] = {m.name: (yld[m.name], orc, sum(prom[m.name]), len(prom[m.name])) for m in ms}
        print("done", n, flush=True)
    for g in ("dev", "held", "test"):
        names = [n for n, s in S.items() if s.tier == g]
        vol = []
        for n in names:
            s = S[n]
            for d in s.eval:
                cs = sorted([c.E for (t, dd), c in s.cells.items() if dd == d], reverse=True)
                if len(cs) >= 3:
                    vol.append(cs[2])
        v3 = float(np.median(vol))
        out["tiers"][g] = {"median_replies_3rd": v3, "families": {}}
        for f, kind in FAM.items():
            rows = {}
            for m in ms:
                pts = []
                for R in GRID:
                    y = sum(res[n][(f, R)][m.name][0] for n in names)
                    o = sum(res[n][(f, R)][m.name][1] for n in names)
                    p = sum(res[n][(f, R)][m.name][2] for n in names)
                    q = sum(res[n][(f, R)][m.name][3] for n in names)
                    a = actions(kind, accounts(kind, R), R) if R else 0
                    pts.append({"R": R, "actions": a, "rel_budget": round(a / v3, 2), "promotion": round(100 * p / max(1, q), 1),
                                "coverage": round(100 * y / max(1, o), 1)})
                rows[m.name] = pts
            out["tiers"][g]["families"][f] = rows
    Path(f"outputs/interaction_experiment/www_eval/ccbench_budget_v2{'' if PLACE == 'top' else '_' + PLACE}.json").write_text(json.dumps(out, indent=1))
    for f in FAM:
        print(f, "test promotion")
        for m in ms:
            print("   ", m.name.ljust(15), [p["promotion"] for p in out["tiers"]["test"]["families"][f][m.name]])


if __name__ == "__main__":
    PLACE = sys.argv[2] if len(sys.argv) > 2 else "top"
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/ccbench_v1")
