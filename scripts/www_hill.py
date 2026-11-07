"""Hill-number family of producer-weighted counts (paper 2 theory): score_t = sum_i n_i^gamma over the producers i
of topic t on the decision day (gamma = 1: posts; gamma -> 0: distinct authors). gamma is fitted per setting on the
development days (grid 0..1), then evaluated on the evaluation days, k = 3; decomposition of the engagement gain:
share of (edges - posts) recovered for free by the fitted Hill count.

    python scripts/www_hill.py [--known-only] → outputs/interaction_experiment/www_eval/hill{_known}.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
MECH = None


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GRID = [round(x, 2) for x in np.linspace(0, 1, 11)]


def main(known_only):
    global MECH
    MECH = _mod("mech", "scripts/www_dose2_mech.py")
    MON = MECH.MON
    S = MECH.settings(known_only)
    out = {}
    for name, s in S.items():
        cells, by = MON.build(s["seeds"], s["info"], s["eb"], s["th"], s["feeds"])
        MON.add_persistent(cells, s["dev"])
        topics = sorted({t for t, _ in cells})
        for key, R in by.items():
            cnt = Counter(s["info"][x]["author_id"] for x in R)
            for g in GRID:
                cells[key][f"H{g}"] = float(sum(v ** g for v in cnt.values())) if g > 0 else float(len(cnt))
        MON.POLICIES[:] = [p for p in MON.POLICIES if not p.startswith("H")] + [f"H{g}" for g in GRID]

        def total(pol, days):
            return sum(MON.select_value(cells, topics, d, pol, 3, 20) for d in days)
        dev = {g: total(f"H{g}", s["dev"]) for g in GRID}
        gstar = max(GRID, key=lambda g: (dev[g], -abs(g - 0.5)))
        ev = {g: total(f"H{g}", s["eval"]) for g in GRID}
        base = {p: total(p, s["eval"]) for p in ("P1_posts", "P2_edges", "P3_authors", "P9_oracle")}
        gap = base["P2_edges"] - base["P1_posts"]
        out[name] = {"gamma_star": gstar, "dev_by_gamma": dev, "eval_by_gamma": ev, "eval": base,
                     "hill_eval": ev[gstar],
                     "free_share_of_engagement_gain": (ev[gstar] - base["P1_posts"]) / gap if gap else None,
                     "hill_vs_authors": ev[gstar] - base["P3_authors"]}
        print(name, {k: v for k, v in out[name].items() if k not in ("dev_by_gamma", "eval_by_gamma")}, flush=True)
    tag = "_known" if known_only else ""
    Path(f"outputs/interaction_experiment/www_eval/hill{tag}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main("--known-only" in sys.argv)
