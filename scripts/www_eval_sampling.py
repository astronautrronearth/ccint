"""E-C reporting protocol under partial neighbourhood collection (PREREG: www2027/results/frozen_eval/PREREG.md).

    python scripts/www_eval_sampling.py → outputs/interaction_experiment/www_eval/sampling.json

Quantity: per topic-week, the share of roots with >= 1 external reply or quote within 24 h of the root.
Designs: uniform and stratified (topic x producer activity) root sampling at 10/25/50 %, 500 repetitions.
Producer activity = the author's posts in the history days (days 1-7): <= 1, 2-4, 5+ (authors absent from the
history days fall in <= 1; the PREREG listed 1, 2-4, 5+). Cost = API pages actually requested per root.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SC = _mod("scale", "scripts/www_scale.py")
D0 = date(2026, 8, 22)
rng = np.random.default_rng(20261106)


def main():
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    runs = SC.scale_runs(m.sha256)
    obs, edges, info = SC.load(runs, ids)
    seeds = sorted(s for s in obs if info[s]["topic"] != "no_topic")
    y = Counter()
    for e in edges:
        s = e["seed_post_id"]
        if s in info and e["edge_type"] in ("reply", "quote") and e["source_actor_id"] != info[s]["author_id"] \
                and e["event_created_at"] is not None \
                and (e["event_created_at"] - info[s]["published_at"]).total_seconds() <= 86400:
            y[s] = 1
    with db.connect() as c:
        pages = Counter()
        for r in c.execute("""SELECT seed_post_id, sum(n_pages) AS p FROM interaction_seed_outcomes
                              WHERE collection_run_id = ANY(%s) AND seed_post_id = ANY(%s) GROUP BY 1""", (runs, seeds)):
            pages[r["seed_post_id"]] = int(r["p"] or 0)
    hist = Counter(info[s]["author_id"] for s in ids if info.get(s) and (info[s]["published_at"].date() - D0).days < 7)
    act = lambda s: 0 if hist[info[s]["author_id"]] <= 1 else (1 if hist[info[s]["author_id"]] <= 4 else 2)
    week = lambda s: min((info[s]["published_at"].date() - D0).days // 7, 4)
    cells = defaultdict(list)
    for s in seeds:
        cells[(info[s]["topic"], week(s))].append(s)
    truth = {k: float(np.mean([y[s] for s in L])) for k, L in cells.items()}
    topics = sorted({t for t, _ in cells})
    weeks = sorted({w for _, w in cells})
    total_pages = sum(pages[s] for s in seeds)
    top3_true = {w: set(sorted(topics, key=lambda t: -truth.get((t, w), -1))[:3]) for w in weeks}
    out = {"n_roots": len(seeds), "incidence_overall": round(float(np.mean([y[s] for s in seeds])), 4),
           "pages_full": total_pages, "pages_per_root_mean": round(total_pages / len(seeds), 3),
           "activity_note": "history-day posts <= 1, 2-4, 5+ (PREREG said 1, 2-4, 5+)", "designs": {}}
    strata_of = {k: defaultdict(list) for k in cells}
    for k, L in cells.items():
        for s in L:
            strata_of[k][act(s)].append(s)
    for design in ("uniform", "stratified"):
        for frac in (0.10, 0.25, 0.50):
            err, cover, top3, used = [], [], [], []
            for _ in range(500):
                est, ok, pg = {}, {}, 0
                for k, L in cells.items():
                    if design == "uniform":
                        n = max(1, int(round(frac * len(L))))
                        S = rng.choice(L, size=n, replace=False)
                        p = float(np.mean([y[s] for s in S]))
                        fpc = 1 - n / len(L)
                        se = np.sqrt(max(p * (1 - p), 1e-9) / n * fpc)
                        pg += sum(pages[s] for s in S)
                    else:
                        p, var = 0.0, 0.0
                        for h, Lh in strata_of[k].items():
                            n = max(1, int(round(frac * len(Lh))))
                            S = rng.choice(Lh, size=n, replace=False)
                            ph = float(np.mean([y[s] for s in S]))
                            w = len(Lh) / len(L)
                            p += w * ph
                            var += w * w * max(ph * (1 - ph), 1e-9) / n * (1 - n / len(Lh))
                            pg += sum(pages[s] for s in S)
                        se = np.sqrt(var)
                    est[k] = p
                    ok[k] = abs(p - truth[k]) <= 1.96 * se + 1e-12
                err.append(np.mean([abs(est[k] - truth[k]) for k in cells]))
                cover.append(np.mean(list(ok.values())))
                top3.append(np.mean([len(set(sorted(topics, key=lambda t: -est.get((t, w), -1))[:3]) & top3_true[w]) / 3
                                     for w in weeks]))
                used.append(pg)
            out["designs"][f"{design}_{int(frac * 100)}"] = {
                "mae": round(float(np.mean(err)), 4), "coverage95": round(float(np.mean(cover)), 3),
                "top3_recovery": round(float(np.mean(top3)), 3),
                "pages_share": round(float(np.mean(used)) / total_pages, 3)}
            print(design, frac, out["designs"][f"{design}_{int(frac * 100)}"], flush=True)
    p = Path("outputs/interaction_experiment/www_eval/sampling.json")
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
