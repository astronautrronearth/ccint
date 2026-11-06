"""E-D contrast domain: NBA vs concurrent cybersecurity (PREREG: www2027/results/frozen_eval/PREREG.md).

    python scripts/www_eval_contrast.py → outputs/interaction_experiment/www_eval/contrast.json

Units: NBA = the matched search term (team or league; a post matching several terms belongs to each);
cybersecurity = topic_v2s topic. Only edges within 24 h of the root are used in both domains.
Days 2026-10-21..2026-11-04: first 3 history, next 4 development, last 8 evaluation. Deviation: the
volume-only feed rule (>= 5 posts on >= 4 days) cannot be met in 3 history days; here >= 5 posts on all
3 history days.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SC = _mod("scale", "scripts/www_scale.py")
MON = _mod("mon", "scripts/www_eval_monitor.py")
E = SC.E
CC = _mod("cc", "scripts/collect_contrast.py")
D0 = date(2026, 10, 21)
DAYS = [D0 + timedelta(i) for i in range(15)]
HIST, DEV, EVAL = DAYS[:3], DAYS[3:7], DAYS[7:]
PAT = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in CC.NBA_TERMS}


def runs_for(sha):
    with db.connect() as c:
        return [r["run_id"] for r in c.execute(
            "SELECT run_id FROM neighborhood_runs WHERE purpose = 'contrast' AND manifest_sha256 = %s", (sha,))]


def domain(name, ids, runs, unit_of):
    obs, edges, info = SC.load(runs, ids)
    seeds = sorted(obs)
    eb = defaultdict(list)
    for e in edges:
        s = e["seed_post_id"]
        if s in info and e["edge_type"] in ("reply", "quote") and e["event_created_at"] is not None \
                and (e["event_created_at"] - info[s]["published_at"]).total_seconds() <= 86400:
            eb[s].append(e)
    with db.connect() as c:
        th = {r["post_id"]: r["text_hash"] for r in c.execute("SELECT post_id, text_hash FROM posts WHERE post_id = ANY(%s)", (seeds,))}
    # expand to (seed, unit) pairs
    pairs, pinfo, peb, pth = [], {}, {}, {}
    for s in seeds:
        for u in unit_of(info[s]):
            k = (s, u)
            pairs.append(k)
            pinfo[k] = dict(info[s], topic=u)
            peb[k] = eb[s]
            pth[k] = th.get(s)
    units = sorted({u for _, u in pairs})
    # concentration over units
    conc = {}
    for u in units:
        L = [k for k in pairs if k[1] == u]
        a = Counter(pinfo[k]["author_id"] for k in L)
        n = len(L)
        conc[u] = {"posts": n, "authors": len(a), "N_eff": 1 / sum((v / n) ** 2 for v in a.values())}
    big = [u for u in units if conc[u]["posts"] >= 30]
    tot = sum(len(eb[s]) for s in seeds)
    self_ = sum(e["source_actor_id"] == info[s]["author_id"] for s in seeds for e in eb[s])
    mech = E.self_mechanism(seeds, info, [e for s in seeds for e in eb[s]])
    hist_posts = Counter()
    hist_days = defaultdict(set)
    for s in seeds:
        d = info[s]["published_at"].date()
        if d in HIST:
            hist_posts[info[s]["author_id"]] += 1
            hist_days[info[s]["author_id"]].add(d)
    feeds = {a for a, n in hist_posts.items() if n >= 5 and len(hist_days[a]) >= 3}
    cells, _ = MON.build(pairs, pinfo, peb, pth, feeds)
    out = {"domain": name, "n_roots": len(seeds), "n_units": len(units), "units_ge30": len(big),
           "rho_posts_accounts": round(float(spearmanr([conc[u]["posts"] for u in big], [conc[u]["authors"] for u in big]).statistic), 3),
           "rho_posts_neff": round(float(spearmanr([conc[u]["posts"] for u in big], [conc[u]["N_eff"] for u in big]).statistic), 3),
           "top1_share_median": round(float(np.median([max(Counter(pinfo[k]["author_id"] for k in pairs if k[1] == u).values()) / conc[u]["posts"] for u in big])), 3),
           "edges_24h": tot, "self_share_24h": round(self_ / tot, 4) if tot else None,
           "continuation_share": mech["shares"].get("continuation"), "feeds_hist": len(feeds),
           "incidence_24h": round(float(np.mean([any(e["source_actor_id"] != info[s]["author_id"] for e in eb[s]) for s in seeds])), 4),
           "monitor": {}}
    for k in (1, 3, 5):
        r = MON.evaluate(cells, units, EVAL, k)
        out["monitor"][f"k{k}"] = {p: {kk: v[kk] for kk in ("total", "vs_posts", "vs_posts_ci", "vs_posts_rel", "vs_posts_rel_ci", "vs_edges", "vs_edges_ci")}
                                   for p, v in r.items() if isinstance(v, dict)}
        out["monitor"][f"k{k}"]["days_with_target"] = r["days_with_target"]
    print(name, {kk: out[kk] for kk in ("n_roots", "units_ge30", "rho_posts_accounts", "rho_posts_neff", "self_share_24h",
                                         "continuation_share", "incidence_24h")}, flush=True)
    print(name, "k3", {p: (v["total"], v["vs_posts_rel"], v["vs_posts_ci"]) for p, v in out["monitor"]["k3"].items() if isinstance(v, dict)}, flush=True)
    return out


def main():
    m = mf.load("www_contrast")
    runs = runs_for(m.sha256)
    nba = [int(r["post_id"]) for r in m.population("in_nba")]
    cyb = [int(r["post_id"]) for r in m.population("in_cyber")]
    nba_unit = lambda r: [t for t, p in PAT.items() if p.search(r["text"] or "")] or ["NBA"]
    cyb_unit = lambda r: [r["topic"]] if r["topic"] != "no_topic" else []
    res = {"runs": runs, "nba": domain("nba", nba, runs, nba_unit), "cyber": domain("cyber", cyb, runs, cyb_unit)}
    reading = {}
    for p in ("P3_authors", "P4_responders"):
        sig = []
        for d in ("nba", "cyber"):
            v = res[d]["monitor"]["k3"][p]
            lo, hi = v["vs_posts_ci"]
            sig.append(1 if lo > 0 else (-1 if hi < 0 else 0))
        reading[p] = "transfers" if sig[0] == sig[1] != 0 else f"boundary (nba {sig[0]}, cyber {sig[1]})"
    res["reading_k3"] = reading
    print(reading)
    p = Path("outputs/interaction_experiment/www_eval/contrast.json")
    p.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
