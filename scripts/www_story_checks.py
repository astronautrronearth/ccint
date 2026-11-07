"""Post-hoc checks for the 'same cut, opposite effects' story (2026-11-07; NOT pre-registered, labelled
exploratory in the paper and in www2027/results/frozen_eval/DEVIATIONS.md).

    python scripts/www_story_checks.py → outputs/interaction_experiment/www_eval/story_checks.json

1. Fragility of the discussion gain: leave-one-day-out and leave-two-days-out (the two largest days) for P2 and
   P4 vs P1, main cyber (monitor.json) and fresh cyber (contrast.json), k = 3; paired daily gains.
2. Discordance from history + development days only (no evaluation-day posts), for the five domain units.
3. Engagement counts on the event objective: registry claims when clusters are ranked by reply+quote edges or by
   distinct external responders (before the review day's cutoff), b = 5, exact match, same simulation as E-B.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402
from ccint.interaction import manifest as mf  # noqa: E402

FR = Path("www2027/results/frozen_eval")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fragility():
    out = {}
    mon = json.loads((FR / "monitor.json").read_text())["primary"]["k3"]
    # contrast.json keeps no per-day values: recompute fresh cyber and NBA with the frozen E-D code
    DOSE = _mod("dose_f", "scripts/www_eval_dose.py")
    CON = DOSE.CON
    m = mf.load("www_contrast")
    runs = CON.runs_for(m.sha256)
    srcs = {"cyber_main": mon}
    for dom, col in (("cyber", "in_cyber"), ("nba", "in_nba")):
        ids = [int(r["post_id"]) for r in m.population(col)]
        unit_of = (lambda r: [t for t, p in CON.PAT.items() if p.search(r["text"] or "")] or ["NBA"]) if dom == "nba" \
            else (lambda r: [r["topic"]] if r["topic"] != "no_topic" else [])
        cap = {}
        orig = CON.MON.evaluate

        def ev(cells, units, days, k, orig=orig, cap=cap):
            r = orig(cells, units, days, k)
            cap[k] = r
            return r
        CON.MON.evaluate = ev
        CON.domain(dom, ids, runs, unit_of)
        CON.MON.evaluate = orig
        srcs["cyber_fresh" if dom == "cyber" else dom] = cap[3]
    for name, r in srcs.items():
        p1 = np.array(r["P1_posts"]["per_day"])
        res = {"days": len(p1), "per_day_P1": p1.tolist()}
        for pol in ("P2_edges", "P4_responders"):
            v = np.array(r[pol]["per_day"])
            g = v - p1
            lodo = [(np.delete(v, i).sum() / np.delete(p1, i).sum() - 1) for i in range(len(p1))]
            l2o = [(np.delete(v, list(c)).sum() / np.delete(p1, list(c)).sum() - 1) for c in combinations(range(len(p1)), 2)]
            top2 = list(np.argsort(-g)[:2])
            res[pol] = {"gain": round(float(v.sum() / p1.sum() - 1), 4), "daily_diff": g.round(1).tolist(),
                        "days_positive": int((g > 0).sum()), "days_negative": int((g < 0).sum()),
                        "lodo_min": round(float(min(lodo)), 4), "lodo_max": round(float(max(lodo)), 4),
                        "leave_two_out_min": round(float(min(l2o)), 4),
                        "without_top2_days": round(float(np.delete(v, top2).sum() / np.delete(p1, top2).sum() - 1), 4)}
        out[name] = res
        print("fragility", name, {k: v for k, v in res.items() if isinstance(v, dict)}, flush=True)
    return out


def discordance_pre_eval():
    """C from history + development days only."""
    DOSE = _mod("dose", "scripts/www_eval_dose.py")
    SC, MON, CON, CC = DOSE.SC, DOSE.MON, DOSE.CON, DOSE.CC
    out = {}
    # main cyber
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    obs, _, info = SC.load(SC.scale_runs(m.sha256), ids)
    pre = set(MON.HIST) | set(MON.DEV)
    up = defaultdict(list)
    for s in obs:
        if info[s]["topic"] != "no_topic" and info[s]["published_at"].date() in pre:
            up[info[s]["topic"]].append(info[s]["author_id"])
    out["cyber_main_all"] = DOSE.concentration(up)
    pre2 = set(CON.HIST) | set(CON.DEV)
    m = mf.load("www_contrast")
    runs = CON.runs_for(m.sha256)
    for dom, col in (("cyber", "in_cyber"), ("nba", "in_nba")):
        ids = [int(r["post_id"]) for r in m.population(col)]
        obs, _, info = SC.load(runs, ids)
        unit_of = (lambda r: [t for t, p in CON.PAT.items() if p.search(r["text"] or "")] or ["NBA"]) if dom == "nba" \
            else (lambda r: [r["topic"]] if r["topic"] != "no_topic" else [])
        up = defaultdict(list)
        for s in obs:
            if info[s]["published_at"].date() in pre2:
                for u in unit_of(info[s]):
                    up[u].append(info[s]["author_id"])
        out["cyber_fresh" if dom == "cyber" else dom] = DOSE.concentration(up)
    m2 = mf.load("www_contrast2")
    with db.connect() as c:
        runs2 = [r["run_id"] for r in c.execute(
            "SELECT run_id FROM neighborhood_runs WHERE purpose = 'contrast2' AND manifest_sha256 = %s", (m2.sha256,))]
    for dom, terms in (("crypto", CC.CRYPTO_TERMS), ("ai", CC.AI_TERMS)):
        pat = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in terms}
        ids = [int(r["post_id"]) for r in m2.population(f"in_{dom}")]
        obs, _, info = SC.load(runs2, ids)
        up = defaultdict(list)
        for s in obs:
            if info[s]["published_at"].date() in pre2:
                for u in [t for t, p in pat.items() if p.search(info[s]["text"] or "")] or ["other"]:
                    up[u].append(info[s]["author_id"])
        out[dom] = DOSE.concentration(up)
    full = json.loads((FR / "dose.json").read_text())["units"]
    from scipy.stats import spearmanr
    keys = ["cyber_main_all", "cyber_fresh", "crypto", "ai", "nba"]
    res = {k: {"C_pre_eval": out[k][0], "C_full": full[k]["C"], "gain_edges": full[k]["gain_edges"]} for k in keys}
    xs = [res[k]["C_pre_eval"] for k in keys]
    ys = [res[k]["gain_edges"] for k in keys]
    res["rho_pre_eval_domains"] = round(float(spearmanr(xs, ys).statistic), 3)
    print("discordance", res, flush=True)
    return res


def registry_engagement():
    REG = _mod("reg", "scripts/www_eval_registry.py")
    SC = REG.SC
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    obs, edges, info = SC.load(SC.scale_runs(m.sha256), ids)
    hist_ids = [s for s in ids if info.get(s) and info[s]["published_at"].date() in REG.HIST]
    feeds = SC.feed_rules(hist_ids, info)["volume_only"]
    vic = [json.loads(l) for l in open("outputs/interaction_experiment/www_eval/victims.jsonl")]
    with db.connect() as c:
        reg = c.execute("""SELECT event_id, title, country, published_at_utc FROM external_events
                           WHERE source = 'ransomware.live' AND published_at_utc >= '2026-08-08'
                             AND published_at_utc < '2026-10-08'""").fetchall()
    clusters = defaultdict(list)
    for v in vic:
        s = v["post_id"]
        if s not in info or not v["victims"]:
            continue
        for name in sorted({REG.norm(x) for x in v["victims"]}):
            if len(name) >= 3:
                clusters[name].append(s)
    first = {k: min(info[s]["published_at"].date() for s in L) for k, L in clusters.items()}
    regn = [(r["event_id"], REG.norm(r["title"] or ""), r["published_at_utc"].date()) for r in reg]
    by_name = defaultdict(list)
    for e in regn:
        by_name[e[1]].append(e)
    M = {k: {e[0] for e in by_name.get(k, []) if first[k] - timedelta(14) <= e[2] <= first[k] + timedelta(14)}
         for k in clusters}
    cand = {s for L in clusters.values() for s in L}
    eb = defaultdict(list)
    for e in edges:
        if e["edge_type"] in ("reply", "quote") and e["seed_post_id"] in cand and e["event_created_at"] is not None:
            eb[e["seed_post_id"]].append(e)

    def cutoff(d):
        return REG.MON.cut(d) if hasattr(REG, "MON") else _mod("mon", "scripts/www_eval_monitor.py").cut(d)
    MONm = _mod("mon2", "scripts/www_eval_monitor.py")

    def score(k, d, pol):
        L = [s for s in clusters[k] if info[s]["published_at"].date() <= d]
        c = MONm.cut(d)
        if pol == "posts":
            return len(L)
        if pol == "edges":
            return sum(e["event_created_at"] < c for s in L for e in eb[s])
        if pol == "random":
            return 0
        if pol == "responders":
            return len({e["source_actor_id"] for s in L for e in eb[s]
                        if e["event_created_at"] < c and e["source_actor_id"] != info[s]["author_id"]})
        raise ValueError(pol)

    out = {"feed_share_of_candidate_posts": round(sum(info[s]["author_id"] in feeds for s in cand) / len(cand), 3),
           "edges_per_feed_post": round(float(np.mean([len(eb[s]) for s in cand if info[s]["author_id"] in feeds])), 3),
           "edges_per_nonfeed_post": round(float(np.mean([len(eb[s]) for s in cand if info[s]["author_id"] not in feeds])), 3)}
    zero = [k for k in clusters if sum(len(eb[s]) for s in clusters[k]) == 0]
    out["clusters_zero_edges_share"] = round(len(zero) / len(clusters), 3)
    out["candidate_posts_zero_edges_share"] = round(sum(len(eb[s]) == 0 for s in cand) / len(cand), 3)
    for pol in (("random",) if ONLY_RANDOM else ("posts", "edges", "responders")):
        r = np.random.default_rng([20261107, len(pol)])
        tot = []
        for _ in range(50):
            reviewed, got = set(), set()
            for d in REG.DAYS[7:]:
                pool = sorted(k for k in clusters if first[k] <= d and k not in reviewed)
                jit = r.random(len(pool)) * 1e-6
                sc = {k: score(k, d, pol) + jit[i] for i, k in enumerate(pool)}
                for k in sorted(pool, key=lambda k: -sc[k])[:5]:
                    reviewed.add(k)
                    if d in REG.EVAL:
                        got |= M[k]
            tot.append(len(got))
        out[pol] = {"claims_b5": round(float(np.mean(tot)), 2),
                    "claims_95_ties": [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))]}
        print("registry", pol, out[pol], flush=True)
    return out


ONLY_RANDOM = "--random-only" in sys.argv

if __name__ == "__main__":
    if ONLY_RANDOM:
        r = registry_engagement()
        Path("outputs/interaction_experiment/www_eval/story_checks_random.json").write_text(json.dumps(r, indent=1, default=str))
        sys.exit()
    res = {"note": "post-hoc, not pre-registered (2026-11-07)", "fragility": fragility(),
           "registry_engagement": registry_engagement(), "discordance_pre_eval": discordance_pre_eval()}
    Path("outputs/interaction_experiment/www_eval/story_checks.json").write_text(json.dumps(res, indent=1, default=str))
