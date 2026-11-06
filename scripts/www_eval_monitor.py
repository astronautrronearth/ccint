"""E-A discussion monitoring and E-E equal-budget suppression (PREREG: www2027/results/frozen_eval/PREREG.md).

    python scripts/www_eval_monitor.py → outputs/interaction_experiment/www_eval/monitor.json

Decision per day d: rank topics from roots published on day d, using only posts and edges before the cutoff
(end of day d); select the top k. Target: distinct external accounts that respond to those roots in the next
24 h and had not responded before the cutoff. Ties in a policy's score are broken at random (mean over 100
orders; not specified in the PREREG, recorded here). Sensitivity without the 'unassigned' topic is an added
analysis, labelled as such in the output.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
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
DAYS = [D0 + timedelta(i) for i in range(33)]
HIST, DEV = DAYS[:7], DAYS[7:19]
# the P_scale manifest has no roots after 2026-09-21 (rules_v2 labels ended there when it was frozen), so the
# PREREG's evaluation days 20-33 reduce to the 12 days 2026-09-10..21 (recorded in DEVIATIONS.md)
EVAL = [d for d in DAYS[19:] if d <= date(2026, 9, 21)]
POLICIES = ["P1_posts", "P2_edges", "P3_authors", "P4_responders", "P5_dedup", "P6_capped", "P7_nofeed",
            "P8_nonself_edges", "P9_oracle", "P10_persistent"]
rng = np.random.default_rng(20261106)


def cut(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=1)


def build(seeds, info, edges_by, text_hash, feeds_hist):
    """Per (topic, day): policy scores and the target."""
    by = defaultdict(list)
    for s in seeds:
        by[(info[s]["topic"], info[s]["published_at"].date())].append(s)
    cells = {}
    for (t, d), R in by.items():
        c = cut(d)
        c2 = c + timedelta(days=1)
        before, after = set(), set()
        n_edges = n_nonself = 0
        for s in R:
            a = info[s]["author_id"]
            for e in edges_by[s]:
                ts = e["event_created_at"]
                if ts is None:
                    continue
                ext = e["source_actor_id"] != a
                if ts < c:
                    n_edges += 1
                    n_nonself += ext
                    if ext:
                        before.add(e["source_actor_id"])
                elif ts < c2 and ext:
                    after.add(e["source_actor_id"])
        auth = Counter(info[s]["author_id"] for s in R)
        cells[(t, d)] = {
            "P1_posts": len(R), "P2_edges": n_edges, "P3_authors": len(auth), "P4_responders": len(before),
            "P5_dedup": len({text_hash.get(s) or s for s in R}), "P6_capped": sum(min(v, 3) for v in auth.values()),
            "P7_nofeed": sum(info[s]["author_id"] not in feeds_hist for s in R), "P8_nonself_edges": n_nonself,
            "target": len(after - before)}
        cells[(t, d)]["P9_oracle"] = cells[(t, d)]["target"]
    return cells, by


def add_persistent(cells, dev_days):
    """P10 (added after the second blind review): rank topics by their mean target over the development days,
    fixed for the evaluation days, i.e. persistent topic differences without any same-day information."""
    dev = defaultdict(list)
    for (t, d), v in cells.items():
        if d in set(dev_days):
            dev[t].append(v["target"])
    mean = {t: float(np.mean(v)) for t, v in dev.items()}
    for (t, d), v in cells.items():
        v["P10_persistent"] = mean.get(t, 0.0)
    return cells


def select_value(cells, topics, d, pol, k, n_orders=100):
    """Expected target of the top-k topics under random tie-breaking (own seeded stream per call: stable)."""
    rng = np.random.default_rng([20261106, POLICIES.index(pol), k, d.toordinal()])
    sc = np.array([cells.get((t, d), {}).get(pol, 0) for t in topics], float)
    tg = np.array([cells.get((t, d), {}).get("target", 0) for t in topics], float)
    vals = []
    for _ in range(n_orders):
        jitter = rng.random(len(topics)) * 1e-6
        idx = np.argsort(-(sc + jitter))[:k]
        vals.append(tg[idx].sum())
    return float(np.mean(vals))


def evaluate(cells, topics, days, k):
    per_day = {p: np.array([select_value(cells, topics, d, p, k) for d in days]) for p in POLICIES}
    out = {}
    base, edges = per_day["P1_posts"], per_day["P2_edges"]
    idx = [rng.integers(0, len(days), len(days)) for _ in range(2000)]
    for p, v in per_day.items():
        diff, de = v - base, v - edges
        bs = [diff[i].sum() for i in idx]
        # relative gain bootstrapped as a ratio of resampled totals (not absolute CI / fixed baseline)
        br = [v[i].sum() / base[i].sum() - 1 for i in idx if base[i].sum() > 0]
        be = [de[i].sum() for i in idx]
        out[p] = {"total": round(float(v.sum()), 1), "vs_posts": round(float(diff.sum()), 1),
                  "vs_posts_ci": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                  "vs_posts_rel": round(float(diff.sum() / base.sum()), 4) if base.sum() else None,
                  "vs_posts_rel_ci": ([round(float(np.percentile(br, 2.5)), 4), round(float(np.percentile(br, 97.5)), 4)]
                                      if br else [None, None]),
                  "vs_edges": round(float(de.sum()), 1),
                  "vs_edges_ci": [round(float(np.percentile(be, 2.5)), 1), round(float(np.percentile(be, 97.5)), 1)],
                  "per_day": [round(float(x), 2) for x in v]}
    out["days_with_target"] = int(sum(any(cells.get((t, d), {}).get("target", 0) > 0 for t in topics) for d in days))
    return out


def suppress(seeds, info, B, how, text_hash):
    """Return the set of eval-day seeds kept after removing exactly B posts."""
    ev = [s for s in seeds if info[s]["published_at"].date() in set(EVAL)]
    if how == "random":
        drop = set(rng.choice(len(ev), size=B, replace=False))
        return [s for i, s in enumerate(ev) if i not in drop]
    if how == "top_output":
        cnt = Counter(info[s]["author_id"] for s in ev)
        drop, n = set(), 0
        for a, _ in cnt.most_common():
            for s in [x for x in ev if info[x]["author_id"] == a]:
                if n >= B:
                    break
                drop.add(s)
                n += 1
            if n >= B:
                break
        return [s for s in ev if s not in drop]
    if how == "dedup_then_random":
        seen, drop = set(), []
        for s in sorted(ev, key=lambda s: info[s]["published_at"]):
            h = (info[s]["topic"], info[s]["published_at"].date(), text_hash.get(s))
            if text_hash.get(s) and h in seen:
                drop.append(s)
            seen.add(h)
        drop = drop[:B]
        rest = [s for s in ev if s not in set(drop)]
        extra = B - len(drop)
        if extra > 0:
            idx = set(rng.choice(len(rest), size=extra, replace=False))
            rest = [s for i, s in enumerate(rest) if i not in idx]
        return rest
    raise ValueError(how)


def main():
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    runs = SC.scale_runs(m.sha256)
    obs, edges, info = SC.load(runs, ids)
    seeds = sorted(s for s in obs if info[s]["topic"] != "no_topic")
    edges_by = defaultdict(list)
    for e in edges:
        if e["edge_type"] in ("reply", "quote") and e["seed_post_id"] in info:
            edges_by[e["seed_post_id"]].append(e)
    with db.connect() as c:
        text_hash = {r["post_id"]: r["text_hash"] for r in c.execute(
            "SELECT post_id, text_hash FROM posts WHERE post_id = ANY(%s)", (seeds,))}
    hist_ids = [s for s in ids if info.get(s) and info[s]["published_at"].date() in set(HIST)]
    feeds_hist = SC.feed_rules(hist_ids, info)["volume_only"]
    cells, by = build(seeds, info, edges_by, text_hash, feeds_hist)
    add_persistent(cells, DEV)
    topics_all = sorted({t for t, _ in cells})
    topics_na = [t for t in topics_all if t != "unassigned"]
    out = {"n_seeds": len(seeds), "topics": topics_all, "feeds_hist_accounts": len(feeds_hist),
           "split_days": {"history": [str(HIST[0]), str(HIST[-1])], "dev": [str(DEV[0]), str(DEV[-1])],
                          "eval": [str(EVAL[0]), str(EVAL[-1])]},
           "tie_break": "mean over 100 random orders (not in PREREG)", "primary": {}, "dev": {},
           "added_without_unassigned": {}}
    for k in (1, 3, 5):
        out["primary"][f"k{k}"] = evaluate(cells, topics_all, EVAL, k)
        out["dev"][f"k{k}"] = evaluate(cells, topics_all, DEV, k)
        out["added_without_unassigned"][f"k{k}"] = evaluate(cells, topics_na, EVAL, k)
        print("k", k, {p: (v["total"], v["vs_posts"], v["vs_posts_ci"]) for p, v in out["primary"][f"k{k}"].items()
                       if isinstance(v, dict)}, flush=True)
    # E-E equal-budget suppression: outcome is the E-A target under P1 after suppression (k = 3)
    ev = [s for s in seeds if info[s]["published_at"].date() in set(EVAL)]
    B = sum(info[s]["author_id"] in feeds_hist for s in ev)
    supp = {"B": B, "n_eval_posts": len(ev)}

    def p1_after(kept, k=3):
        kept_by = Counter((info[s]["topic"], info[s]["published_at"].date()) for s in kept)
        cc = {key: dict(v, P1_posts=kept_by.get(key, 0)) for key, v in cells.items()}
        return {"target_total": round(sum(select_value(cc, topics_all, d, "P1_posts", k, 20) for d in EVAL), 1),
                "distinct_producers_kept": len({info[s]["author_id"] for s in kept})}
    supp["feed_removal"] = p1_after([s for s in ev if info[s]["author_id"] not in feeds_hist])
    supp["none"] = p1_after(ev)
    for how, reps in (("random", 200), ("top_output", 1), ("dedup_then_random", 50)):
        res = [p1_after(suppress(seeds, info, B, how, text_hash)) for _ in range(reps)]
        tt = np.array([r["target_total"] for r in res])
        supp[how] = {"target_total_mean": round(float(tt.mean()), 1),
                     "target_total_95": [round(float(np.percentile(tt, 2.5)), 1), round(float(np.percentile(tt, 97.5)), 1)],
                     "distinct_producers_kept_mean": round(float(np.mean([r["distinct_producers_kept"] for r in res])), 1),
                     "reps": reps}
        print(how, supp[how], flush=True)
    out["E_E_suppression_k3"] = supp

    # E-C2 (added after the blind review; not in the PREREG): the actual selection policies when only a uniform
    # sample of each day's roots has its neighbourhood collected. Scores use the sampled roots only; the target is
    # always computed on all roots.
    by_cell = defaultdict(list)
    for s in seeds:
        by_cell[(info[s]["topic"], info[s]["published_at"].date())].append(s)
    ec2 = {}
    r2 = np.random.default_rng(20261109)
    for frac in (0.10, 0.25, 0.50):
        res = {"P2_edges": [], "P4_responders": []}
        for _ in range(100):
            cc = {}
            for key, L in by_cell.items():
                n = max(1, int(round(frac * len(L))))
                S = [L[i] for i in r2.choice(len(L), size=n, replace=False)]
                c = cut(key[1])
                resp, ne = set(), 0
                for s in S:
                    for e in edges_by[s]:
                        if e["event_created_at"] is not None and e["event_created_at"] < c:
                            ne += 1
                            if e["source_actor_id"] != info[s]["author_id"]:
                                resp.add(e["source_actor_id"])
                cc[key] = dict(cells[key], P2_edges=ne, P4_responders=len(resp))
            for pol in res:
                res[pol].append(sum(select_value(cc, topics_all, d, pol, 3, 10) for d in EVAL))
        ec2[f"frac{int(frac * 100)}"] = {pol: {"mean": round(float(np.mean(v)), 1),
                                              "p95": [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)]}
                                        for pol, v in res.items()}
        print("E-C2", frac, ec2[f"frac{int(frac * 100)}"], flush=True)
    out["E_C2_partial_policy_k3"] = ec2
    p = Path("outputs/interaction_experiment/www_eval/monitor.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
