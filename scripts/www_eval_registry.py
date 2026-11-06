"""E-B registry-claim discovery (PREREG: www2027/results/frozen_eval/PREREG.md; deviations in DEVIATIONS.md).

    python scripts/www_eval_registry.py → outputs/interaction_experiment/www_eval/registry.json
                                          (+ registry_fuzzy_review.csv for blind human adjudication)

Candidates: victim names extracted from ransomware-topic P_scale roots (www_extract_victims.py), clustered by
normalised name. Each day from day 8, every policy reviews the b highest-scoring unreviewed clusters, scores using
posts up to that day only. A review recovers a registry claim when the cluster name matches a ransomware.live
record (exact normalised match = primary; fuzzy = sensitivity) published within 14 days before or after the
cluster's first post. Primary metric: unique claims recovered by reviews on the evaluation days (20-33).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from difflib import SequenceMatcher
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
HIST = set(DAYS[:7])
EVAL = {d for d in DAYS[19:] if d <= date(2026, 9, 21)}  # P_scale has no roots after 2026-09-21 (DEVIATIONS.md)
SUFFIX = r"\b(inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|gmbh|ag|sa|sas|plc|bv|nv|srl|spa|pty|lp|llp|pc|group|holdings?)\b"
POL = ["Q1_posts", "Q2_authors", "Q3_dedup", "Q4_capped", "Q5_nofeed"]
rng = np.random.default_rng(20261108)


def norm(x: str) -> str:
    x = x.lower()
    x = re.sub(r"https?://", "", x)
    x = re.sub(r"\.(com|net|org|co\.uk|ca|fr|de|io|us|gov|edu|local)\b", " ", x)
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(SUFFIX, " ", x)
    return re.sub(r"\s+", " ", x).strip()


def fuzzy(a: str, b: str) -> bool:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    inter = ta & tb
    if inter and (inter == ta or inter == tb) and max(len(t) for t in inter) >= 4:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.9


def main():
    m = mf.load("www_scale")
    ids = [int(r["post_id"]) for r in m.population("in_scale")]
    runs = SC.scale_runs(m.sha256)
    obs, edges, info = SC.load(runs, ids)
    hist_ids = [s for s in ids if info.get(s) and info[s]["published_at"].date() in HIST]
    feeds = SC.feed_rules(hist_ids, info)["volume_only"]
    vic = [json.loads(l) for l in open("outputs/interaction_experiment/www_eval/victims.jsonl")]
    with db.connect() as c:
        th = {r["post_id"]: r["text_hash"] for r in c.execute("SELECT post_id, text_hash FROM posts WHERE post_id = ANY(%s)",
                                                              ([v["post_id"] for v in vic],))}
        reg = c.execute("""SELECT event_id, title, country, published_at_utc FROM external_events
                           WHERE source = 'ransomware.live' AND published_at_utc >= '2026-08-08'
                             AND published_at_utc < '2026-10-08'""").fetchall()
    clusters = defaultdict(list)          # name -> posts
    for v in vic:
        s = v["post_id"]
        if s not in info or not v["victims"]:
            continue
        for name in sorted({norm(x) for x in v["victims"]}):
            if len(name) >= 3:
                clusters[name].append(s)
    first = {k: min(info[s]["published_at"].date() for s in L) for k, L in clusters.items()}
    regn = [(r["event_id"], norm(r["title"] or ""), r["published_at_utc"].date(), r["country"]) for r in reg]
    reg_by_name = defaultdict(list)
    for e in regn:
        reg_by_name[e[1]].append(e)

    def matches(k, use_fuzzy):
        lo, hi = first[k] - timedelta(14), first[k] + timedelta(14)
        out = {e[0] for e in reg_by_name.get(k, []) if lo <= e[2] <= hi}
        if use_fuzzy:
            out |= {e[0] for e in regn if lo <= e[2] <= hi and e[1] and fuzzy(k, e[1])}
        return out
    M = {False: {k: matches(k, False) for k in clusters}, True: {k: matches(k, True) for k in clusters}}

    def score(k, d, pol):
        L = [s for s in clusters[k] if info[s]["published_at"].date() <= d]
        if pol == "Q1_posts":
            return len(L)
        if pol == "Q2_authors":
            return len({info[s]["author_id"] for s in L})
        if pol == "Q3_dedup":
            return len({th.get(s) or s for s in L})
        if pol == "Q4_capped":
            return sum(min(v, 3) for v in Counter(info[s]["author_id"] for s in L).values())
        return sum(info[s]["author_id"] not in feeds for s in L)

    pub = {e[0]: e[2] for e in regn}
    REPS = 50

    def simulate(pol, b, use_fuzzy):
        """Mean over REPS random tie-breaking orders (pool sorted first, so order does not depend on hashing)."""
        r = np.random.default_rng([20261106, POL.index(pol), b, int(use_fuzzy)])
        per_day, totals, delays = Counter(), [], []
        for _ in range(REPS):
            reviewed, got = set(), {}
            for d in DAYS[7:]:
                pool = sorted(k for k in clusters if first[k] <= d and k not in reviewed)
                jit = r.random(len(pool)) * 1e-6
                sc = {k: score(k, d, pol) + jit[i] for i, k in enumerate(pool)}
                for k in sorted(pool, key=lambda k: -sc[k])[:b]:
                    reviewed.add(k)
                    if d in EVAL:
                        for e in sorted(M[use_fuzzy][k]):
                            got.setdefault(e, d)
            totals.append(len(got))
            delays += [(got[e] - pub[e]).days for e in got]
            for e in got:
                per_day[pub[e]] += 1 / REPS
        return {"claims": round(float(np.mean(totals)), 2), "claims_95_ties": [float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))],
                "median_delay_days": float(np.median(delays)) if delays else None, "per_pubday": per_day}

    out = {"n_posts_with_victim": sum(bool(v["victims"]) for v in vic), "n_clusters": len(clusters),
           "n_registry": len(regn), "registry_countries": dict(Counter(e[3] for e in regn)),
           "feeds_hist_accounts": len(feeds), "results": {}}
    for use_fuzzy in (False, True):
        tag = "fuzzy" if use_fuzzy else "exact"
        ever = set().union(*M[use_fuzzy].values()) if M[use_fuzzy] else set()
        out[f"claims_with_any_cluster_{tag}"] = len(ever)
        out[f"claims_with_cluster_in_eval_{tag}"] = len(set().union(*[M[use_fuzzy][k] for k in clusters if first[k] in EVAL] or [set()]))
        for b in (1, 5, 10):
            out["results"][f"{tag}_b{b}"] = {p: simulate(p, b, use_fuzzy) for p in POL}
            R = out["results"][f"{tag}_b{b}"]
            # paired comparison against Q1 on expected per-day counts, bootstrap over registry publication days
            days_pub = sorted(set().union(*[set(R[p]["per_pubday"]) for p in POL]))
            for p in POL:
                if p != "Q1_posts":
                    diff = np.array([R[p]["per_pubday"].get(d, 0) - R["Q1_posts"]["per_pubday"].get(d, 0) for d in days_pub])
                    bs = [diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(2000)]
                    R[p]["vs_Q1"] = round(float(diff.sum()), 2)
                    R[p]["vs_Q1_ci"] = [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
            for p in POL:
                R[p]["per_pubday"] = {str(d): round(v, 3) for d, v in R[p]["per_pubday"].items()}
            print(tag, b, {p: (v["claims"], v.get("vs_Q1"), v.get("vs_Q1_ci")) for p, v in R.items()}, flush=True)

    # E-E on E-B (PREREG E-E outcome "E-B registry coverage"): remove exactly B mention posts, where B is the number
    # of history-rule feed posts among the candidate posts, then rank by Q1 (posts) with b = 5, exact matching.
    allposts = sorted({s for L in clusters.values() for s in L})
    B = sum(info[s]["author_id"] in feeds for s in allposts)

    def q1_after(keep, reps=20):
        r = np.random.default_rng([20261107, len(keep)])
        kept = {k: [s for s in L if s in keep] for k, L in clusters.items()}
        fk = {k: min(info[s]["published_at"].date() for s in v) for k, v in kept.items() if v}
        tot = []
        for _ in range(reps):
            reviewed, got = set(), set()
            for d in DAYS[7:]:
                pool = sorted(k for k in fk if fk[k] <= d and k not in reviewed)
                jit = r.random(len(pool)) * 1e-6
                sc = {k: sum(info[s]["published_at"].date() <= d for s in kept[k]) + jit[i] for i, k in enumerate(pool)}
                for k in sorted(pool, key=lambda k: -sc[k])[:5]:
                    reviewed.add(k)
                    if d in EVAL:
                        got |= M[False][k]
            tot.append(len(got))
        return round(float(np.mean(tot)), 2)
    cnt = Counter(info[s]["author_id"] for s in allposts)
    top, n = set(), 0
    for a_, _ in cnt.most_common():
        for s in allposts:
            if info[s]["author_id"] == a_ and n < B:
                top.add(s)
                n += 1
        if n >= B:
            break
    rand = [q1_after(set(allposts) - set(rng.choice(allposts, size=B, replace=False)), reps=5) for _ in range(40)]
    out["E_E_on_registry_b5"] = {"B": B, "n_posts": len(allposts), "none": q1_after(set(allposts)),
                                 "feed_removal": q1_after({s for s in allposts if info[s]["author_id"] not in feeds}),
                                 "top_output": q1_after(set(allposts) - top),
                                 "random_mean": float(np.mean(rand)), "random_95": [float(np.percentile(rand, 2.5)), float(np.percentile(rand, 97.5))]}
    print("E-E on registry", out["E_E_on_registry_b5"], flush=True)
    # fuzzy-only matches for blind human adjudication (names only; no policy information)
    rows = sorted({(k, e[1], e[0]) for k in clusters for e in regn
                   if e[0] in M[True][k] and e[0] not in M[False][k]})
    with open("outputs/interaction_experiment/www_eval/registry_fuzzy_review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_name", "registry_name", "registry_event_id", "same_organisation(y/n)"])
        w.writerows([[a, b, c, ""] for a, b, c in rows])
    out["n_fuzzy_only_pairs"] = len(rows)
    p = Path("outputs/interaction_experiment/www_eval/registry.json")
    p.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", p, "fuzzy pairs", len(rows))


if __name__ == "__main__":
    main()
