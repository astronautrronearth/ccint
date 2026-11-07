"""Crossover replication in the fresh window (PREREG: www2027/results/frozen_eval/PREREG_CROSSOVER.md).

    python scripts/www_crossover_fresh.py manifest                 # ransomware candidates -> www_crossover manifest
    python scripts/www_crossover_fresh.py collect --shard i --nshard n
    python scripts/www_crossover_fresh.py extract --base-url URL   # victims (same prompt/model as E-B)
    python scripts/www_crossover_fresh.py eval                     # -> outputs/interaction_experiment/www_eval/crossover_fresh.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402

PREFILTER = (r"(ransom|leak|extort|lockbit|akira|qilin|blackcat|alphv|clop|cl0p|play |medusa|rhysida|8base|bianlian|"
             r"hunters|inc ransom|safepay|dragonforce|data breach|victim|hack|breach|stolen data|darkweb|dark web)")
POP = """FROM posts p JOIN post_labels l ON l.post_id = p.post_id AND l.label_version = 'rules_v2' AND l.is_cyber
         WHERE p.source_key = 'bluesky' AND p.lang LIKE 'en%%' AND length(coalesce(p.text, '')) > 0
           AND p.published_at >= '2026-10-21' AND p.published_at < '2026-11-05'"""
D0 = date(2026, 10, 21)
DAYS = [D0 + timedelta(i) for i in range(15)]
HIST, EVAL = DAYS[:3], DAYS[7:]
MAN = Path("manifests/interaction_experiment/www_crossover")
VICT = Path("outputs/interaction_experiment/www_eval/victims_fresh.jsonl")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def candidate_pool_ids():
    """Posts to run topic_v2s on: the fresh cybersecurity population matching the prefilter."""
    with db.connect() as c:
        return [r["post_id"] for r in c.execute(f"SELECT p.post_id {POP} AND p.text ~* %s ORDER BY p.post_id", (PREFILTER,))]


def manifest():
    csv_path, meta_path = MAN / "seed_manifest.csv", MAN / "seed_manifest.meta.json"
    if csv_path.exists():
        sys.exit(f"{csv_path} exists; refusing to overwrite")
    ids = candidate_pool_ids()
    with db.connect() as c:
        rows = c.execute("""SELECT p.post_id, p.raw_payload->>'uri' AS post_uri, p.author_id, p.published_at, p.source_key,
                                   t.topic_key
                            FROM posts p LEFT JOIN post_labels t ON t.post_id = p.post_id AND t.label_version = 'topic_v2s'
                            WHERE p.post_id = ANY(%s) ORDER BY p.post_id""", (ids,)).fetchall()
    unl = sum(r["topic_key"] is None for r in rows)
    if unl:
        sys.exit(f"{unl} prefiltered posts still lack topic_v2s; label first")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["post_id", "post_uri", "author_id", "created_at", "source", "in_ransom"])
    n = 0
    for r in rows:
        if r["topic_key"] == "ransomware" and r["post_uri"]:
            w.writerow([r["post_id"], r["post_uri"], r["author_id"], r["published_at"].isoformat(), r["source_key"], "true"])
            n += 1
    data = buf.getvalue().encode()
    MAN.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(data)
    meta = {"manifest_version": "www_crossover", "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "definition": "PREREG_CROSSOVER: fresh-window rules_v2 is_cyber en posts, prefilter regex, topic_v2s = ransomware",
            "prefiltered": len(rows), "n": n, "sha256": hashlib.sha256(data).hexdigest()}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def collect(shard, nshard):
    from ccint.interaction import manifest as mf
    from ccint.interaction.collect import git_commit, run
    from ccint.interaction.derive import derive_run
    m = mf.load("www_crossover")
    seeds = [s for s in m.population("in_ransom") if int(s["post_id"]) % nshard == shard]
    with db.connect() as c:
        done = {r["seed_post_id"] for r in c.execute("""
            SELECT DISTINCT o.seed_post_id FROM interaction_seed_outcomes o
            JOIN neighborhood_runs n ON n.run_id = o.collection_run_id
            WHERE n.purpose = 'crossover' AND n.manifest_sha256 = %s""", (m.sha256,))}
    todo = [s for s in seeds if int(s["post_id"]) not in done]
    print(f"shard {shard}/{nshard}: seeds {len(seeds)} to collect {len(todo)}", flush=True)
    if not todo:
        return
    rid = run(m, todo, purpose="crossover")
    with db.transaction() as conn:
        n = derive_run(conn, rid, git_commit())
    print(f"run {rid}: {len(todo)} seeds, {n} edges", flush=True)


def extract(base_url):
    import httpx
    from ccint.interaction import manifest as mf
    EV = _mod("ev", "scripts/www_extract_victims.py")
    ids = [int(r["post_id"]) for r in mf.load("www_crossover").population("in_ransom")]
    with db.connect() as c:
        rows = c.execute("SELECT post_id, text FROM posts WHERE post_id = ANY(%s) ORDER BY post_id", (ids,)).fetchall()
    with httpx.Client() as client, ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda r: (r["post_id"], EV.ask(client, base_url, r["text"] or "")), rows))
    VICT.parent.mkdir(parents=True, exist_ok=True)
    with VICT.open("w") as f:
        for pid, v in res:
            f.write(json.dumps({"post_id": pid, "victims": v}) + "\n")
    print("posts", len(res), "failed", sum(v is None for _, v in res), "with victim", sum(bool(v) for _, v in res))


def evaluate():
    from ccint.interaction import manifest as mf
    REG = _mod("reg", "scripts/www_eval_registry.py")
    SC, MON = REG.SC, _mod("mon", "scripts/www_eval_monitor.py")
    m = mf.load("www_crossover")
    ids = [int(r["post_id"]) for r in m.population("in_ransom")]
    with db.connect() as c:
        runs = [r["run_id"] for r in c.execute(
            "SELECT run_id FROM neighborhood_runs WHERE purpose = 'crossover' AND manifest_sha256 = %s", (m.sha256,))]
        # feed rule (E-D): >= 5 posts on each of the 3 history days, on the whole fresh cyber population
        hp = defaultdict(Counter)
        for r in c.execute(f"SELECT p.author_id, p.published_at {POP} AND p.published_at < %s", (str(DAYS[3]),)):
            hp[r["author_id"]][r["published_at"].date()] += 1
        reg = c.execute("""SELECT event_id, title, country, published_at_utc FROM external_events
                           WHERE source = 'ransomware.live' AND published_at_utc >= '2026-10-07'
                             AND published_at_utc < '2026-11-09'""").fetchall()
    feeds = {a for a, d in hp.items() if sum(d.values()) >= 5 and len(d) == 3}
    obs, edges, info = SC.load(runs, ids)
    obs = set(obs)
    vic = [json.loads(l) for l in open(VICT)]
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
    pub = {e[0]: e[2] for e in regn}
    eb = defaultdict(list)
    for e in edges:
        s = e["seed_post_id"]
        if e["edge_type"] in ("reply", "quote") and s in info and e["event_created_at"] is not None \
                and (e["event_created_at"] - info[s]["published_at"]).total_seconds() <= 2 * 86400:
            eb[s].append(e)

    def score(L, d, pol):
        L = [s for s in L if info[s]["published_at"].date() <= d]
        c = MON.cut(d)
        if pol == "posts":
            return len(L)
        if pol == "nofeed":
            return sum(info[s]["author_id"] not in feeds for s in L)
        if pol == "edges":
            return sum(e["event_created_at"] < c for s in L for e in eb[s])
        if pol == "responders":
            return len({e["source_actor_id"] for s in L for e in eb[s]
                        if e["event_created_at"] < c and e["source_actor_id"] != info[s]["author_id"]})
        raise ValueError(pol)

    def simulate(cl, pol, b, reps=50, seed=0):
        fk = {k: min(info[s]["published_at"].date() for s in L) for k, L in cl.items() if L}
        r = np.random.default_rng([20261107, seed, b, len(pol)])
        per, tot = Counter(), []
        for _ in range(reps):
            reviewed, got = set(), set()
            for d in DAYS[3:]:
                pool = sorted(k for k in fk if fk[k] <= d and k not in reviewed)
                jit = r.random(len(pool)) * 1e-6
                sc = {k: score(cl[k], d, pol) + jit[i] for i, k in enumerate(pool)}
                for k in sorted(pool, key=lambda k: -sc[k])[:b]:
                    reviewed.add(k)
                    if d in EVAL:
                        got |= M[k]
            tot.append(len(got))
            for e in got:
                per[pub[e]] += 1 / reps
        return {"claims": round(float(np.mean(tot)), 2), "ties_95": [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))],
                "per_pubday": per}

    rng = np.random.default_rng(20261107)
    out = {"n_candidates": len(ids), "n_observed": len(obs), "n_with_victim": sum(bool(v["victims"]) for v in vic),
           "n_clusters": len(clusters), "n_registry": len(regn), "feeds": len(feeds),
           "attainable_eval": len(set().union(*[M[k] for k in clusters if first[k] in set(EVAL)] or [set()])),
           "attainable_any": len(set().union(*M.values()) if M else set())}
    cand = {s for L in clusters.values() for s in L}
    out["feed_share_candidates"] = round(sum(info[s]["author_id"] in feeds for s in cand) / max(1, len(cand)), 3)
    fe = [len(eb[s]) for s in cand if info[s]["author_id"] in feeds and s in obs]
    ne = [len(eb[s]) for s in cand if info[s]["author_id"] not in feeds and s in obs]
    out["edges_per_post"] = {"feed": round(float(np.mean(fe)), 3) if fe else None, "nonfeed": round(float(np.mean(ne)), 3) if ne else None}
    for b in (1, 5, 10):
        R = {p: simulate(clusters, p, b) for p in ("posts", "edges", "responders", "nofeed")}
        days = sorted(set().union(*[set(v["per_pubday"]) for v in R.values()]))
        for p in ("edges", "responders", "nofeed"):
            diff = np.array([R[p]["per_pubday"].get(d, 0) - R["posts"]["per_pubday"].get(d, 0) for d in days])
            bs = [diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(2000)] if len(diff) else [0]
            R[p]["vs_posts"] = round(float(diff.sum()), 2)
            R[p]["vs_posts_ci"] = [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
        for v in R.values():
            v["per_pubday"] = {str(d): round(x, 3) for d, x in v["per_pubday"].items()}
        out[f"b{b}"] = R
        print("b", b, {p: (v["claims"], v.get("vs_posts"), v.get("vs_posts_ci")) for p, v in R.items()}, flush=True)
    # equal-budget removal (as E-E), b = 5, ranked by posts
    allp = sorted(cand)
    B = sum(info[s]["author_id"] in feeds for s in allp)
    cnt = Counter(info[s]["author_id"] for s in allp)
    top, n = set(), 0
    for a_, _ in cnt.most_common():
        for s in allp:
            if info[s]["author_id"] == a_ and n < B:
                top.add(s)
                n += 1
        if n >= B:
            break

    def keep(K):
        return {k: [s for s in L if s in K] for k, L in clusters.items()}
    rand = [simulate(keep(set(allp) - set(rng.choice(allp, size=B, replace=False))), "posts", 5, reps=5, seed=i)["claims"]
            for i in range(40)] if B else [None]
    out["E_E_b5"] = {"B": B, "n_posts": len(allp), "none": out["b5"]["posts"]["claims"],
                     "feed_removal": simulate(keep({s for s in allp if info[s]["author_id"] not in feeds}), "posts", 5)["claims"],
                     "top_output": simulate(keep(set(allp) - top), "posts", 5)["claims"],
                     "random_mean": float(np.mean(rand)) if B else None,
                     "random_95": [float(np.percentile(rand, 2.5)), float(np.percentile(rand, 97.5))] if B else None}
    print("E-E", out["E_E_b5"], flush=True)
    r5 = out["b5"]
    lo = r5["edges"]["vs_posts_ci"][1]
    out["reading"] = ("inconclusive (fewer than 10 attainable claims)" if out["attainable_eval"] < 10 else
                      "replicated" if lo < 0 else "not replicated" if r5["edges"]["vs_posts"] >= 0 else "inconclusive")
    print("reading", out["reading"], out["attainable_eval"], flush=True)
    Path("outputs/interaction_experiment/www_eval/crossover_fresh.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "collect", "extract", "eval"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--base-url")
    a = ap.parse_args()
    {"manifest": manifest, "collect": lambda: collect(a.shard, a.nshard), "extract": lambda: extract(a.base_url),
     "eval": evaluate}[a.cmd]()
