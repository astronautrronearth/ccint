"""Case-study and timeline data for the WWW figures (2026-11-06). Same population and human feed labels as
www_producer.py; no handles or author ids in the output.

    python scripts/www_case_data.py --audit outputs/www_audit/scoring → outputs/interaction_experiment/www_gonogo/cases.json

timeline: per topic and day, posts by feeds and non-feeds, cross-user participants, and E1 burst flags (k=3).
feed_case: the largest ransomware feed account: its post count, cross-user edges it received, and three of
           its posts (text with the victim and group names replaced by slots, likes, reposts).
self_case: candidate vulnerability seeds whose author continues the post at least twice within 10 minutes;
           seed and continuation texts (truncated) and delays, for manual selection and paraphrase.
registry:  per listed Canadian ransomware victim, cyber posts, accounts and the largest like count.
"""
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
from ccint import db  # noqa: E402

spec = importlib.util.spec_from_file_location("prod", "scripts/www_producer.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

DAYS = [date(2026, 8, 22) + timedelta(d) for d in range((date(2026, 9, 23) - date(2026, 8, 22)).days + 1)]


def burst_flags(x, k=3.0, floor_min=5):
    out = [False] * len(x)
    for i in range(7, len(x)):
        base = np.array(x[i - 7:i], float)
        med = np.median(base)
        mad = max(np.median(np.abs(base - med)), 1.0)
        out[i] = bool(x[i] > med + k * mad and x[i] >= floor_min)
    return out


def find_text(obj, uri):
    if isinstance(obj, str) and obj[:1] in "[{":
        try:
            obj = json.loads(obj)
        except ValueError:
            return None
    if isinstance(obj, dict):
        if obj.get("uri") == uri and isinstance(obj.get("record"), dict):
            return obj["record"].get("text")
        for v in obj.values():
            r = find_text(v, uri)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_text(v, uri)
            if r:
                return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    a = ap.parse_args()
    m, runs, seeds, oc, edges, info, tc = P.S.desc.load()
    pops, partial, ps, info2, beh, vol = P.S.load()
    M = pops["main"]
    HF = P.human_feeds(a.audit)
    author = {s: info[s]["author_id"] for s in M}
    topic = {s: info[s]["topic"] for s in M}
    pub = {s: info[s]["published_at"].date() for s in M}
    part, cross_n = defaultdict(set), Counter()
    for e in edges:
        s = e["seed_post_id"]
        if s in author and e["edge_type"] in ("reply", "quote") and e["source_actor_id"] != author[s]:
            part[s].add(e["source_actor_id"])
            cross_n[s] += 1

    # timeline
    tl = {}
    for t in ("ransomware", "data_breach", "vulnerability_cve"):
        L = [s for s in M if topic[s] == t]
        byday = defaultdict(list)
        for s in L:
            byday[pub[s]].append(s)
        posts = [len(byday[d]) for d in DAYS]
        tl[t] = {"days": [d.isoformat() for d in DAYS], "posts": posts,
                 "feed_posts": [sum(author[s] in HF for s in byday[d]) for d in DAYS],
                 "participants": [len(set().union(*[part[s] for s in byday[d]])) if byday[d] else 0 for d in DAYS],
                 "burst": burst_flags(posts)}
        c = tl[t]["participants"]
        tl[t]["participant_burst"] = burst_flags(c, floor_min=1)

    # feed case
    RW = [s for s in M if topic[s] == "ransomware"]
    top_author, n_top = Counter(author[s] for s in RW).most_common(1)[0]
    mine = sorted([s for s in RW if author[s] == top_author], key=lambda s: info[s]["published_at"])
    with db.connect() as c:
        rows = {r["post_id"]: r for r in c.execute(
            "SELECT post_id, text, like_count, repost_count, reply_count, published_at FROM posts WHERE post_id = ANY(%s)",
            (mine,))}
        ents = defaultdict(list)
        for r in c.execute("SELECT post_id, entity_text, entity_type FROM post_entities WHERE post_id = ANY(%s)", (mine,)):
            ents[r["post_id"]].append((r["entity_text"], r["entity_type"]))
    ex = []
    for s in mine[:: max(1, len(mine) // 3)][:3]:
        r = rows[s]
        ex.append({"published_at": r["published_at"].isoformat(), "text": r["text"], "entities": ents[s],
                   "likes": r["like_count"], "reposts": r["repost_count"], "replies": r["reply_count"]})
    feed_case = {"is_human_feed": top_author in HF, "posts_in_topic": n_top,
                 "posts_in_topic_with_any_cross": sum(cross_n[s] > 0 for s in mine),
                 "cross_edges_received": sum(cross_n[s] for s in mine),
                 "likes_total": sum(rows[s]["like_count"] or 0 for s in mine),
                 "examples": ex}

    # self-thread candidates
    seeds_v = {s for s in M if topic[s] in ("vulnerability_cve", "unassigned", "malware_infostealer")}
    cont = defaultdict(list)
    for e in edges:
        s = e["seed_post_id"]
        if s in seeds_v and e["edge_type"] == "reply" and e["depth"] == 1 and e["source_actor_id"] == author[s]:
            d = (e["event_created_at"] - info[s]["published_at"]).total_seconds() / 60
            cont[s].append((d, e["source_post_uri"]))
    cand = [s for s, v in cont.items() if len(v) >= 2 and min(d for d, _ in v) <= 10 and author[s] not in HF]
    cand.sort(key=lambda s: (-len(part[s]), -len(cont[s])))
    self_case = []
    with db.connect() as c:
        for s in cand[:10]:
            seed = c.execute("SELECT text FROM posts WHERE post_id = %s", (s,)).fetchone()
            snaps = c.execute("SELECT raw_payload FROM interaction_snapshots WHERE seed_post_id = %s AND endpoint = 'app.bsky.feed.getPostThread'",
                              (s,)).fetchall()
            texts = []
            for d, uri in sorted(cont[s]):
                t = None
                for sn in snaps:
                    t = find_text(sn["raw_payload"], uri)
                    if t:
                        break
                texts.append({"delay_min": round(d, 1), "text": (t or "")[:220]})
            self_case.append({"topic": topic[s], "seed_text": (seed["text"] or "")[:260], "continuations": texts,
                              "cross_participants": len(part[s]), "cross_edges": cross_n[s]})

    # registry strip
    ev = json.loads(Path("reports/oracle_events.json").read_text())
    reg = []
    for e in ev:
        h = [x for x in e["hits"] if x["is_cyber"]]
        reg.append({"posts": len(h), "accounts": len({x["handle"] for x in h}),
                    "by_account": sorted(Counter(x["handle"] for x in h).values(), reverse=True),
                    "top_handles_rank": [x["handle"] for x in h],
                    "max_likes": max([x.get("likes") or 0 for x in h], default=0)})
    # anonymise: replace handles by their overall rank in registry coverage
    cov = Counter(hd for r in reg for hd in set(r["top_handles_rank"]))
    rank = {hd: i for i, (hd, _) in enumerate(cov.most_common())}
    for r in reg:
        r["accounts_by_rank"] = sorted({rank[hd] for hd in r.pop("top_handles_rank")})
    out = {"timeline": tl, "feed_case": feed_case, "self_case": self_case, "registry": reg}
    p = Path("outputs/interaction_experiment/www_gonogo/cases.json")
    p.write_text(json.dumps(out, default=str, ensure_ascii=False, indent=1))
    print(json.dumps({k: (v if k != "timeline" else {t: sum(x["burst"]) for t, x in v.items()}) for k, v in out.items()},
                     default=str, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
