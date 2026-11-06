"""Data for the teaser figure (2026-11-06): per topic, posts by author rank (anonymous), feed flag per rank,
accounts, effective producers and cross-user participants. Same population and human feed labels as
www_producer.py.

    python scripts/www_teaser_data.py --audit outputs/www_audit/scoring → outputs/interaction_experiment/www_gonogo/teaser.json
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = ["src", "scripts"]
spec = importlib.util.spec_from_file_location("prod", "scripts/www_producer.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    a = ap.parse_args()
    m, runs, seeds, oc, edges, info, tc = P.S.desc.load()
    pops, partial, ps, info2, beh, vol = P.S.load()
    M = pops["main"]
    HF = P.human_feeds(a.audit)
    author = {s: info[s]["author_id"] for s in M}
    part = defaultdict(set)
    for e in edges:
        s = e["seed_post_id"]
        if s in author and e["edge_type"] in ("reply", "quote") and e["source_actor_id"] != author[s]:
            part[s].add(e["source_actor_id"])
    out = {}
    for t in sorted({info[s]["topic"] for s in M}):
        L = [s for s in M if info[s]["topic"] == t]
        c = Counter(author[s] for s in L).most_common()
        n = len(L)
        out[t] = {"posts": n, "accounts": len(c), "N_eff": round(1 / sum((k / n) ** 2 for _, k in c), 2),
                  "cross_participants": len(set().union(*[part[s] for s in L])),
                  "by_rank": [k for _, k in c], "feed_by_rank": [int(x in HF) for x, _ in c]}
    p = Path("outputs/interaction_experiment/www_gonogo/teaser.json")
    p.write_text(json.dumps(out))
    print({t: (v["posts"], v["accounts"], v["N_eff"], v["cross_participants"], v["by_rank"][:5]) for t, v in out.items()})


if __name__ == "__main__":
    main()
