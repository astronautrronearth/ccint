"""Registry anchor for the WWW short paper: who posts about externally registered ransomware events.

    python scripts/www_registry.py   → outputs/interaction_experiment/www_gonogo/registry.json

Input: reports/oracle_events.json (entity-directed Bluesky search for the 27 Canadian ransomware.live
victims, 2026-08-22 → 2026-09-22; see METHOD_DERIVATION.md §0.2(a)). Read-only. Recomputes the funnel's
last stage ("event has a post by a non-feed account") under the three feed definitions used in the
paper, instead of the fig_funnel heuristic, so the definitions are not mixed. No handles in the output.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from ccint import db  # noqa: E402
from ccint.analytics.broadcast import SQL as BSQL, detect_broadcast_authors  # noqa: E402

ev = json.loads(Path("reports/oracle_events.json").read_text())
did = lambda uri: uri.split("/")[2]
with db.connect() as c:
    beh = detect_broadcast_authors(c, label_version="llm_v2d").author_ids
    vol = {r["author_id"] for r in c.execute(BSQL, {"label_version": "llm_v2d", "since": None, "until": None,
           "min_lag_hours": 24}) if r["n_posts"] >= 5 and r["n_days"] >= 4}
    atype = {r["author_id"]: r["actor_type"] for r in c.execute(
        "SELECT author_id, actor_type FROM author_profiles WHERE profile_version = 'actor_v1'")}
intel = {a for a, t in atype.items() if t.startswith("intel_feed")}

cyber = [e for e in ev if any(h["is_cyber"] for h in e["hits"])]
hits = [h for e in cyber for h in e["hits"] if h["is_cyber"]]
accts = {did(h["uri"]) for h in hits}
per_acct = defaultdict(set)
for e in cyber:
    for h in e["hits"]:
        if h["is_cyber"]:
            per_acct[did(h["uri"])].add(e["event_id"])
out = {
    "events": len(ev), "events_with_cyber_post": len(cyber),
    "events_in_corpus": sum(any(h["is_cyber"] and h["in_corpus"] for h in e["hits"]) for e in cyber),
    "cyber_hits": len(hits), "accounts": len(accts), "max_likes_any_hit": max(h["likes"] for h in hits),
    "hits_with_zero_likes": sum(h["likes"] == 0 for h in hits),
    "top3_accounts_event_coverage": sorted((len(v) for v in per_acct.values()), reverse=True)[:3],
    "by_feed_definition": {},
}
for name, F in (("behavioral_v1", beh), ("volume_only", vol), ("actor_v1_intel_feed", intel)):
    nf = [e for e in cyber if any(h["is_cyber"] and did(h["uri"]) not in F for h in e["hits"])]
    out["by_feed_definition"][name] = {
        "accounts_flagged": len(accts & F),
        "events_with_nonfeed_post": len(nf),
        "nonfeed_accounts": len({did(h["uri"]) for e in cyber for h in e["hits"]
                                 if h["is_cyber"] and did(h["uri"]) not in F}),
        "nonfeed_actor_types": sorted({atype.get(did(h["uri"]), "no_profile") for e in nf for h in e["hits"]
                                       if h["is_cyber"] and did(h["uri"]) not in F}),
    }
p = Path("outputs/interaction_experiment/www_gonogo/registry.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
