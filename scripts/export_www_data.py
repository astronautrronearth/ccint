"""Export seed- and edge-level tables for the WWW 2027 short paper (run 475, manifest v2).

    CCINT_EXPORT_SALT=... python scripts/export_www_data.py
    → outputs/www_export/seeds.csv, edges.csv, meta.json

Privacy: every account id is replaced by sha256(salt || did)[:16]. Only accounts profiled
(actor_v1) as news, organisations or intel feeds keep a public handle. No post text, no URIs.
Keep the salt out of the repo; reuse it for any later export so hashes line up.

Seed selection and edge selection follow interaction_descriptive.load(): latest non-failed
outcome per seed, edges only from the run the seed's thread outcome came from.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("desc", "scripts/interaction_descriptive.py")
desc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desc)
from ccint import db  # noqa: E402
from ccint.analytics.broadcast import SQL as BSQL, detect_broadcast_authors  # noqa: E402
from ccint.interaction.collect import EP_QUOTES, EP_REPOSTS, EP_THREAD  # noqa: E402

SALT = os.environ.get("CCINT_EXPORT_SALT")
if not SALT:
    sys.exit("set CCINT_EXPORT_SALT")
OUT = Path("outputs/www_export")
PUBLIC_TYPES = {"news_with_reach", "news_syndicated", "org_reaching", "org_unheard",
                "intel_feed_raw", "intel_feed_amplified"}
OK = {"success", "partial"}


def h(did: str | None) -> str:
    return hashlib.sha256((SALT + (did or "")).encode()).hexdigest()[:16] if did else ""


def main() -> None:
    m, runs, seeds, oc, edges, info, tc = desc.load()
    with db.connect() as c:
        beh = detect_broadcast_authors(c, label_version="llm_v2d").author_ids
        vol = {r["author_id"] for r in c.execute(BSQL, {"label_version": "llm_v2d", "since": None,
               "until": None, "min_lag_hours": 24}) if r["n_posts"] >= 5 and r["n_days"] >= 4}
        kev = {r["post_id"]: (r["n"], r["k"]) for r in c.execute("""
            SELECT pc.post_id, count(DISTINCT pc.cve_id) n, bool_or(coalesce(cr.known_exploited, false)) k
            FROM post_cves pc LEFT JOIN cve_records cr USING (cve_id)
            WHERE pc.post_id = ANY(%s) GROUP BY 1""", (list(seeds),))}

    def status(s):
        v = oc.get(s)
        if not v:
            return "missing"
        t, q = v[EP_THREAD]["outcome"], v[EP_QUOTES]["outcome"]
        if t in OK and q in OK:
            return "observed"
        return "unavailable" if "unavailable" in (t, q) else "failed"

    cnt = defaultdict(Counter)
    erows = []
    for e in edges:
        s = e["seed_post_id"]
        self_ = e["source_actor_id"] == info[s]["author_id"]
        k = e["edge_type"] if e["edge_type"] != "reply" else f"reply_d{e['depth']}"
        cnt[s][k] += 1
        cnt[s][k + "_self"] += self_
        erows.append({"seed_post_id": s, "edge_type": e["edge_type"], "depth": e["depth"],
                      "source_actor_hash": h(e["source_actor_id"]), "is_seed_author": int(self_),
                      "event_created_at": e["event_created_at"].isoformat() if e["event_created_at"] else ""})

    srows = []
    for s, r in sorted(seeds.items()):
        i = info[s]
        a = i["author_id"]
        t = tc.get(s) or {}
        n_cve, k = kev.get(s, (0, None))
        srows.append({
            "post_id": s, "author_hash": h(a),
            "author_handle_public": i["author_handle"] if i["actor_type"] in PUBLIC_TYPES else "",
            "published_at": i["published_at"].isoformat(), "topic_v2": i["topic"],
            "actor_type_v1": i["actor_type"], "actor_function_v1": i["actor_function"],
            "feed_behavioral_v1": int(a in beh), "feed_volume_only": int(a in vol),
            "in_main": int(r["in_main"] == "true"), "in_legacy": int(r["in_legacy"] == "true"),
            "in_rules": int(r["in_rules"] == "true"), "status": status(s),
            "thread_outcome": (oc.get(s, {}).get(EP_THREAD) or {}).get("outcome", ""),
            "quotes_outcome": (oc.get(s, {}).get(EP_QUOTES) or {}).get("outcome", ""),
            "reposts_outcome": (oc.get(s, {}).get(EP_REPOSTS) or {}).get("outcome", ""),
            "n_cve": n_cve, "has_kev_cve": "" if k is None else int(k),
            "ingest_reply_count": i["rc0"] if i["rc0"] is not None else "",
            "ingest_quote_count": i["qc0"] if i["qc0"] is not None else "",
            "thread_reply_count": t.get("rc", ""), "thread_quote_count": t.get("qc", ""),
            "thread_repost_count": t.get("rpc", ""),
            **{f"n_{k}": cnt[s][k] for k in ("reply_d1", "reply_d2", "quote", "repost")},
            **{f"n_{k}_self": cnt[s][k + "_self"] for k in ("reply_d1", "reply_d2", "quote")},
        })

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("seeds.csv", srows), ("edges.csv", erows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / "meta.json").write_text(json.dumps({
        "manifest_sha256": m.sha256, "full_runs": runs, "derivation_version": desc.DERIVATION_VERSION,
        "n_seeds": len(srows), "n_edges": len(erows), "hash": "sha256(salt||did)[:16]",
        "public_handle_types": sorted(PUBLIC_TYPES),
        "feed_behavioral_v1": "broadcast.detect_broadcast_authors(label_version=llm_v2d), defaults",
        "feed_volume_only": ">=5 relevant llm_v2d posts on >=4 days (outcome-free)",
        "observed": "thread and quote endpoints success/partial",
        "edge_selection": "edges from the run whose thread outcome is used (interaction_descriptive.load)",
    }, indent=1))
    print(f"wrote {OUT / 'seeds.csv'} and {OUT / 'edges.csv'} ({len(srows)} seeds, {len(erows)} edges)")


if __name__ == "__main__":
    main()
