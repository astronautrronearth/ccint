"""Seed manifest for the dose-response domains crypto and AI (2026-11-06; PREREG_DOSE in www2027/results/frozen_eval).

    python scripts/build_seed_manifest_contrast.py   → manifests/interaction_experiment/www_contrast/

Two populations, 2026-10-21..2026-11-04 UTC, lang en*, non-empty text: NBA (source bluesky_nba, not rules_v2.is_cyber)
and concurrent cybersecurity (source bluesky, rules_v2.is_cyber). Random sample of 10,000 each (seed 20261106).
[MUST] read-only on the database; refuses to overwrite an existing manifest.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ccint import db

OUT = Path("manifests/interaction_experiment/www_contrast2")
N_EACH = 10000
SQL = """
SELECT p.post_id, p.raw_payload->>'uri' AS post_uri, p.author_id, p.published_at AS created_at, p.source_key,
       coalesce(l.is_cyber, false) AS is_cyber
FROM posts p LEFT JOIN post_labels l ON l.post_id = p.post_id AND l.label_version = 'rules_v2'
WHERE p.source_key IN ('bluesky_crypto', 'bluesky_ai') AND p.lang LIKE 'en%%'
  AND length(coalesce(p.text, '')) > 0
  AND p.published_at >= '2026-10-21' AND p.published_at < '2026-11-05'
ORDER BY p.post_id"""


def main() -> None:
    csv_path, meta_path = OUT / "seed_manifest.csv", OUT / "seed_manifest.meta.json"
    if csv_path.exists():
        sys.exit(f"{csv_path} exists; refusing to overwrite")
    with db.connect() as c:
        rows = c.execute(SQL).fetchall()
    import random
    nba = [r for r in rows if r["source_key"] == "bluesky_crypto" and not r["is_cyber"] and r["post_uri"]]
    cyb = [r for r in rows if r["source_key"] == "bluesky_ai" and not r["is_cyber"] and r["post_uri"]]
    rnd = random.Random(20261106)
    pick = {r["post_id"]: "crypto" for r in rnd.sample(nba, min(N_EACH, len(nba)))}
    pick.update({r["post_id"]: "ai" for r in rnd.sample(cyb, min(N_EACH, len(cyb)))})
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["post_id", "post_uri", "author_id", "created_at", "source", "in_crypto", "in_ai"])
    for r in rows:
        if r["post_id"] in pick:
            d = pick[r["post_id"]]
            w.writerow([r["post_id"], r["post_uri"], r["author_id"], r["created_at"].isoformat(), r["source_key"],
                        str(d == "crypto").lower(), str(d == "ai").lower()])
    data = buf.getvalue().encode()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(data)
    meta = {"manifest_version": "www_contrast2", "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "definition": "random 10,000 each of crypto (bluesky_crypto) and AI (bluesky_ai), not rules_v2.is_cyber; lang en*, text non-empty, 2026-10-21 <= published_at < 2026-11-05; seed 20261106", "population_sizes": {"crypto": len(nba), "ai": len(cyb)},
            "n": data.count(b"\n") - 1, "sha256": hashlib.sha256(data).hexdigest()}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
