"""Seed manifest for the WWW replication at scale (2026-11-06; PREREG in www2027/results/frozen_scale).

    python scripts/build_seed_manifest_scale.py   → manifests/interaction_experiment/www_scale/

P_scale = rules_v2.is_cyber, lang en*, non-empty text, Bluesky, published 2026-08-22..2026-09-23 UTC.
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

OUT = Path("manifests/interaction_experiment/www_scale")
SQL = """
SELECT p.post_id, p.raw_payload->>'uri' AS post_uri, p.author_id, p.published_at AS created_at
FROM posts p JOIN post_labels l ON l.post_id = p.post_id AND l.label_version = 'rules_v2'
WHERE l.is_cyber AND p.source_key = 'bluesky' AND p.lang LIKE 'en%%'
  AND length(coalesce(p.text, '')) > 0
  AND p.published_at >= '2026-08-22' AND p.published_at < '2026-09-24'
ORDER BY p.post_id"""


def main() -> None:
    csv_path, meta_path = OUT / "seed_manifest.csv", OUT / "seed_manifest.meta.json"
    if csv_path.exists():
        sys.exit(f"{csv_path} exists; refusing to overwrite")
    with db.connect() as c:
        rows = c.execute(SQL).fetchall()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["post_id", "post_uri", "author_id", "created_at", "source", "in_scale"])
    for r in rows:
        if r["post_uri"]:
            w.writerow([r["post_id"], r["post_uri"], r["author_id"], r["created_at"].isoformat(), "bluesky", "true"])
    data = buf.getvalue().encode()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(data)
    meta = {"manifest_version": "www_scale", "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "definition": "rules_v2.is_cyber AND lang en* AND text non-empty AND bluesky AND 2026-08-22 <= published_at < 2026-09-24",
            "n": data.count(b"\n") - 1, "sha256": hashlib.sha256(data).hexdigest()}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
