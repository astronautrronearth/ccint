"""Seed manifest for PREREG_DOSE2 (six new domains), 2026-11-07.

    python scripts/build_seed_manifest_dose2.py → manifests/interaction_experiment/www_dose2/
Per domain: 10,000 random roots of source bluesky_<d>, lang en*, non-empty text, not rules_v2 is_cyber,
2026-10-21 <= published_at < 2026-11-05; a post URI inserted under more than one domain source is excluded from all.
Read-only on the database; refuses to overwrite.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path[:0] = ["src"]
from ccint import db  # noqa: E402

DOMS = ["climate", "film", "games", "soccer", "space", "stocks"]
OUT = Path("manifests/interaction_experiment/www_dose2")
SQL = """
SELECT p.post_id, p.raw_payload->>'uri' AS post_uri, p.author_id, p.published_at AS created_at, p.source_key,
       coalesce(l.is_cyber, false) AS is_cyber
FROM posts p LEFT JOIN post_labels l ON l.post_id = p.post_id AND l.label_version = 'rules_v2'
WHERE p.source_key = ANY(%s) AND p.lang LIKE 'en%%' AND length(coalesce(p.text, '')) > 0
  AND p.published_at >= '2026-10-21' AND p.published_at < '2026-11-05'
ORDER BY p.post_id"""


def main() -> None:
    csv_path, meta_path = OUT / "seed_manifest.csv", OUT / "seed_manifest.meta.json"
    if csv_path.exists():
        sys.exit(f"{csv_path} exists; refusing to overwrite")
    with db.connect() as c:
        rows = c.execute(SQL, ([f"bluesky_{d}" for d in DOMS],)).fetchall()
    rows = [r for r in rows if r["post_uri"] and not r["is_cyber"]]
    uc = Counter(r["post_uri"] for r in rows)
    rows = [r for r in rows if uc[r["post_uri"]] == 1]
    rnd = random.Random(20261107)
    pick, sizes = {}, {}
    for d in DOMS:
        pop = [r for r in rows if r["source_key"] == f"bluesky_{d}"]
        sizes[d] = len(pop)
        for r in rnd.sample(pop, min(10000, len(pop))):
            pick[r["post_id"]] = d
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["post_id", "post_uri", "author_id", "created_at", "source"] + [f"in_{d}" for d in DOMS])
    for r in rows:
        if r["post_id"] in pick:
            w.writerow([r["post_id"], r["post_uri"], r["author_id"], r["created_at"].isoformat(), r["source_key"]]
                       + [str(pick[r["post_id"]] == d).lower() for d in DOMS])
    data = buf.getvalue().encode()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(data)
    meta = {"manifest_version": "www_dose2", "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "definition": "PREREG_DOSE2: 10,000 random per domain, seed 20261107", "population_sizes": sizes,
            "n": data.count(b"\n") - 1, "sha256": hashlib.sha256(data).hexdigest()}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
