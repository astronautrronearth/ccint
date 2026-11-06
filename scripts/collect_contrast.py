"""Contrast-domain collection for the WWW evaluation (PREREG: www2027/results/frozen_eval/PREREG.md).

    python scripts/collect_contrast.py --since 2026-10-21 --until 2026-11-05

Collects Bluesky posts for a non-security domain (NBA basketball) into the same `posts` table under a
separate source_key (`bluesky_nba`) and query_version (`nba_v1`), so the cybersecurity corpus and its
checkpoints are untouched. Day-chunked backfill, a fixed number of pages per term and day (the sampling
frame is "the most recent pages per term-day", recorded in collection_runs.query_spec).
"""
import argparse
import sys
from datetime import datetime, timezone

sys.path[:0] = ["src"]
from ccint.collectors.bluesky import BlueskyCollector  # noqa: E402
from ccint import db  # noqa: E402
from ccint.ingest import run_collection  # noqa: E402

# One term per team (city added where the nickname is an ordinary word), plus the league.
NBA_TERMS = [
    "NBA", "Lakers", "Celtics", "Warriors", "Knicks", "Miami Heat", "Bucks", "Nuggets", "Mavericks",
    "Phoenix Suns", "76ers", "Clippers", "Timberwolves", "OKC Thunder", "Cavaliers", "Raptors", "Pistons",
    "Pacers", "Grizzlies", "Pelicans", "Spurs", "Houston Rockets", "Orlando Magic", "Atlanta Hawks",
    "Charlotte Hornets", "Chicago Bulls", "Brooklyn Nets", "Utah Jazz", "Washington Wizards",
    "Sacramento Kings", "Trail Blazers",
]


class ContrastCollector(BlueskyCollector):
    source_key = "bluesky_nba"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--pages", type=int, default=3)
    a = ap.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)
    until = datetime.fromisoformat(a.until).replace(tzinfo=timezone.utc)
    with ContrastCollector(NBA_TERMS, query_version="nba_v1", lang="en") as col:
        res = run_collection(col, mode="backfill", since=since, until=until, max_pages_per_term=a.pages,
                             chunk_days=1)
    # RawPost.source_key is a module constant in the Bluesky collector; restamp this run's new rows
    with db.transaction() as c:
        n = c.execute("UPDATE posts SET source_key = %s WHERE first_seen_run = %s AND source_key = 'bluesky'",
                      (ContrastCollector.source_key, res["run_id"])).rowcount
    print(res, "restamped", n)


if __name__ == "__main__":
    main()
