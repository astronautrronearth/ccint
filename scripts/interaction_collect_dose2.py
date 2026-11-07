"""Neighbourhood collection for the E-D contrast manifest (www_contrast), sharded.

    python scripts/interaction_collect_contrast.py --shard 0 --nshard 3   # one process per shard
Re-running a shard collects only seeds of that shard without an outcome in an earlier scale run.
Same collector, depth, endpoints and page limit as the main run. Network only, no models.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ccint import db
from ccint.interaction import manifest as mf
from ccint.interaction.collect import git_commit, run
from ccint.interaction.derive import derive_run

OUT = Path("outputs/interaction_experiment/collection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshard", type=int, required=True)
    a = ap.parse_args()
    m = mf.load("www_dose2")
    seeds = [s for s in sum((m.population(f"in_{d}") for d in ("climate", "film", "games", "soccer", "space", "stocks")), []) if int(s["post_id"]) % a.nshard == a.shard]
    with db.connect() as c:
        done = {r["seed_post_id"] for r in c.execute("""
            SELECT DISTINCT o.seed_post_id FROM interaction_seed_outcomes o
            JOIN neighborhood_runs n ON n.run_id = o.collection_run_id
            WHERE n.purpose = 'dose2' AND n.manifest_sha256 = %s""", (m.sha256,))}
    todo = [s for s in seeds if int(s["post_id"]) not in done]
    print(f"shard {a.shard}/{a.nshard}: seeds {len(seeds)} already {len(seeds) - len(todo)} to collect {len(todo)}", flush=True)
    if not todo:
        return
    rid = run(m, todo, purpose="dose2")
    with db.transaction() as conn:
        n = derive_run(conn, rid, git_commit())
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"dose2_runs_{a.shard}.json"
    runs = json.loads(p.read_text()) if p.exists() else []
    runs.append(rid)
    p.write_text(json.dumps(runs))
    print(f"run {rid}: {len(todo)} seeds, {n} edges", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    main()
