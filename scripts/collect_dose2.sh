#!/usr/bin/env bash
# PREREG_DOSE2: search collection of the six new domains, sequentially (concurrent runs double-insert shared posts).
cd "$(dirname "$0")/.."
for d in climate film games soccer space stocks; do
  echo "=== $(date -u +%FT%TZ) $d"
  ./.venv/bin/python scripts/collect_contrast.py --since 2026-10-21 --until 2026-11-05 --domain $d > logs/dose2_collect_$d.log 2>&1
  tail -n 1 logs/dose2_collect_$d.log
done
echo "=== $(date -u +%FT%TZ) DOSE2 COLLECTION DONE"
