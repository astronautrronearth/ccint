#!/usr/bin/env bash
# PREREG_DOSE3: search collection of the six test domains, sequentially, same window as PREREG_DOSE2.
cd "$(dirname "$0")/.."
for d in music books f1 tennis health food; do
  echo "=== $(date -u +%FT%TZ) $d"
  ./.venv/bin/python scripts/collect_contrast.py --since 2026-10-21 --until 2026-11-05 --domain $d > logs/dose3_collect_$d.log 2>&1
  tail -n 1 logs/dose3_collect_$d.log
done
echo "=== $(date -u +%FT%TZ) DOSE3 COLLECTION DONE"
./.venv/bin/python scripts/build_seed_manifest_dose3.py || { echo "manifest failed"; exit 1; }
for i in 0 1 2 3; do ./.venv/bin/python scripts/interaction_collect_dose3.py --shard $i --nshard 4 > logs/dose3_nbhd_$i.log 2>&1 & done
wait
tail -n 1 logs/dose3_nbhd_*.log
echo "=== $(date -u +%FT%TZ) DOSE3 PIPELINE DONE"
