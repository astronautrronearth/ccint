#!/usr/bin/env bash
# PREREG_DOSE2 after search collection: manifest -> 4 neighbourhood shards -> eval.
cd "$(dirname "$0")/.."
exec >>logs/dose2_pipeline.log 2>&1
until grep -q "DOSE2 COLLECTION DONE" logs/dose2_collect.log; do sleep 30; done
echo "=== $(date -u +%FT%TZ) collection done"; cat logs/dose2_collect.log
./.venv/bin/python scripts/build_seed_manifest_dose2.py || { echo "manifest failed"; exit 1; }
for i in 0 1 2 3; do ./.venv/bin/python scripts/interaction_collect_dose2.py --shard $i --nshard 4 > logs/dose2_nbhd_$i.log 2>&1 & done
wait
tail -n 1 logs/dose2_nbhd_*.log
./.venv/bin/python scripts/www_eval_dose2.py
echo "=== $(date -u +%FT%TZ) DOSE2 PIPELINE DONE"
