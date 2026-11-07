#!/usr/bin/env bash
# Resume PREREG_DOSE2 neighbourhoods after the crossover collection (rate limit shared), then evaluate.
cd "$(dirname "$0")/.."
exec >>logs/dose2_pipeline.log 2>&1
until grep -q "PIPELINE DONE" logs/crossover_pipeline.log; do sleep 30; done
echo "=== $(date -u +%FT%TZ) resume dose2 neighbourhoods"
for i in 0 1 2 3; do ./.venv/bin/python scripts/interaction_collect_dose2.py --shard $i --nshard 4 >> logs/dose2_nbhd_$i.log 2>&1 & done
wait
./.venv/bin/python scripts/www_eval_dose2.py
echo "=== $(date -u +%FT%TZ) DOSE2 PIPELINE DONE"
