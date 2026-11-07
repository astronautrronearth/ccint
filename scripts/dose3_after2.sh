#!/usr/bin/env bash
# Rerun of dose3_after.sh after the shard-0 NUL crash (run 4684 derived by hand; remaining 440 seeds in a new run).
cd "$(dirname "$0")/.."
exec >>logs/dose3_after.log 2>&1
echo "=== $(date -u +%FT%TZ) after2 waiting"
until grep -q "DOSE3 PIPELINE DONE" logs/dose3_collect.log && grep -q "^run .* seeds, .* edges" logs/dose3_nbhd_0b.log; do
  grep -q Traceback logs/dose3_nbhd_0b.log && { echo "SHARD0B FAILED"; exit 1; }
  sleep 30; done
cp outputs/interaction_experiment/www_eval/manipulation.json outputs/interaction_experiment/www_eval/manipulation_pre_dose3.json
./.venv/bin/python scripts/www_manipulation.py > logs/manip_dose3.log 2>&1 || echo "MANIP FAILED"
./.venv/bin/python scripts/www_dose3_tests.py
echo "=== $(date -u +%FT%TZ) DOSE3 TESTS DONE"
