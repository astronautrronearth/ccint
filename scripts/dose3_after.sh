#!/usr/bin/env bash
cd "$(dirname "$0")/.."
exec >>logs/dose3_after.log 2>&1
until grep -q "DOSE3 PIPELINE DONE" logs/dose3_collect.log; do sleep 60; done
cp outputs/interaction_experiment/www_eval/manipulation.json outputs/interaction_experiment/www_eval/manipulation_pre_dose3.json
./.venv/bin/python scripts/www_manipulation.py > logs/manip_dose3.log 2>&1 || echo "MANIP FAILED"
./.venv/bin/python scripts/www_dose3_tests.py
echo "=== $(date -u +%FT%TZ) DOSE3 TESTS DONE"
