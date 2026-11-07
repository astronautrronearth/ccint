#!/usr/bin/env bash
# After the six held-out domains are collected and www_eval_dose2.py has run: every pre-registered analysis
# (PREREG_DOSE2 + amendments 1-8) on all eleven settings, unchanged code at this commit.
cd "$(dirname "$0")/.."
exec >>logs/dose2_all.log 2>&1
until grep -q "DOSE2 PIPELINE DONE" logs/dose2_pipeline.log; do sleep 60; done
echo "=== $(date -u +%FT%TZ) start all analyses"
./.venv/bin/python scripts/www_dose2_mech.py
./.venv/bin/python scripts/www_dose2_theory.py
./.venv/bin/python scripts/www_baselines_union.py
./.venv/bin/python scripts/www_budget_fetch.py --seeds 20
echo "=== $(date -u +%FT%TZ) ALL ANALYSES DONE"
