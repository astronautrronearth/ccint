#!/usr/bin/env bash
# PREREG_CROSSOVER pipeline after topic labelling: manifest -> (3 collection shards || victim extraction on GPU) -> eval.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/crossover_pipeline.log; exec >>"$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) waiting for labelling PIDs $*"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
echo "=== $(date -u +%FT%TZ) labelling done"; grep -h "topic topic_v2s" logs/label_topics_crossover_*.log
./.venv/bin/python scripts/www_crossover_fresh.py manifest || { echo "manifest failed"; exit 1; }
for i in 0 1 2; do ./.venv/bin/python scripts/www_crossover_fresh.py collect --shard $i --nshard 3 > logs/crossover_collect_$i.log 2>&1 & done
PORT=8015; MODEL=Qwen/Qwen3-4B-Instruct-2507; REV=cdbee75f17c01a7cc42f958dc650907174af0554
export PATH=/home/xinyu/miniconda3/envs/cloq/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=1 setsid /home/xinyu/miniconda3/envs/cloq/bin/vllm serve $MODEL --revision $REV --served-model-name user \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --generation-config vllm --seed 0 > logs/crossover_vllm.log 2>&1 &
VPID=$!
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 10; done
./.venv/bin/python scripts/www_crossover_fresh.py extract --base-url http://127.0.0.1:$PORT/v1
kill -TERM -- -$VPID 2>/dev/null; sleep 10; kill -KILL -- -$VPID 2>/dev/null
echo "=== $(date -u +%FT%TZ) extraction done; waiting for collection"
wait
tail -n 2 logs/crossover_collect_*.log
./.venv/bin/python scripts/www_crossover_fresh.py eval
echo "=== $(date -u +%FT%TZ) PIPELINE DONE"
