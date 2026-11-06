#!/usr/bin/env bash
# E-B victim extraction on a local vLLM server (same model and revision as topic_v2s), GPU 0 by default.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="$PWD/logs"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/extract_victims.log" 2>&1
PORT=8012
MODEL=Qwen/Qwen3-4B-Instruct-2507; REV=cdbee75f17c01a7cc42f958dc650907174af0554
VLLM=/home/xinyu/miniconda3/envs/cloq/bin/vllm
export PATH=/home/xinyu/miniconda3/envs/cloq/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
GPU=${GPU:-0}
echo "=== $(date -u +%FT%TZ) start on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU setsid "$VLLM" serve $MODEL --revision $REV --served-model-name user \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --generation-config vllm --seed 0 \
  > "$LOG_DIR/extract_victims_vllm.log" 2>&1 &
VPID=$!
stop_server() { kill -TERM -- -$VPID 2>/dev/null; sleep 10; kill -KILL -- -$VPID 2>/dev/null; echo "server stopped"; }
trap stop_server EXIT
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "vLLM not ready"; exit 1; }
./.venv/bin/python scripts/www_extract_victims.py --base-url http://127.0.0.1:$PORT/v1
echo "=== $(date -u +%FT%TZ) done"
