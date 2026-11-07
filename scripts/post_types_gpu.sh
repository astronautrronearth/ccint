#!/usr/bin/env bash
# E6 content types on a local vLLM server (same model/revision as topic_v2s), GPU 1.
cd "$(dirname "$0")/.."
exec >>logs/post_types.log 2>&1
PORT=8016; MODEL=Qwen/Qwen3-4B-Instruct-2507; REV=cdbee75f17c01a7cc42f958dc650907174af0554
export PATH=/home/xinyu/miniconda3/envs/cloq/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=${GPU:-1} setsid /home/xinyu/miniconda3/envs/cloq/bin/vllm serve $MODEL --revision $REV --served-model-name user \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --generation-config vllm --seed 0 > logs/post_types_vllm.log 2>&1 &
VPID=$!
trap 'kill -TERM -- -$VPID 2>/dev/null; sleep 10; kill -KILL -- -$VPID 2>/dev/null' EXIT
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 10; done
echo "=== $(date -u +%FT%TZ) start"
./.venv/bin/python scripts/www_post_types.py --base-url http://127.0.0.1:$PORT/v1
echo "=== $(date -u +%FT%TZ) POST TYPES DONE"
