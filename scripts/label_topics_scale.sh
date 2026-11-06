#!/usr/bin/env bash
# topic_v2s: the topic_v2 classifier (same model, revision, prompt, seed, 24 workers as label_gpu.sh) run on
# the www_scale manifest under a new label_version, so frozen topic_v2 labels are untouched (2026-11-06).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="$PWD/logs"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/label_topics_scale.log" 2>&1
PORT=8011
MODEL=Qwen/Qwen3-4B-Instruct-2507; REV=cdbee75f17c01a7cc42f958dc650907174af0554
VLLM=/home/xinyu/miniconda3/envs/cloq/bin/vllm
GPU=${GPU:-1}
echo "=== $(date -u +%FT%TZ) start on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU setsid "$VLLM" serve $MODEL --revision $REV --served-model-name user \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --generation-config vllm --seed 0 \
  > "$LOG_DIR/label_topics_scale_vllm.log" 2>&1 &
VPID=$!
stop_server() { kill -TERM -- -$VPID 2>/dev/null; sleep 10; kill -KILL -- -$VPID 2>/dev/null; echo "server stopped"; }
trap stop_server EXIT
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "vLLM not ready"; exit 1; }
./.venv/bin/python - <<PY
import csv, logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
from ccint.labelers import llm_runner
# select the scale population (rules_v2 is_cyber) instead of the base gate's is_relevant
llm_runner._SELECT = llm_runner._SELECT.replace("AND l.is_relevant", "AND l.is_cyber")
ids = [int(r["post_id"]) for r in csv.DictReader(open("manifests/interaction_experiment/www_scale/seed_manifest.csv"))]
print("topic topic_v2s:", llm_runner.run(base_url="http://127.0.0.1:$PORT/v1", model="user", base_version="rules_v2",
      version="topic_v2s", workers=24, post_ids=ids), flush=True)
PY
echo "=== $(date -u +%FT%TZ) done"
