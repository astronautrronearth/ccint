#!/usr/bin/env bash
# topic_v2s on the crossover candidate pool (PREREG_CROSSOVER): fresh-window rules_v2 is_cyber posts matching the
# ransomware prefilter regex, same model/revision/config as label_topics_contrast.sh. Sharded: SHARD/NSHARD, GPU, PORT.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="$PWD/logs"; mkdir -p "$LOG_DIR"
SHARD=${SHARD:-0}; NSHARD=${NSHARD:-1}; GPU=${GPU:-0}; PORT=${PORT:-8013}
exec >>"$LOG_DIR/label_topics_crossover_$SHARD.log" 2>&1
MODEL=Qwen/Qwen3-4B-Instruct-2507; REV=cdbee75f17c01a7cc42f958dc650907174af0554
VLLM=/home/xinyu/miniconda3/envs/cloq/bin/vllm
export PATH=/home/xinyu/miniconda3/envs/cloq/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "=== $(date -u +%FT%TZ) start shard $SHARD/$NSHARD on GPU $GPU port $PORT"
CUDA_VISIBLE_DEVICES=$GPU setsid "$VLLM" serve $MODEL --revision $REV --served-model-name user \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization ${UTIL:-0.80} --generation-config vllm --seed 0 \
  > "$LOG_DIR/label_topics_crossover_vllm_$SHARD.log" 2>&1 &
VPID=$!
stop_server() { kill -TERM -- -$VPID 2>/dev/null; sleep 10; kill -KILL -- -$VPID 2>/dev/null; echo "server stopped"; }
trap stop_server EXIT
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "vLLM not ready"; exit 1; }
./.venv/bin/python - <<PY
import logging, sys
sys.path[:0] = ["src", "scripts"]
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
from ccint import db
from ccint.labelers import llm_runner
from www_crossover_fresh import candidate_pool_ids
llm_runner._SELECT = llm_runner._SELECT.replace("AND l.is_relevant", "AND l.is_cyber")
ids = [i for i in candidate_pool_ids() if i % $NSHARD == $SHARD]
print("shard ids", len(ids), flush=True)
print("topic topic_v2s:", llm_runner.run(base_url="http://127.0.0.1:$PORT/v1", model="user", base_version="rules_v2",
      version="topic_v2s", workers=24, post_ids=ids), flush=True)
PY
echo "=== $(date -u +%FT%TZ) done"
