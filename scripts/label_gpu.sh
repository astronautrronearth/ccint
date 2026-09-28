#!/usr/bin/env bash
# 在 GPU 上补跑判定门(llm_v2c)与主题(topic_v2),跑完即关服务。
#
#   scripts/label_gpu.sh          # 条件不满足就跳过(每日任务用)
#   scripts/label_gpu.sh --wait   # 一直等到条件满足再跑(补积压用)
#
# [MUST] GPU 与 CLOQ 共用。CLOQ 的实验在跑时一律不动:它的 user simulator 与本判定门是
# 同一个模型,往它的 8000 端口加负载会改变它的批处理,进而可能改变实验输出。
# [MUST] 自起的服务用 8010 端口,不用 8000:CLOQ 的 ctx_control.sh 看到 8000 有服务就直接复用,
# 会误用这里的服务。
# [MUST] 配置与 llm_v2c 已有标签一致:同一模型与 revision、prompt canada_entities v2 + is_cyber v1、
# seed 0、24 并发。改任何一项都应当换 label_version,而不是续写 llm_v2c。
set -uo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="${CCINT_LOG_DIR:-$PWD/logs}"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/label_gpu.log" 2>&1
exec 8>"$LOG_DIR/.label_gpu.lock"
flock -n 8 || { echo "$(date -u +%FT%TZ) already running, skip"; exit 0; }

WAIT=0; [ "${1:-}" = "--wait" ] && WAIT=1
PORT=8010
MODEL=Qwen/Qwen3-4B-Instruct-2507; REV=cdbee75f17c01a7cc42f958dc650907174af0554
VLLM=/home/xinyu/miniconda3/envs/cloq/bin/vllm
NEED_MIB=12000
log() { echo "=== $(date -u +%FT%TZ) $*"; }

cloq_busy() { pgrep -f "cloq.run_arm|scripts/ctx_control.sh|scripts/pipeline" >/dev/null; }
free_gpu() {   # 空闲显存 >= NEED_MIB 的第一张卡
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v need=$NEED_MIB '$2 >= need {print $1; exit}'
}

log "start (wait=$WAIT)"
PENDING=$(./.venv/bin/python -c "
from ccint import db
with db.connect(autocommit=True) as c:
    print(c.execute(\"SELECT count(*) n FROM posts p WHERE length(coalesce(p.text,''))>0 AND NOT EXISTS (SELECT 1 FROM post_labels x WHERE x.post_id=p.post_id AND x.label_version='llm_v2c')\").fetchone()['n'])")
log "posts without llm_v2c: $PENDING"
[ "${PENDING:-0}" -gt 0 ] || { log "nothing to label"; exit 0; }
while :; do
  if cloq_busy; then why="CLOQ experiment running"
  else GPU=$(free_gpu); [ -n "$GPU" ] && break; why="no GPU with ${NEED_MIB} MiB free"; fi
  [ $WAIT = 1 ] || { log "skip: $why"; exit 0; }
  sleep 300
done
log "using GPU $GPU on port $PORT"

CUDA_VISIBLE_DEVICES=$GPU setsid "$VLLM" serve $MODEL --revision $REV --served-model-name user \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --generation-config vllm --seed 0 \
  > "$LOG_DIR/label_gpu_vllm.log" 2>&1 &
VPID=$!
stop_server() { kill -TERM -- -$VPID 2>/dev/null; sleep 10; kill -KILL -- -$VPID 2>/dev/null; log "server stopped"; }
trap stop_server EXIT

for i in $(seq 1 90); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break
  kill -0 $VPID 2>/dev/null || { log "vLLM exited during startup, see label_gpu_vllm.log"; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { log "vLLM not ready after 15 min"; exit 1; }
log "server ready"

./.venv/bin/python - <<EOF
import logging; logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
from ccint.labelers import llm_v2_runner, llm_runner
url = "http://127.0.0.1:$PORT/v1"
print("gate llm_v2c:", llm_v2_runner.run(base_url=url, model="user", workers=24, seed=0,
      label_version="llm_v2c", canada_prompt_version=2), flush=True)
print("topic topic_v2:", llm_runner.run(base_url=url, model="user", base_version="llm_v2c",
      version="topic_v2", workers=24), flush=True)
EOF
rc=$?
log "labelling finished (rc=$rc)"
exit $rc
