#!/usr/bin/env bash
# 每日刷新:网页源采集 → CVE 抽取 → KEV 同步 → NVD 富化 → 公开快照 → 推送演示站。
#
# 全部在 CPU 上跑,不调用任何付费 API。Bluesky 采集由 collect_incremental.sh 每 15 分钟负责,
# 这里不重复。
#
# [MUST] 不做标注。判定门需要 LLM,选哪种算力是另一个决定;这里不能偷偷跑一个模型,
# 否则每天的标签会来自一个没人审过的配置。
#
# 环境变量:
#   SKIP_NVD=1    跳过 NVD 富化(测试用)
#   SKIP_PUSH=1   只生成,不推送
#   DEMO_DIR      演示站仓库路径,默认 ~/ccint-demo
set -uo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="${CCINT_LOG_DIR:-$PWD/logs}"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/daily_refresh.log" 2>&1
exec 9>"$LOG_DIR/.daily_refresh.lock"
flock -n 9 || { echo "$(date -u +%FT%TZ) another refresh is running, skip"; exit 0; }

PY=./.venv/bin/python
DEMO_DIR="${DEMO_DIR:-$HOME/ccint-demo}"
step() { echo "=== $(date -u +%FT%TZ) $*"; }
fail=0

step "start"

step "web sources"
timeout 20m ./.venv/bin/ccint collect web --since 3d || { echo "web collect failed"; fail=1; }

step "CVE extraction + KEV sync"
timeout 20m $PY -c "
from ccint.osint import cve
print(cve.link_corpus(only_new=True))
print(cve.sync_kev())" || { echo "cve link/kev failed"; fail=1; }

if [ "${SKIP_NVD:-0}" != 1 ]; then
  # 无 API key 时 NVD 限速约 6.5s/个;设上限,剩下的明天接着查
  step "NVD enrichment (max 45 min)"
  timeout 45m $PY scripts/enrich_cves.py --min-posts 5 || echo "nvd enrichment stopped (timeout or error), continuing"
fi

step "export public snapshot"
if ! timeout 20m $PY scripts/export_portal_snapshot.py --public --out outputs/portal_public/data.json; then
  echo "export failed, not publishing"; exit 1
fi

if [ "${SKIP_PUSH:-0}" != 1 ]; then
  step "publish demo site"
  cp outputs/portal_public/site/index.html outputs/portal_public/site/data.json "$DEMO_DIR/"
  cd "$DEMO_DIR"
  if git diff --quiet; then
    echo "no change"
  else
    # 演示站只保留一个提交:每天的快照不需要历史,否则每天多一份 1.7 MB 的 blob
    git add -A && git commit -q --amend -m "Signal Desk demo snapshot $(date -u +%F)" \
      && git push -q -f origin main && echo "pushed"
  fi
fi

step "done (fail=$fail)"
exit $fail
