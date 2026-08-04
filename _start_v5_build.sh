#!/bin/bash
# 启动多构象数据集 v5 构建（后台 + 看门狗）
# WSL2 + GPU (jax 0.5.3 cuda12)
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ -f "$PROJECT_DIR/.v5_build_stopped" ]]; then
    echo "V5 build intentionally stopped at the accepted dataset size" >&2
    exit 0
fi
source "${DISORDERFLOW_VENV:-venv_wsl}/bin/activate"

LOG="${DISORDERFLOW_BUILD_LOG:-$PROJECT_DIR/_build_v5.log}"

# 直接运行 build 脚本（看门狗可选；此处用单进程 + 自动 resume）
# --num_recycle 0: 省显存（RTX 5060 仅 8GB）
# --max_consecutive_failures 0: 永不自动暂停（看门狗逻辑由脚本自身处理）
# 脚本会自动从已有条目 resume
nohup python -u scripts/build/build_conformation_cf.py \
    --n_seeds 5 \
    --num_recycle 0 \
    --split train \
    --gc_pause_sec 3 \
    --max_consecutive_failures 0 \
    --timeout_minutes 90 \
    > "$LOG" 2>&1 &

echo "Build launched, PID=$!"
sleep 3
echo "=== first 20 lines of log ==="
head -20 "$LOG"
