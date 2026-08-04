#!/bin/bash
# 后台启动 v5 多构象数据集完整构建（含看门狗自动重启）
# 环境：WSL2 Ubuntu-24.04 + venv_wsl (jax 0.5.3 cuda12 + alphafold + lmdb + torch)
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ -f "$PROJECT_DIR/.v5_build_stopped" ]]; then
    echo "V5 build intentionally stopped at the accepted dataset size" >&2
    exit 0
fi
source "${DISORDERFLOW_VENV:-venv_wsl}/bin/activate"

# pip's CUDA runtime packages are not registered with the system linker after
# a fresh WSL session. AlphaFold imports torch while reading source records.
CUPTI_LIB="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_cupti/lib"
NVJITLINK_LIB="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH="$NVJITLINK_LIB:$CUPTI_LIB:${LD_LIBRARY_PATH:-}"

# 直接运行 build 脚本（自带 auto-resume）
# --num_recycle 0: 省显存（RTX 5060 8GB）
# --max_consecutive_failures 0: 永不自动暂停（崩溃由看门狗重启）
exec python -u scripts/build/build_conformation_cf.py \
    --n_seeds 5 \
    --num_recycle 0 \
    --split train \
    --gc_pause_sec 3 \
    --max_consecutive_failures 0 \
    --timeout_minutes 90
