#!/bin/bash
# WSL2 GPU launcher for V11 Sequence-Aware Confidence Training
# Auto-detects nvidia libs from pip packages
set -e

PROJECT_ROOT="/mnt/c/biological/DisorderFlow"
VENV_DIR="$PROJECT_ROOT/venv_wsl"
SITE_PACKAGES="$VENV_DIR/lib/python3.13/site-packages"

# Collect LD_LIBRARY_PATH from pip-installed nvidia packages
LD_PATH=""
for libdir in "$SITE_PACKAGES"/nvidia/*/lib; do
  [ -d "$libdir" ] && LD_PATH="$LD_PATH:$libdir"
done
export LD_LIBRARY_PATH="${LD_PATH#:}:$LD_LIBRARY_PATH"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

echo "=== V11 Sequence-Aware Confidence Training ==="
python3 -c 'import torch; print("PyTorch", torch.__version__, "CUDA:", torch.cuda.is_available())'
echo "================================================"

python3 train.py configs/train/bfn_v11_seqconf_xpu.yml \
    --finetune ./logs/bfn_idp_combined_cuda_2026_06_02__13_48_09/checkpoints/best.pt \
    --device cuda \
    2>&1 | tee -a logs/v11_seqconf_training.log

echo "=== Training complete ==="
