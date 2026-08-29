#!/bin/bash
# V13.1 A2→A3 sequential training + P2 probes
# Single submission to avoid task cleanup issues
set -e
cd /mnt/d/biological/DisorderFlow
source venv_wsl/bin/activate

V20_CKPT="logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt"
STAMP=$(date +%Y%m%d_%H%M%S)
echo "V13.1 A2->A3 chain starting at $STAMP"

# ── A2: Resume from existing checkpoint ──
echo "=== A2: Resume training (no anti_degen) ==="
A2_LOG="logs/bfn_v20_ablate_a2_nodegen_2026_07_04__13_57_37_v13_a2_nodegen_cpu"
if [ -f "$A2_LOG/checkpoints/best.pt" ]; then
    echo "Resuming A2 from $A2_LOG/checkpoints/best.pt"
    CUDA_VISIBLE_DEVICES="" python train.py \
        logs/bfn_v20_ablate_a2_nodegen/bfn_v20_ablate_a2_nodegen.yml \
        --device cpu \
        --resume "$A2_LOG/checkpoints/best.pt" \
        --tag v13_a2_resume2
else
    echo "No A2 checkpoint found, starting fresh"
    mkdir -p "$A2_LOG/checkpoints"
    CUDA_VISIBLE_DEVICES="" python train.py \
        logs/bfn_v20_ablate_a2_nodegen/bfn_v20_ablate_a2_nodegen.yml \
        --device cpu \
        --finetune "$V20_CKPT" \
        --tag v13_a2_fresh
fi
echo "A2 training done at $(date)"

# Find A2 checkpoint
A2_CKPT=$(find logs/ -name "best.pt" -path "*ablate_a2*" -newer /dev/null 2>/dev/null | sort | tail -1)
echo "A2 checkpoint: $A2_CKPT"

# ── A2 P2 probe ──
echo "=== A2 P2 probe ==="
python scripts/v13_p2_ablation_probe.py \
    "$A2_CKPT" A2_nodegen --n-samples 8 --n-seeds 3
echo "A2 P2 done at $(date)"

# ── A3: Head only (fresh start) ──
echo "=== A3: Training (head_seq only) ==="
CUDA_VISIBLE_DEVICES="" python train.py \
    logs/bfn_v20_ablate_a3_headonly/bfn_v20_ablate_a3_headonly.yml \
    --device cpu \
    --finetune "$V20_CKPT" \
    --tag v13_a3_chain
echo "A3 training done at $(date)"

# Find A3 checkpoint
A3_CKPT=$(find logs/ -name "best.pt" -path "*ablate_a3*" 2>/dev/null | sort | tail -1)
echo "A3 checkpoint: $A3_CKPT"

# ── A3 P2 probe ──
echo "=== A3 P2 probe ==="
python scripts/v13_p2_ablation_probe.py \
    "$A3_CKPT" A3_headonly --n-samples 8 --n-seeds 3
echo "A3 P2 done at $(date)"

echo "=== V13.1 A2+A3 chain COMPLETE at $(date) ==="
