#!/bin/bash
# Correct failed contrastive/conditioning objectives, then re-run all retention gates.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "${DISORDERFLOW_VENV:-venv_wsl}/bin/activate"

BASE_CHECKPOINT="${DISORDERFLOW_CORRECTIVE_BASE:-$(<logs/v5_1_pipeline/state/final_checkpoint.txt)}"
CONFIG="configs/train/bfn_v5_1_stage5_corrective.yml"
RUN_ROOT="${DISORDERFLOW_CORRECTIVE_LOGDIR:-logs/v5_1_corrective}"
RESULT_ROOT="results/v5_1_corrective"
STATE_DIR="$RUN_ROOT/state"
mkdir -p "$STATE_DIR" "$RESULT_ROOT"
exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1

python scripts/validate_v5_1_training.py \
    --checkpoint "$BASE_CHECKPOINT" \
    --config "$CONFIG" \
    --expected-count 1289 \
    --calibration-report data/confidence_conformation_v5/rmsf_calibration_1289.json \
    --output "$STATE_DIR/readiness.json"

python train.py "$CONFIG" \
    --init "$BASE_CHECKPOINT" \
    --device cuda \
    --accum_steps 4 \
    --no_amp \
    --logdir "$RUN_ROOT/train" \
    --tag corrective

checkpoint=$(find "$RUN_ROOT/train" -type f -path '*/checkpoints/best.pt' \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "$checkpoint" ]]; then
    echo "Corrective training produced no best checkpoint" >&2
    exit 1
fi

python scripts/validate_v5_1_training.py \
    --checkpoint "$checkpoint" \
    --config "$CONFIG" \
    --expected-count 1289 \
    --calibration-report data/confidence_conformation_v5/rmsf_calibration_1289.json

python scripts/evaluate_v5_1_phase3.py \
    --checkpoint "$checkpoint" \
    --device cuda \
    --require-gates \
    --output "$RESULT_ROOT/phase3_condition_ablation.json"

python scripts/evaluate_stage1_nmr_ranking.py \
    --checkpoint "$checkpoint" \
    --baseline-checkpoint "$BASE_CHECKPOINT" \
    --config configs/train/bfn_v5_1_stage1_disorder.yml \
    --calibration-report data/confidence_conformation_v5/rmsf_calibration_1289.json \
    --lmdb data/confidence_conformation_v5_clustered/confidence_calibration.lmdb \
    --device cuda \
    --output "$RESULT_ROOT/nmr_retention.json"

printf '%s\n' "$checkpoint" > "$STATE_DIR/final_checkpoint.txt"
echo "Corrective pipeline complete: $checkpoint"
