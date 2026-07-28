#!/bin/bash
# Prepare homology-safe v5.1 data and execute the four-stage CUDA curriculum.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "${DISORDERFLOW_VENV:-venv_wsl}/bin/activate"

MODE="${1:---wait}"
BASE_CHECKPOINT="${DISORDERFLOW_BASE_CHECKPOINT:-data/pretrained_candidates/AntibodyDesignBFN_best.pt}"
MMSEQS="${DISORDERFLOW_MMSEQS:-$HOME/.local/bin/mmseqs}"
RUN_ROOT="${DISORDERFLOW_V5_1_LOGDIR:-logs/v5_1_pipeline}"
EXPECTED_COUNT="${DISORDERFLOW_V5_EXPECTED_COUNT:-1289}"
CALIBRATION_REPORT="${DISORDERFLOW_CALIBRATION_REPORT:-data/confidence_conformation_v5/rmsf_calibration_1289.json}"
STATE_DIR="$RUN_ROOT/state"
PIPELINE_LOG="$RUN_ROOT/pipeline.log"

STAGES=(
    "stage1_disorder:configs/train/bfn_v5_1_stage1_disorder.yml"
    "stage2_interface:configs/train/bfn_v5_1_stage2_interface.yml"
    "stage3_codesign:configs/train/bfn_v5_1_stage3_codesign.yml"
    "stage4_idp_conditioned:configs/train/bfn_v5_1_stage4_idp_conditioned.yml"
)

if [[ "$MODE" == "--dry-run" ]]; then
    echo "Base checkpoint: $BASE_CHECKPOINT"
    echo "Required raw v5.1 entries: $EXPECTED_COUNT"
    printf 'Stage: %s\n' "${STAGES[@]}"
    exit 0
fi
if [[ "$MODE" != "--wait" && "$MODE" != "--run" ]]; then
    echo "Usage: $0 [--dry-run|--wait|--run]" >&2
    exit 2
fi

mkdir -p "$STATE_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] v5.1 pipeline mode=$MODE"

raw_count() {
    python - <<'PY'
import lmdb, pickle
env = lmdb.open('data/confidence_conformation_v5/confidence_train.lmdb',
                readonly=True, lock=False, readahead=False, subdir=True)
with env.begin() as txn:
    value = txn.get(b'__len__')
    print(pickle.loads(value) if value else 0)
env.close()
PY
}

if [[ "$MODE" == "--wait" ]]; then
    while [[ "$(raw_count)" -ne "$EXPECTED_COUNT" ]]; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for raw v5.1: $(raw_count)/$EXPECTED_COUNT"
        sleep 300
    done
    while pgrep -f '[b]uild_conformation_cf.py' >/dev/null; do
        sleep 60
    done
fi

python scripts/validate_v5_1_training.py \
    --checkpoint "$BASE_CHECKPOINT" --raw-only \
    --expected-count "$EXPECTED_COUNT"

if [[ ! -e "$CALIBRATION_REPORT" ]]; then
    python scripts/calibrate_af2_rmsf.py \
        --require-count "$EXPECTED_COUNT" --offline \
        --output "$CALIBRATION_REPORT"
fi

if [[ ! -e data/confidence_conformation_v5_clustered/confidence_train.lmdb ]]; then
    python scripts/build/split_conformation_v5.py \
        --output-dir data/confidence_conformation_v5_clustered \
        --mmseqs "$MMSEQS" \
        --expected-count "$EXPECTED_COUNT" \
        --calibration-report "$CALIBRATION_REPORT"
fi
python scripts/audit_dataset_splits.py \
    --dataset conformation_v5_1 \
    data/confidence_conformation_v5_clustered/confidence_train.lmdb \
    data/confidence_conformation_v5_clustered/confidence_val.lmdb \
    --mmseqs "$MMSEQS" \
    --output data/confidence_conformation_v5_clustered/split_audit.json

if [[ ! -e data/phase3_v5_1_pair_clustered/train.lmdb ]]; then
    python scripts/build/resplit_phase3_v5_1.py --mmseqs "$MMSEQS"
fi
python scripts/audit_dataset_splits.py \
    --dataset phase3_v5_1 \
    data/phase3_v5_1_pair_clustered/train.lmdb \
    data/phase3_v5_1_pair_clustered/val.lmdb \
    --mmseqs "$MMSEQS" \
    --output data/phase3_v5_1_pair_clustered/split_audit.json

python scripts/validate_v5_1_training.py \
    --checkpoint "$BASE_CHECKPOINT" \
    --expected-count "$EXPECTED_COUNT" \
    --calibration-report "$CALIBRATION_REPORT" \
    --output "$STATE_DIR/readiness.json"

checkpoint="$BASE_CHECKPOINT"
for stage_spec in "${STAGES[@]}"; do
    stage="${stage_spec%%:*}"
    config="${stage_spec#*:}"
    marker="$STATE_DIR/$stage.checkpoint"
    stage_root="$RUN_ROOT/$stage"
    if [[ -f "$marker" ]]; then
        checkpoint="$(<"$marker")"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing completed $stage: $checkpoint"
        continue
    fi

    python scripts/validate_v5_1_training.py \
        --checkpoint "$checkpoint" --config "$config" \
        --expected-count "$EXPECTED_COUNT" \
        --calibration-report "$CALIBRATION_REPORT"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $stage from $checkpoint"
    python train.py "$config" \
        --init "$checkpoint" \
        --device cuda \
        --accum_steps 4 \
        --no_amp \
        --logdir "$stage_root" \
        --tag v5_1

    next_checkpoint=$(find "$stage_root" -type f -path '*/checkpoints/best.pt' \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -z "$next_checkpoint" ]]; then
        echo "No best checkpoint produced by $stage" >&2
        exit 1
    fi
    python scripts/validate_v5_1_training.py \
        --checkpoint "$next_checkpoint" --config "$config" \
        --expected-count "$EXPECTED_COUNT" \
        --calibration-report "$CALIBRATION_REPORT"
    printf '%s\n' "$next_checkpoint" > "$marker"
    checkpoint="$next_checkpoint"
done

printf '%s\n' "$checkpoint" > "$STATE_DIR/final_checkpoint.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete: $checkpoint"
