#!/bin/bash
# WSL2 GPU launcher for Overconfidence Validation
# Compares BFN self-confidence against AF2 ground truth
set -e

PROJECT_ROOT="/mnt/c/biological/DisorderFlow"
cd "$PROJECT_ROOT"

# Detect available venvs
if [ -f venv_wsl/bin/activate ]; then
    VENV="venv_wsl"
elif [ -f venv/bin/activate ]; then
    VENV="venv"
else
    echo "ERROR: No virtualenv found (tried venv_wsl, venv)"
    echo "Create one: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$VENV/bin/activate"
echo "Using venv: $VENV"
echo "Python: $(which python3)"
python3 -c "
import torch
print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3

# ── Choose mode ──
MODE="${1:-all}"
MODELS="${2:-phase3,phase4,idp_combined}"
SAMPLES="${3:-10}"
AF2_RECYCLE="${4:-3}"

echo "================================================"
echo "  Overconfidence Validation"
echo "  Mode:    $MODE"
echo "  Models:  $MODELS"
echo "  Samples: $SAMPLES"
echo "  AF2 Rec: $AF2_RECYCLE"
echo "================================================"

case "$MODE" in
    all)
        python3 validate_overconfidence.py \
            --all \
            --models "$MODELS" \
            --samples "$SAMPLES" \
            --af2-recycle "$AF2_RECYCLE" \
            --device cuda \
            2>&1 | tee oc_validation.log
        ;;
    phase1)
        python3 validate_overconfidence.py \
            --phase1 \
            --models "$MODELS" \
            --samples "$SAMPLES" \
            --device cuda \
            2>&1 | tee oc_validation_phase1.log
        echo ""
        echo "Phase 1 complete. Run Phase 2 (AF2 validation):"
        echo "  bash _launch_oc_validation.sh phase2"
        ;;
    phase2)
        python3 validate_overconfidence.py \
            --phase2 --phase3 \
            --models "$MODELS" \
            --af2-recycle "$AF2_RECYCLE" \
            2>&1 | tee oc_validation_phase2.log
        ;;
    phase3)
        python3 validate_overconfidence.py \
            --phase3 \
            --models "$MODELS" \
            2>&1 | tee oc_validation_phase3.log
        ;;
    *)
        echo "Usage: $0 [all|phase1|phase2|phase3] [models] [samples] [af2_recycle]"
        echo ""
        echo "Examples:"
        echo "  $0 all                                    # Full pipeline"
        echo "  $0 phase1                                 # BFN design only"
        echo "  $0 phase2 phase3,phase4,idp_combined    # AF2 validation on specific models"
        echo "  $0 phase3                                 # Analysis only"
        exit 1
        ;;
esac

echo ""
echo "=== Validation complete ==="
echo "Results: $PROJECT_ROOT/oc_validation_results/"
