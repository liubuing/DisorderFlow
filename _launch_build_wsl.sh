#!/bin/bash
# WSL2 GPU build launcher for conformation dataset V4
# Uses in-process XLA caching (~3-4x faster than subprocess mode)
set -e

PROJECT_ROOT="/mnt/c/biological/DisorderFlow"
cd "$PROJECT_ROOT"
source venv_wsl/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CPP_MIN_LOG_LEVEL=3
# Limit JAX GPU memory to avoid OOM at L > 270 (RTX 5060 = 8GB)
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

echo "=== Conformation Build (WSL2 GPU, in-process) ==="
echo "Project: $PROJECT_ROOT"
echo "Python: $(which python3)"
python3 -c "
import jax
print(f'JAX {jax.__version__}  Devices: {jax.devices()}')
print(f'GPU: {jax.devices()[0].device_kind}')
"

echo "================================================"

python3 -u scripts/build/build_conformation_cf.py \
    --n_seeds 5 \
    --num_recycle 0 \
    --split train \
    --gc_pause_sec 1 \
    --crash_cooldown_sec 10 \
    --timeout_minutes 30 \
    --max_consecutive_failures 10 \
    2>&1 | tee -a conformation_build_wsl.log

echo "=== Build finished ==="
