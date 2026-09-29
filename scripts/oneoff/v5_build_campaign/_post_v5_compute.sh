#!/bin/bash
# Legacy entry point retained as a guarded alias for the v5.1 pipeline.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
source "${DISORDERFLOW_VENV:-venv_wsl}/bin/activate"

exec bash scripts/run_v5_1_training_pipeline.sh --run
