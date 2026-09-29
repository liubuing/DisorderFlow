#!/bin/bash
# Auto-restart build watchdog - keeps the build alive through crashes/WSL restarts
cd /mnt/d/biological/DisorderFlow
source venv_wsl/bin/activate

while true; do
    echo "=== Watchdog: Starting build at $(date) ==="
    python3 -u scripts/build/build_conformation_cf.py \
        --n_seeds 5 --num_recycle 0 --split train \
        --gc_pause_sec 1 --timeout_minutes 30 \
        --max_consecutive_failures 10 \
        2>&1 | tee -a /mnt/d/biological/DisorderFlow/_build_v5.log

    EXIT=$?
    echo "=== Watchdog: Build exited with code $EXIT at $(date) ==="
    echo "=== Restarting in 10 seconds... ==="
    sleep 10
done
