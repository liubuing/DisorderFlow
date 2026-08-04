#!/bin/bash
set -e
cd /mnt/c/biological/DisorderFlow
source venv_wsl/bin/activate
rm -rf results/publication/h3_t2.1_temporal_final_v1/work
nohup python -u scripts/benchmark_h3_t2.1_recovery.py > _t2.1_benchmark.log 2>&1 &
echo "PID: $!"
sleep 5
head -30 _t2.1_benchmark.log
echo "..."
echo "Monitor with: tail -f /mnt/c/biological/DisorderFlow/_t2.1_benchmark.log"
