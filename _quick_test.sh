#!/bin/bash
cd /mnt/c/biological/DisorderFlow
source venv_wsl/bin/activate
python -u scripts/benchmark_h3_t2.1_recovery.py 2>&1 | head -10
