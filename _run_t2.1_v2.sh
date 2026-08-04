#!/bin/bash
# T2.1 v2 benchmark runner
set -e
cd /mnt/c/biological/DisorderFlow
source venv_openmm/bin/activate
exec python -u scripts/benchmark_h3_t2.1_recovery.py --config configs/benchmarks/peptide_h3_t2.1_v2_temporal_final.yml
