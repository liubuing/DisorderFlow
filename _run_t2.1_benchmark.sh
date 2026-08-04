#!/bin/bash
# T2.1 benchmark runner - run inside WSL directly
set -e
cd /mnt/c/biological/DisorderFlow
source venv_wsl/bin/activate

echo "=== T2.1 Benchmark ==="
echo "Config: configs/benchmarks/peptide_h3_t2.1_temporal_final_v1.yml"
echo "Output: results/publication/h3_t2.1_temporal_final_v1"
echo ""

# Check OpenMM platforms
python -c "
from openmm import Platform
for i in range(Platform.getNumPlatforms()):
    p = Platform.getPlatform(i)
    print(f'  Platform {i}: {p.getName()} speed={p.getSpeed()}')
"

echo ""
echo "Starting benchmark..."
python scripts/benchmark_h3_t2.1_recovery.py "$@"

echo ""
echo "=== Generating report ==="
python scripts/analyze_h3_t2.1_recovery.py
