#!/bin/bash
# Run T2.1 v2 one record at a time with caching
cd /mnt/c/biological/DisorderFlow
source venv_openmm/bin/activate
rm -f _t2.1_v2_progress.log

for i in $(seq 1 31); do
    echo "[$(date)] Record $i/31" | tee -a _t2.1_v2_progress.log
    python -u scripts/benchmark_h3_t2.1_recovery.py \
        --config configs/benchmarks/peptide_h3_t2.1_v2_temporal_final.yml \
        --max-records 1 \
        2>&1 | tail -3 | tee -a _t2.1_v2_progress.log
    
    # Remove work dir so next run picks next uncached record
    # (but keep results)
    completed=$(ls results/publication/h3_t2.1_v2_temporal_final/work/ 2>/dev/null | wc -l)
    echo "  Completed so far: $completed" | tee -a _t2.1_v2_progress.log
done

echo "[$(date)] All 31 records done" | tee -a _t2.1_v2_progress.log
