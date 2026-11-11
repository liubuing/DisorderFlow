# ECLS controlled-exposure pilot (M3, 281 PDBs)

Status: **pilot only, not a controlled-exposure verdict.** The full pre-2021
dataset (4,530 archive-covered PDBs) is being built; this report documents the
pipeline validation and the pilot-scale null.

## What was trained

- Data: the first 300 inventory PDBs; 281 PDBs / 557 chains passed parsing.
- Model: ProteinMPNN-style 3 encoder + 3 decoder layers, hidden 128, k=48
  (~1.7M parameters), trained from scratch.
- Objective: masked-residue NLL, one chain masked per complex, the other
  chain visible as context.
- Optimizer: Adam, lr 1e-4, batch 2 complexes, 20 epochs.
- Loss: train 6.02 → 1.77; validation 5.33 → 1.48. The model is learning, but
  it is small and undertrained relative to the full pre-2021 set.

## ECLS result (pilot checkpoint)

| Statistic | Value |
|---|---:|
| Mean ECLS advantage | **0.015699** |
| Median ECLS advantage | 0.060156 |
| Cluster-bootstrap 95% CI | **[-0.113458, 0.151689]** |
| Positive clusters | 8/15 (0.533) |
| Preregistered development gate | **fail** |

The pilot model does **not** reproduce the ECLS effect: the CI crosses zero,
the positive-cluster fraction is near chance, and the magnitude is ~9% of the
frozen ProteinMPNN effect.

## Interpretation (bounded)

- This is **not** evidence that ECLS is exposure-dependent. The pilot model is
  trained on 281 PDBs and 20 epochs; it is a pipeline validation, not the
  controlled-exposure arm.
- It does show that a small from-scratch model does not automatically inherit
  the ECLS effect. The effect may require a larger/better-trained model, a
  broader structural prior, or both.
- The full M3 arm (4,530 archive-covered pre-2021 PDBs, longer training) is
  the actual controlled-exposure test. Its result is pending.

## Reproduce

```bash
python scripts/build_m3_dataset_v1.py --max-pdb 300 \
  --out data/controlled_replication_v1/m3_dataset
python scripts/train_m3_v1.py \
  --data data/controlled_replication_v1/m3_dataset \
  --out results/ecls_cross_arch_v1/m3_pilot --epochs 20
python scripts/benchmark_h3_epitope_delta.py \
  --config configs/benchmarks/peptide_h3_ecls_m3_pilot_v1.yml \
  --out-dir results/ecls_cross_arch_v1/m3_ecls
```

The M3 pilot config keeps the frozen temporal panel and shuffle protocol; only
the checkpoint changes.
