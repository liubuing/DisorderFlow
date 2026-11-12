# M3 training-scale and seed-sensitivity

Date: 2026-11-11/12. The homology-filtered M3 training run was evaluated at
epochs 1, 3, 5, 10, 20, and repeated with two additional seeds at epochs
1/3/5/10.

## Seed 1 full curve (20261111)

| Epoch | Mean | Median | 95% CI | Positive | Gate |
|---:|---:|---:|---|---:|---|
| 1 | 0.0778 | 0.0169 | [-0.0478, 0.2151] | 8/15 | fail |
| 3 | 0.1287 | 0.1287 | [0.0101, 0.2601] | 8/15 | fail |
| 5 | 0.1760 | 0.1002 | [0.0739, 0.2853] | 11/15 | pass |
| 10 | 0.1884 | 0.2050 | [0.0767, 0.2989] | 11/15 | pass |
| 20 | 0.1256 | 0.1120 | [0.0228, 0.2350] | 10/15 | fail |

## Seed replication at epochs 1/3/5/10

| Epoch | Seed 20261111 | Seed 20261112 | Seed 20261113 |
|---:|---:|---:|---:|
| 1 | 0.0778 | 0.0148 | **-0.0478** |
| 3 | 0.1287 | 0.0782 | **-0.0424** |
| 5 | 0.1760 | 0.1094 | **-0.0220** |
| 10 | **0.1884 (pass)** | **0.2001 (pass)** | **0.0132 (fail)** |
| 20 | **0.1256 (fail)** | **0.2457 (pass)** | — |

At epoch 10 the three seeds give 0.188, 0.200 and 0.013: mean **0.134**, SD
**0.105**, **2/3 pass**, **1/3 null**. Seed 20261113 is null or negative at
every measured epoch and reaches only 7/15 positive clusters at epoch 10.

At epoch 20 the two trained seeds diverge in opposite directions: seed
20261111 falls to 0.126 (fail), seed 20261112 rises to 0.246 (pass). The
average is 0.186 (SD 0.085).

## Interpretation

- The apparent "training curve" is **not robust**: seed 3 is null at every
  measured epoch; seed 2 is weaker than seed 1 at epochs 1–5; at epoch 20
  seed 1 falls while seed 2 rises.
- The robust statement is narrower: **a from-scratch model can reproduce the
  effect, but only in 2 of 3 seeds at 10 epochs.** The M3 arm is therefore a
  **partial, seed-dependent replication**, not a clean controlled-exposure
  result.
- There is no robust epoch-20 decline: the two seeds move in opposite
  directions. The earlier "decline" was a single-seed observation.
- M2 AntiFold is a fixed pretrained model and has no training-seed variance;
  its cross-architecture replication remains the robust positive result.

## Caveat

Homology-filtered training set, max chain length 300, three seeds for epochs
1/3/5/10, two seeds for epoch 20. This is a descriptive sensitivity analysis,
not a scaling law.

## Data

```text
results/ecls_cross_arch_v1/m3_scale_eval/seed_summary.json
results/ecls_cross_arch_v1/m3_scale_eval/scale_curve.json
results/ecls_cross_arch_v1/m3_scale_eval_seed20261112/scale_curve.json
results/ecls_cross_arch_v1/m3_scale_eval_seed20261113/scale_curve.json
scripts/eval_m3_scale_v1.py
```
