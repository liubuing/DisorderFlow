# ECLS controlled-exposure result (M3, from scratch on pre-2021 SAbDab)

Status: **M3 complete; replication is seed-dependent (2 of 3 seeds).** The
frozen ECLS decision is unchanged. No binding claim.

## Training contract

- Data: pre-2021 SAbDab experimental structures only (deposition
  ≤ 2021-08-02); 4,506 PDBs / 8,585 chains after parsing.
- No OAS modeled structures, no AlphaFold DB predictions.
- Model: ProteinMPNN-style 3 encoder + 3 decoder layers, hidden 128, k=48
  (~1.7M parameters), trained from scratch.
- Objective: masked-residue NLL, one chain masked per complex.
- Optimizer: Adam, lr 1e-4, batch 2 complexes, 10 epochs, max chain length 300.
- Loss: train 2.93 → 1.41; validation 2.33 → 1.35.

## ECLS result

| Statistic | M0 ProteinMPNN | M2 AntiFold | **M3 from scratch** |
|---|---:|---:|---:|
| Mean ECLS advantage | 0.172281 | 0.306928 | **0.205317** |
| Median | 0.118699 | 0.308123 | **0.196236** |
| 95% CI | [0.059156, 0.291920] | [0.124085, 0.497105] | **[0.088715, 0.321356]** |
| Positive clusters | 12/15 | 12/15 | **14/15** |
| Gate | reference | replication | **replication** |

Per-cluster comparison:

| Pair | Pearson | Spearman | Sign agreement |
|---|---:|---:|---:|
| M0 vs M3 | 0.638 | 0.714 | 11/15 |
| M2 vs M3 | **0.865** | **0.861** | **13/15** |

M3-negative cluster: only E27 (-0.3524). M0-negative clusters E149, E181 and
E214 flip positive in M3; E27 flips negative.

## Seed sensitivity (added 2026-11-12)

The table above is seed 20261111. Two additional seeds were trained with the
same data and contract:

| Seed | Epoch 10 mean | Positive clusters | Gate |
|---|---:|---:|---|
| 20261111 | **0.1884** | 11/15 | pass |
| 20261112 | **0.2001** | 13/15 | pass |
| 20261113 | **0.0132** | 7/15 | **fail** |

Epoch-10 mean across seeds: **0.134**, SD **0.105**. Seed 20261113 is null or
negative at every measured epoch (-0.048, -0.042, -0.022, +0.013 for epochs
1/3/5/10). At epoch 20 the two trained seeds diverge: seed 20261111 = 0.126
(fail), seed 20261112 = 0.246 (pass). The M3 arm is therefore a **partial,
seed-dependent replication**, not a clean controlled-exposure result and not
a robust training-scale trend.

## Exposure and overlap check

- PDB overlap between the 4,506 training PDBs and the 21 temporal-panel PDBs:
  **0**.
- One exact H3 substring overlap: panel `8h8q_H_L` (cluster E121) H3
  `ALFDGYYPWFAS` occurs in training chain `7e6p_H`.
- Sensitivity excluding E121: M0 mean 0.1603 (11/14 positive), M3 mean
  **0.2060 (13/14 positive)**. The result is robust to the single exact H3
  overlap.
- Homology-level exposure is not fully excluded; the panel is post-2021, but
  a pre-2021 model can still have seen homologs. This is the same limitation
  the frozen ECLS protocol already discloses.

## Preregistered decision

The replication rule required: mean > 0, 95% CI lower > 0, ≥10/15 positive
clusters, magnitude within 2× of 0.172.

| Criterion | M3 | Pass |
|---|---|---|
| Mean > 0 | 0.2053 | yes |
| CI lower > 0 | 0.0887 | yes |
| Positive clusters ≥10/15 | 14/15 | yes |
| Magnitude within 2× | 1.19× | yes |

**Verdict (seed 20261111): `replication`. Arm-level verdict:
`partial_seed_dependent_replication`.** A model trained from scratch on
pre-2021 antibody structures can reproduce the ECLS advantage on the
post-2021 temporal panel, but only in 2 of 3 seeds; one seed is null. The
effect is not explained by the released ProteinMPNN weights or by the exact
panel structures, but it is not robust to the training run at this scale.

## What this shows and does not show

- Shows: ECLS is architecture-general (M2); a controlled pre-2021 model
  reproduces it in 2/3 seeds (M3), but the from-scratch result is
  seed-dependent. M2 is the robust positive arm.
- Does not show: binding, affinity, specificity, or that homology-level
  exposure is absent.
- M3 is a small 1.7M model trained for 10 epochs; the magnitude (0.205) is
  similar to M0 (0.172), not a claim of superiority.
- No wet-lab measurement.

## Data

```text
results/ecls_cross_arch_v1/m3_full10/model_weights/m3.pt
results/ecls_cross_arch_v1/m3_full10/training_log.json
results/ecls_cross_arch_v1/m3_full10_ecls/results.json
results/ecls_cross_arch_v1/decision_m3_full10.json
scripts/build_m3_dataset_v1.py
scripts/train_m3_v1.py
configs/benchmarks/peptide_h3_ecls_m3_full10_v1.yml
```
