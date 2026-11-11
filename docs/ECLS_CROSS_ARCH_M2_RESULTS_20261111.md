# ECLS cross-architecture result (M2, AntiFold zero-shot)

Status: **M2 complete; replication criterion met.** M3 (from-scratch
pre-2021 model) remains pending. The frozen ECLS decision is unchanged.

## Method

- Panel: the frozen temporal final, 31 structures / 15 antigen clusters
  (`selected_ids_sha256 = 7cbc63b7…1821cf`).
- Model: AntiFold (ESM-IF1 fine-tuned on SAbDab+OAS), zero-shot.
- Scoring: the same ECLS definition and the same complex/apo PDB construction
  as the frozen ProteinMPNN analysis. AntiFold per-residue log-probabilities
  at the frozen H3 positions replace ProteinMPNN's `log_p`.
- Shuffles: up to 200 composition-matched shuffles, seed 2309 + sum(ord(id)).
- Preregistration: `docs/ECLS_CROSS_ARCH_PREREG_V1.md`,
  `results/ecls_cross_arch_v1/preregistration.json`.

## Results

| Statistic | ProteinMPNN (M0) | AntiFold (M2) |
|---|---:|---:|
| Mean ECLS advantage | 0.172281 | **0.306928** |
| Median ECLS advantage | 0.118699 | **0.308123** |
| Cluster-bootstrap 95% CI | [0.059156, 0.291920] | **[0.124085, 0.497105]** |
| Positive clusters | 12/15 (0.8) | **12/15 (0.8)** |
| Magnitude ratio vs M0 | 1.00 | **1.78×** |

Per-cluster comparison (mean advantage):

- Pearson **0.720** (p = 0.0025), Spearman **0.739** (p = 0.0016).
- Sign agreement **11/15 (73%)**.
- ProteinMPNN-negative clusters: E149, E181, E214.
- AntiFold-negative clusters: E214, E27, E91.
- Only E214 is negative in both models.

## Preregistered decision

The replication rule required: mean > 0, 95% CI lower > 0, ≥10/15 positive
clusters, magnitude within 2× of 0.172.

| Criterion | AntiFold | Pass |
|---|---|---|
| Mean > 0 | 0.3069 | yes |
| CI lower > 0 | 0.1241 | yes |
| Positive clusters ≥10/15 | 12/15 | yes |
| Magnitude within 2× | 1.78× | yes |

**Verdict: `replication`.** The ECLS native-versus-shuffle advantage is not
specific to ProteinMPNN: a different inverse-folding architecture
(ESM-IF1/AntiFold) reproduces the direction, the positive-cluster proportion,
and a correlated per-cluster effect.

## What this does and does not show

- It shows that ECLS is **architecture-general** at the level of the frozen
  panel.
- It does **not** control AntiFold's training exposure: AntiFold was
  fine-tuned on SAbDab + OAS, and its exposure to the temporal panel is not
  documented. This is a cross-architecture test, not the controlled-exposure
  test.
- M3 (from scratch on pre-2021 SAbDab only) is the controlled-exposure arm.
- No binding, affinity or specificity claim is made.

## M3 status

A **pilot** from-scratch model (281 PDBs / 557 chains, 20 epochs) did not
reproduce the effect: mean 0.0157, CI [-0.1135, 0.1517], 8/15 positive. This is
a pipeline-validation null, not the controlled-exposure verdict; the full
4,506-PDB training is the actual M3 arm and is running separately. See
`docs/ECLS_CROSS_ARCH_M3_PILOT_20261111.md`.

## Data

```text
results/ecls_cross_arch_v1/antifold/antifold_results.json   # per-record scores
results/ecls_cross_arch_v1/antifold/antifold_summary.json   # aggregate
results/ecls_cross_arch_v1/antifold/work/                   # complex/apo PDBs + CSVs
scripts/ecls_antifold_zero_shot_v1.py
```
