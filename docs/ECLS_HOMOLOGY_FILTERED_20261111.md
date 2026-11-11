# ECLS homology-filtered M3 training

Date: 2026-11-11. The full M3 training set was filtered to remove structures
related to the frozen temporal panel, then retrained from scratch for 10
epochs and evaluated on the same panel.

## Exclusion rule

A training PDB was excluded if either:

1. a training chain contained a panel H3 exact or 1-mismatch window; or
2. the full-VH 3-mer cosine similarity to any panel VH was ≥0.75.

Result: **36 of 4,506 PDBs excluded** (35 by VH cosine, 1 by exact H3:
`7e6p` / `ALFDGYYPWFAS`).

The 0.75 threshold corresponds to the top ~1% of the VH-cosine distribution
(median 0.407, p99 0.742). It is a 3-mer proxy, not a full alignment.

## Training

- Data: 4,470 PDBs after exclusion.
- Model: ProteinMPNN-style 3+3, hidden 128, k=48, ~1.7M parameters.
- 10 epochs, max chain length 300.
- Train loss 4.29 → 1.41; validation 1.14.

## ECLS result

| Statistic | Value |
|---|---:|
| Mean ECLS advantage | **0.188379** |
| Median | 0.204988 |
| 95% cluster-bootstrap CI | **[0.076693, 0.298913]** |
| Positive clusters | 11/15 (0.733) |
| Gate | **pass** |

Comparison:

| Model | Mean | 95% CI | Positive |
|---|---:|---|---:|
| M3 all pre-2021 | 0.205317 | [0.088715, 0.321356] | 14/15 |
| **M3 homology-filtered** | **0.188379** | **[0.076693, 0.298913]** | **11/15** |

The homology filter lowers the mean slightly (0.205 → 0.188) but the effect
remains positive with a CI above zero and passes the frozen gates.

## Interpretation

- The ECLS result is not driven by the 36 removed structures, including the
  single exact H3 overlap.
- Homology-level exposure is reduced but not eliminated: the filter uses a
  3-mer VH proxy and does not remove all framework homologs or H3 sequences
  with >1 mismatch.
- The result strengthens the controlled-exposure claim but does not make it
  absolute.

## Data

```text
results/ecls_factorial_v1/homology_exclusions.json
results/ecls_cross_arch_v1/m3_homology_filtered/training_log.json
results/ecls_cross_arch_v1/m3_homology_ecls/results.json
scripts/build_m3_homology_exclusions_v1.py
```
