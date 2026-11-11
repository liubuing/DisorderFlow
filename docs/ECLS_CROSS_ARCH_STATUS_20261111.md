# ECLS cross-architecture + controlled-exposure status

Date: 2026-11-11. M0 is the frozen ProteinMPNN result. M2 (AntiFold
zero-shot) is complete. M3 (from-scratch pre-2021 model) has a pilot null and
a full training run in progress.

## Results so far

| Arm | Model | Mean ECLS advantage | 95% CI | Positive clusters | Verdict |
|---|---|---:|---|---:|---|
| M0 | ProteinMPNN v_48_020 (frozen) | 0.172281 | [0.059156, 0.291920] | 12/15 | reference |
| M2 | AntiFold zero-shot | **0.306928** | **[0.124085, 0.497105]** | **12/15** | **replication** |
| M3 pilot | from-scratch, 281 PDBs, 20 epochs | 0.015699 | [-0.113458, 0.151689] | 8/15 | pilot null |
| M3 full (early) | from-scratch, 4,506 PDBs, 3 epochs | 0.060258 | [-0.054864, 0.181916] | 7/15 | early null; undertrained |
| **M3 full (10 epochs)** | from-scratch, 4,506 PDBs, 10 epochs | **0.205317** | **[0.088715, 0.321356]** | **14/15** | **replication** |

M2 per-cluster comparison to M0: Pearson 0.720 (p=0.0025), Spearman 0.739
(p=0.0016), sign agreement 11/15.

## What M2 establishes

- ECLS is **architecture-general** at the frozen panel: an ESM-IF1/AntiFold
  scorer reproduces the direction, the positive-cluster proportion, and a
  correlated per-cluster effect.
- It does not control AntiFold's training exposure. M2 is a cross-architecture
  test, not the exposure test.

## What M3 pilot does and does not establish

- The pilot model learns (train 6.02 → 1.77; valid 1.48) but does not
  reproduce ECLS at 281 PDBs / 20 epochs.
- This is a **pipeline-validation null**, not a controlled-exposure verdict.
  The full 4,506-PDB run is the actual M3 arm.

## Decision tree for the full M3

| Full M3 outcome | Interpretation |
|---|---|
| Positive, CI lower > 0, ≥10/15 | ECLS survives a from-scratch pre-2021 model; not explained by the released ProteinMPNN weights |
| Partial / null | ECLS may require a well-trained structural prior; the exposure question remains unresolved, and the paper should report the scale dependence rather than claim exposure independence |

The 3-epoch checkpoint was undertrained and its null result is **not** a
controlled-exposure verdict. The completed 10-epoch full run is the actual M3
arm: it passes the frozen ECLS gates (mean 0.2053, CI lower 0.0887, 14/15
positive). Exact PDB overlap with the temporal panel is zero; one exact H3
substring overlap (7e6p_H vs 8h8q, cluster E121) was found, and excluding E121
leaves both M0 and M3 positive (M3 mean 0.2060, 13/14 positive).

## Paper framing (if the full M3 is null)

Title candidate: **Cross-architecture replication of an epitope-conditioned
likelihood contrast in antibody–peptide scoring.**

- M2 is the positive result: the contrast is not ProteinMPNN-specific.
- M3 is the boundary: a small from-scratch model does not inherit the effect,
  so the contrast depends on the quality/scale of the structural prior.
- This is a stronger, more honest methods paper than "ECLS replicates" alone.
- No binding, affinity or specificity claim.

## Claim boundary

- No wet-lab measurement.
- The frozen ECLS decision is unchanged.
- M2/M3 are different models; positive results generalize the contrast, not
  binding.
- The full M3 result must be read with its training scale and data cutoff.
