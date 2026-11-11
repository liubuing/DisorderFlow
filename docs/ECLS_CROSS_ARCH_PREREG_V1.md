# ECLS cross-architecture and controlled-exposure preregistration v1

Status: **frozen_before_scoring**, 2026-11-11. Machine-readable copy:
`results/ecls_cross_arch_v1/preregistration.json`.

## Question

Does the ECLS native-versus-shuffle advantage survive (a) a different
inverse-folding architecture (AntiFold/ESM-IF1) and (b) a model trained from
scratch on pre-2021 antibody structures only?

The frozen ECLS result used ProteinMPNN v_48_020. This experiment tests
whether the effect is specific to that model or to its training exposure.

## Arms

| Arm | Model | Training | Role |
|---|---|---|---|
| M0 | ProteinMPNN v_48_020 | released weights | Frozen reference (0.172281) |
| M2 | AntiFold (ESM-IF1 fine-tuned on SAbDab+OAS) | zero-shot | Cross-architecture test |
| M3 | ProteinMPNN-style 3+3 layer model, ~1.7M params | from scratch on pre-2021 SAbDab experimental structures | Controlled-exposure test |

M2 tests architecture generality. M3 tests whether the effect survives a model
that cannot have seen the post-2021 temporal panel.

## Evaluation panel

The frozen temporal final: 31 structures / 15 official antigen clusters
(`results/publication/h3_ecls_temporal_final_v1/final_decision.json`,
`selected_ids_sha256 = 7cbc63b7afb42dbdc13da9469f99ab7e927fc5b3f14f500713705869da1821cf`).
The panel is terminal and is not re-selected.

## ECLS definition

```
ECLS(s) = NLL_complex(s) - NLL_peptide_stripped(s)
advantage = mean ECLS(composition-matched shuffle) - ECLS(native H3)
```

Up to 200 composition-matched shuffles per record, using the frozen shuffle
protocol and seed (2309 + sum of record-id characters). H3 positions are the
frozen `parsed_h3_indices` from the audit. All non-H3 positions are fixed.

For M2, the AntiFold log-probability matrix at H3 positions is used in place
of ProteinMPNN's `log_p`; the complex/apo PDB construction is identical. For
M3, the same evaluation code is used with the M3 checkpoint.

## Decision rules

| Outcome | Rule | Interpretation |
|---|---|---|
| Replication | mean advantage > 0, 95% cluster-bootstrap CI lower > 0, ≥10/15 positive clusters, magnitude within 2× of 0.172 | ECLS is not specific to ProteinMPNN |
| Partial | positive direction but CI includes 0, or magnitude < 0.5× | Suggestive; needs a larger panel |
| Null | CI includes 0 or sign flips | ECLS is architecture- or exposure-dependent |
| Binding surprise | AAYL Spearman > 0.2 with CI lower > 0 | Requires independent confirmation before any binding claim |

The primary replication criterion is the same as the frozen ECLS gates, with
the magnitude clause added for cross-model comparison.

## M3 training contract (frozen before training)

- Data: pre-2021 SAbDab experimental structures only (deposition date ≤
  2021-08-02); no OAS modeled structures, no AlphaFold DB predictions.
- Split: 90/10 by PDB-level cluster; no temporal-panel PDB or exact H3 overlap
  in training.
- Model: ProteinMPNN-style encoder/decoder (3+3 layers, hidden 128, k=48).
- Objective: masked residue cross-entropy with random decoding order.
- Optimizer: Adam, lr 1e-4, batch 32, 10–20 epochs; checkpoint by validation
  recovery.
- No checkpoint selection on the temporal panel; the checkpoint is chosen on
  the pre-2021 validation split only.

## Claim boundary

- Tests a likelihood contrast, not binding, affinity or specificity.
- AntiFold and M3 are different models; a positive result generalizes the ECLS
  contrast, not the binding claim.
- No wet-lab measurement is reported.
- The frozen ECLS decision is unchanged regardless of the outcome.

## Frozen inputs

```text
temporal final IDs   7cbc63b7afb42dbdc13da9469f99ab7e927fc5b3f14f500713705869da1821cf
temporal config      configs/benchmarks/peptide_h3_ecls_temporal_final_v1.yml
frozen advantage     0.172281, CI [0.059156, 0.291920], 12/15 positive
shuffle trials       200
shuffle seed         2309 + sum(ord(id))
```
