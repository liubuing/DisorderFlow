# Controlled-replication base model v1 (proposal)

Status: **proposal, not executed.** This document specifies a bounded
experiment, not a new platform claim. It exists because the frozen ECLS result
uses ProteinMPNN v_48_020, whose training exposure cannot be fully excluded by
the post-2021 temporal filter.

## 1. Question

Does the ECLS native advantage survive when the scorer is trained under a
**known, frozen, pre-2021 exposure boundary**?

If yes, the ECLS result is not an artifact of the base model having seen the
temporal panel. If no, the frozen result is bounded as exposure-dependent.
Both outcomes are publishable; neither changes the frozen ECLS decision.

## 2. Design

Train or fine-tune an antibody-specific inverse-folding scorer on structures
deposited on or before **2021-08-02** (the ProteinMPNN training snapshot used
by the ECLS manuscript). Evaluate the same ECLS contrast on the frozen
post-2021 temporal panel (31 structures / 15 antigen clusters) with the same
shuffle protocol and seeds.

This is a **replication under controlled exposure**, not a new endpoint and
not a binding predictor.

## 3. Data sources

| Source | Content | Role | License / caveat |
|---|---|---|---|
| Local SAbDab2 archive | `data/sabdab2_current/splits.tar.gz`, 876 MB, MD5 recorded | Primary experimental structures; filter deposition date ≤ 2021-08-02 | CC BY 4.0 (dataset); cite source |
| SAbDab current snapshot | 21,142 summary rows | Curation and parent mapping | SAbDab terms; cite |
| AbMPNN dataset (`paulvstrashnov/ab-dataset`) | 151,308 QC-passing structures: 3,391 SAbDab experimental + 147,917 OAS modeled; train 121,058 / valid 15,212 / test 15,038 | Secondary arm only; useful if experimental SAbDab is too small for from-scratch training | Repository has no root license; derived from SAbDab/OAS. Verify terms before redistribution. OAS structures are **modeled**, not experimental |
| AntiFold (`oxpig/AntiFold`) | ESM-IF1 fine-tuned on SAbDab + OAS; BSD-3-Clause; model weights released | External antibody-specific baseline; zero-shot or fine-tune | BSD-3; cite paper |
| ProteinMPNN v_48_020 | 1.7M-parameter inverse-folding model; MIT | Frozen baseline and the model used by the ECLS manuscript | MIT |
| Antibody inverse-folding benchmark (`biomap-research/InverseFoldingEvaluation`) | PDB inputs and design outputs for AntiFold, ProteinMPNN, ESM-IF, LM-Design | Metric definitions and baseline comparison | No root license listed; use for method reference, not redistribution |

**Explicitly excluded:** AlphaFold DB predicted structures, AF2/AF3-generated
complexes, and any structure whose deposition date is after the cutoff. Using
predicted structures as training data would make the replication circular.

The HuggingFace dataset API was unreachable during this survey (2026-11-11).
Candidate HF datasets should be re-checked before the data freeze; the
GitHub-hosted AbMPNN dataset is sufficient to start.

## 4. Splits and leakage control

1. **Temporal split**: train/validation ≤ 2021-08-02; evaluation panel is the
   frozen post-2021 ECLS panel.
2. **Cluster split**: cluster training chains at 50% sequence identity
   (MMseqs2 or CD-HIT); no cluster may cross train/validation.
3. **Panel exclusion**: remove any training structure that shares an exact H3,
   peptide, or PDB entry with the 31-structure temporal panel.
4. **Antibody-only**: train on antibody variable domains; do not mix general
   protein chains unless the arm is explicitly labelled.
5. **Manifest**: record every input file with SHA-256; freeze the manifest
   before training.

## 5. Model arms

| Arm | Model | Training | Purpose |
|---|---|---|---|
| M0 | ProteinMPNN v_48_020 | none (frozen) | Reproduce the frozen ECLS number |
| M1 | ProteinMPNN architecture | fine-tune on pre-2021 SAbDab | Cheapest controlled replication |
| M2 | AntiFold / ESM-IF1 | zero-shot, then optional fine-tune | Antibody-specific external baseline |
| M3 | ProteinMPNN or ESM-IF1 | from scratch on AbMPNN SAbDab subset (3,391) | Tests whether 3.4k experimental structures suffice |
| M4 | M3 + OAS modeled structures | from scratch on 151k | Tests whether modeled data changes the result; secondary, disclose the bias |

Recommended order: **M0 → M1 → M2 → M3 → M4.** Do not start M4 until M1/M3
show a stable effect.

## 6. Training details (M1/M3)

- Architecture: ProteinMPNN-style 3 encoder + 3 decoder layers, ~1.7M
  parameters; hidden width 128; k-nearest-neighbor graph (k=48).
- Objective: masked residue cross-entropy on antibody CDR and framework
  positions; random masking order.
- Optimizer: Adam, lr 1e-4, weight decay 1e-5, 10–20 epochs, batch 32.
- Hardware: one 24 GB GPU; M1 ≈ 1–3 GPU-days, M3 ≈ 3–7 GPU-days.
- Checkpoints: save best validation recovery and last; record SHA-256.
- Determinism: fixed seeds; log git commit and config hash.

## 7. Evaluation

Primary endpoint (identical to the frozen ECLS protocol):

```
ECLS(s) = NLL_complex(s) - NLL_peptide_stripped(s)
advantage = mean ECLS(composition-matched shuffle) - ECLS(native)
```

- Same 31 structures / 15 antigen clusters; same up-to-200 shuffles; same
  seeds; same cluster bootstrap (10,000 resamples).
- Report mean, median, 95% CI, positive clusters, and per-cluster effects.
- Secondary: native recovery on the FLEX C1 pools; AAYL binding Spearman
  (expected null); runtime.

## 8. Decision rules (preregistered)

| Outcome | Rule | Interpretation |
|---|---|---|
| Replication | Mean advantage > 0, 95% CI lower > 0, ≥ 10/15 clusters positive, magnitude within 2× of 0.172 | ECLS is not explained by ProteinMPNN exposure |
| Partial | Direction positive but CI includes 0, or magnitude < 0.5× | Suggestive; needs a larger panel, not a new claim |
| Null | CI includes 0 or sign flips | Bound the frozen result as exposure-dependent |
| Binding surprise | AAYL Spearman > 0.2 with CI lower > 0 | Requires independent confirmation before any claim |

No outcome authorizes a binding or affinity claim. No outcome changes the
frozen ECLS decision.

## 9. Preregistration and contracts

Before any training or scoring:

1. Freeze this protocol with a SHA-256 and a `preregistration.json`.
2. Freeze the data manifest, split assignment, model config and evaluation
   script hashes.
3. Add contract tests: temporal cutoff, no panel overlap, shuffle
   composition preservation, deterministic seeds, output schema.
4. Write results to a new `results/ecls_controlled_replication_v1/` directory;
   never overwrite the frozen ECLS artifacts.

## 10. Budget

| Item | Estimate |
|---|---|
| Data curation and split | 2–4 weeks |
| M1 training and evaluation | 1 week |
| M2 zero-shot/fine-tune | 3–5 days |
| M3/M4 (optional) | 1–2 weeks |
| Total | 5–9 weeks, one GPU |

## 11. Risks

- **OAS modeled structures** are not experimental and may bake in a different
  model's biases; keep M4 secondary.
- **Temporal leakage** is reduced but not eliminated by the cutoff; disclose.
- **Antibody bias**: SAbDab is not a random sample of antibody space.
- **License**: AbMPNN repo has no root license; verify before redistributing
  derived data. AntiFold is BSD-3; ProteinMPNN is MIT.
- **Compute**: M3/M4 may exceed one GPU-week; gate on M1 results.

## 12. Deliverables

- Frozen preregistration and data manifest.
- Trained checkpoint(s) with SHA-256.
- Per-cluster ECLS table and bootstrap summary.
- A short report with the replication/null decision and the claim boundary.
- A reproducibility command list.
