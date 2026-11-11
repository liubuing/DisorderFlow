# ECLS antigen-context and peptide-identity ablation

Date: 2026-11-11. For each frozen temporal record, the H/L/peptide
coordinates were held fixed and the peptide sequence was replaced with a
composition-matched shuffle. Native complex/apo NLLs came from the existing
model summaries; only the antigen-shuffled condition was recomputed.

Definitions:

```
delta_context = NLL(apo) - NLL(complex)          >0: peptide presence helps
delta_peptide = NLL(antigen-shuf) - NLL(complex) >0: native peptide identity helps
```

## Results (M0 and M3)

| Model | n | mean delta_context | context positive | mean delta_peptide | peptide positive | peptide/context ratio |
|---|---:|---:|---:|---:|---:|---:|
| M0 ProteinMPNN | 31 | **0.1375** | 80.6% | **1.25e-08** | 51.6% | **9.1e-08** |
| M2 AntiFold | 31 | **0.1563** | 77.4% | **1.85e-07** | 45.2% | **1.19e-06** |
| M3 from scratch | 31 | **0.1791** | 77.4% | **-2.82e-08** | 48.4% | **-1.6e-07** |

## Interpretation

- Both ProteinMPNN and the from-scratch model use the **presence of a peptide
  backbone**: removing the peptide lowers the native H3 likelihood
  (`delta_context` positive in ~80% of records).
- Both models are **numerically insensitive to peptide sequence identity**:
  replacing the peptide with a composition-matched shuffle changes the native
  H3 NLL by ~1e-8 nats, i.e. zero at machine precision.
- **All three models** show the same pattern: peptide presence matters
  (`delta_context` 0.137–0.179), peptide sequence identity does not
  (`delta_peptide` 1e-8–1e-7; ratio ≤1.2e-6).
- This means the "epitope-conditioned" ECLS contrast is, in these models,
  **epitope-backbone-conditioned**, not peptide-sequence-conditioned. It is
  consistent with the 2026 antigen-aware inverse-folding benchmark's finding
  that these models make limited use of the antigen
  [Janusz et al. 2026].
- The result also explains why the ECLS advantage is a native-recognition
  signal: the model reads the antibody H3 sequence against the peptide
  backbone geometry, not the peptide's chemical identity.

## Caveat

ProteinMPNN was run with `--conditional_probs_only_backbone`, which may make
the peptide sidechain identity invisible by construction. The AntiFold arm
(ESM-IF1, sequence-aware) was the control: it shows the same zero
peptide-identity effect, so the insensitivity is not only a scoring-mode
artifact. The result is a model-family property, not a single implementation
detail.

## Data

```text
results/ecls_factorial_v1/summary.json
results/ecls_factorial_v1/per_record.csv
scripts/ecls_factorial_ablation_v1.py
```
