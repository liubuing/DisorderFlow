# ECLS vs AAYL affinity endpoint (M3 model)

Status: **descriptive retrospective; null result.** The M3 model reproduces
the ECLS native-versus-shuffle contrast, but its ECLS does not rank the
AlphaSeq-estimated Kd endpoint.

## Method

- Cohort: 508 H3-only AAYL variants (AAYL49 n=281; AAYL51 n=227), two parent
  lineages, one antigen.
- Endpoint: `9 - median(log10 estimated Kd in nM)`; higher = stronger.
  Desired direction: Spearman(ECLS, endpoint) < 0.
- Structures: parent AAYL complex/apo backbones; each variant H3 sequence was
  injected at the parent H3 positions.
- Scorer: the M3 from-scratch checkpoint (pre-2021 SAbDab, 10 epochs).
- ECLS: H3 NLL(complex) - H3 NLL(apo), computed from ProteinMPNN conditional
  probabilities at the H3 positions.
- Bootstrap: 10,000 draws, seed 20261111.

## Results

| Family | n | Spearman(ECLS, endpoint) | 95% bootstrap CI |
|---|---:|---:|---|
| AAYL49 | 281 | **+0.0094** | [-0.1114, 0.1277] |
| AAYL51 | 227 | **-0.0016** | [-0.1373, 0.1310] |

Both intervals span zero; neither family shows the desired negative
direction. The effect is effectively zero, not a weak positive.

Comparison with previous scorers on the same retrospective:

| Scorer | AAYL49 | AAYL51 |
|---|---:|---:|
| Frozen ProteinMPNN ECLS | -0.1214 | -0.0715 |
| BFN corrected-role ECLS | +0.0621 | -0.0489 |
| **M3 from scratch** | **+0.0094** | **-0.0016** |

All three are near zero or inconsistent in sign; none supports an affinity
ranking claim.

## Interpretation

The ECLS contrast and affinity ranking are **dissociable**:

- ECLS replicates across architectures and under controlled pre-2021 training
  (M2/M3).
- On 508 variants whose binding spans a measurable range, M3 ECLS is
  uncorrelated with the estimated Kd.
- The contrast is therefore a structural native-recognition signal, not a
  binding or candidate-ranking score.

## Caveats

- The endpoint is AlphaSeq-estimated Kd, not SPR/BLI/ITC.
- The structures are AF3-predicted parent complexes, not experimental variant
  structures.
- One antigen, two parent lineages; the retrospective is unblinded.
- No claim of binding, affinity or specificity is made.

## Data

```text
results/ecls_cross_arch_v1/aayl_m3/scores.json
results/ecls_cross_arch_v1/aayl_m3/report.json
scripts/ecls_aayl_m3_v1.py
```
