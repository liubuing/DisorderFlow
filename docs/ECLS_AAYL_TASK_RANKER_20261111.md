# AAYL task-specific pairwise ranker vs ECLS

Date: 2026-11-11. Within each AAYL family, a logistic pairwise ranker was
trained on 5-fold CV using three feature sets: ECLS alone, 20-dimensional
composition alone, and ECLS + composition. Reported values are held-out
Spearman with the AlphaSeq-estimated Kd endpoint (higher = better ranking;
desired positive).

## Results

| Family | Features | Mean held-out Spearman | SD |
|---|---|---:|---:|
| AAYL49 (n=281) | ECLS | -0.047 | 0.080 |
| AAYL49 | composition | 0.027 | 0.048 |
| AAYL49 | ECLS + composition | 0.022 | 0.051 |
| AAYL51 (n=227) | ECLS | -0.141 | 0.128 |
| AAYL51 | composition | 0.187 | 0.130 |
| AAYL51 | ECLS + composition | **0.198** | 0.125 |

## Interpretation

- A task-specific ranker trained on composition recovers a weak positive
  ranking in AAYL51 (0.187) and essentially nothing in AAYL49 (0.027).
- ECLS alone is negative in both families (-0.047, -0.141).
- Adding ECLS to composition changes the AAYL51 mean from 0.187 to 0.198
  (+0.011) and the AAYL49 mean from 0.027 to 0.022 (-0.005). ECLS adds no
  meaningful ranking signal.
- This is consistent with the zero-shot AAYL result (ECLS Spearman +0.009 /
  -0.002) and with the 2026 antigen-aware benchmark's weak-affinity finding
  [Janusz et al. 2026]. Task-specific training can extract a little
  composition-driven signal; the ECLS contrast is not that signal.

## Caveat

Within-family 5-fold CV, small n, AlphaSeq-estimated Kd, two parent lineages.
This is a development analysis, not a new ranker claim.

## Data

```text
results/ecls_factorial_v1/task_ranker.json
scripts/ecls_aayl_task_ranker_v1.py
```
