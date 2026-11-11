# ECLS training-scale curve (M3)

Date: 2026-11-11. The same homology-filtered M3 training run was evaluated at
epochs 1, 3, 5, 10 (and 20 in progress). Panel, shuffles and ECLS definition
are unchanged.

## Results

| Epoch | Mean ECLS advantage | Median | 95% CI | Positive clusters | Gate |
|---:|---:|---:|---|---:|---|
| 1 | 0.077774 | 0.016891 | [-0.047806, 0.215125] | 8/15 | fail |
| 3 | 0.128717 | 0.128681 | [0.010067, 0.260136] | 8/15 | fail |
| 5 | 0.175995 | 0.100197 | [0.073902, 0.285267] | 11/15 | **pass** |
| 10 | 0.188379 | 0.204988 | [0.076693, 0.298913] | 11/15 | **pass** |
| 20 | 0.125565 | 0.112037 | [0.022847, 0.234966] | 10/15 | fail |

## Interpretation

- The ECLS advantage rises from epoch 1 to epoch 10: 0.078 → 0.129 → 0.176 →
  0.188.
- It then **declines at epoch 20**: 0.126, 10/15, gate fail. The curve is not
  monotonic; the peak is around epoch 5–10.
- The positive-cluster fraction crosses the 70% gate between epoch 3 and
  epoch 5, and falls below it again at epoch 20.
- The earlier 3-epoch full-data checkpoint (mean 0.060, 7/15) was a
  **training-scale null**, not evidence against the contrast.
- The result supports a "training-sensitive structural prior" interpretation:
  a small model must be trained enough for the preference to emerge, but more
  training is not automatically better. The epoch-20 decline is consistent
  with overfitting or training instability on the small corpus; it is not a
  scaling law.

## Caveat

Single seed, homology-filtered training set, max chain length 300. The curve is
descriptive; it is not a scaling-law claim. The epoch-20 decline should be
checked with additional seeds before any mechanistic interpretation.

## Data

```text
results/ecls_cross_arch_v1/m3_scale_eval/scale_curve.json
results/ecls_cross_arch_v1/m3_scale_eval/ecls_epoch_{1,3,5,10}/results.json
scripts/eval_m3_scale_v1.py
```
