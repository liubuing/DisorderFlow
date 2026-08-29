# iBEC E1 Generator Behavior Audit

## Frozen Panel

- 20 independent antibody-peptide components.
- 2400 raw generation attempts; 2400 successful.
- Three seeds and eight attempts per component/arm.
- Self-perplexity values are not compared across model families.

## Primary Comparison

Decision: **expanded_exploration_with_developability_tradeoff**

BFN expanded H3 sequence exploration but showed a developability tradeoff.

| Metric | BFN mean | ProteinMPNN mean | Paired difference | 95% CI |
|---|---:|---:|---:|---:|
| unique_fraction | 0.8333 | 0.7500 | 0.0833 | [-0.0083, 0.1896] |
| mean_position_entropy_nats | 2.4052 | 0.3465 | 2.0587 | [2.0086, 2.1129] |
| mean_pairwise_hamming_fraction | 0.9344 | 0.2169 | 0.7175 | [0.6869, 0.7502] |
| developability_pass_fraction_at_0_55 | 0.8250 | 0.9896 | -0.1646 | [-0.2479, -0.0875] |
| mean_full_heavy_developability_risk | 0.4604 | 0.4210 | 0.0394 | [0.0118, 0.0639] |
| mean_native_recovery | 0.0843 | 0.3625 | -0.2782 | [-0.3300, -0.2300] |
| unique_fraction_seed_std | 0.0000 | 0.0499 | -0.0499 | [-0.0727, -0.0300] |
| position_entropy_seed_std | 0.0190 | 0.0357 | -0.0168 | [-0.0272, -0.0068] |

## Boundary

Diversity is a generator-distribution property, not evidence of binding, affinity, or biological quality. The original phrase 'higher diversity and lower perplexity' must be replaced by the allowed claim above.
