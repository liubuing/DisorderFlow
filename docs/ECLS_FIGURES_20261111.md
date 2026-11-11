# ECLS ablation figures

Date: 2026-11-11. Four figures for the cross-architecture/controlled-exposure
manuscript. Files are local under `publication/figures/` (gitignored); the
script is tracked.

| Figure | File | Content |
|---|---|---|
| 1 | `ecls_fig1_context_identity.png` / `.pdf` | Peptide presence is read; peptide identity is not (M0/M2/M3) |
| 2 | `ecls_fig2_scale_curve.png` / `.pdf` | ECLS vs training epoch (1/3/5/10/20); pass/fail |
| 3 | `ecls_fig3_task_ranker.png` / `.pdf` | AAYL pairwise ranker: ECLS vs composition vs both |
| 4 | `ecls_fig4_cluster_agreement.png` / `.pdf` | Per-antigen-cluster ECLS agreement (M0/M2/M3) |

Regenerate:

```bash
python scripts/make_ecls_ablation_figures_v1.py
```

Data sources:

- `results/ecls_factorial_v1/summary.json`
- `results/ecls_factorial_v1_m2/summary.json`
- `results/ecls_cross_arch_v1/m3_scale_eval/scale_curve.json`
- `results/ecls_factorial_v1/task_ranker.json`
- `results/publication/h3_ecls_temporal_final_v1/results.json`
- `results/ecls_cross_arch_v1/antifold/antifold_summary.json`
- `results/ecls_cross_arch_v1/m3_full10_ecls/results.json`
