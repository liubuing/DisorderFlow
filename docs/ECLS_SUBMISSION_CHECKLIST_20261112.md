# ECLS cross-architecture manuscript: submission checklist

Date: 2026-11-12. Local manuscript:
`publication/ecls_cross_arch_manuscript_draft_v1.md`.
Cover letter: `publication/ecls_cross_arch_cover_letter.md`.
Figures: `publication/figures/ecls_fig1..4_*.png/.pdf`.

## Scientific content

| Item | Status |
|---|---|
| M0 ProteinMPNN reference | frozen, 0.1723 [0.0592, 0.2919], 12/15 |
| M2 AntiFold cross-architecture | complete, 0.3069 [0.1241, 0.4971], 12/15 |
| M3 from-scratch, 3 seeds | complete; 0.188/0.200/0.013; 2/3 pass, 1 null |
| Antigen-context/peptide-identity ablation | complete; identity effect ~1e-8 in M0/M2/M3 |
| Homology-filtered M3 | complete; 0.1884 [0.0767, 0.2989], 11/15 |
| AAYL affinity endpoint | complete; +0.0094 / −0.0016, CIs span zero |
| Task-specific ranker | complete; composition weak, ECLS adds no signal |
| AGZArank head-to-head | blocked: checkpoint and synthetic data proprietary |
| Epoch-20 seeds 2/3 | not completed; blocked by external GPU contention; optional robustness point |

## Figures

- Fig 1: context vs identity (M0/M2/M3).
- Fig 2: three-seed scale curve; seed 3 null.
- Fig 3: AAYL task-specific ranker.
- Fig 4: per-antigen-cluster agreement.

All four are generated at 300 dpi with PDF counterparts.

## Manuscript structure

- Title, author placeholders, companion disclosure.
- Abstract with the 2/3 seed result.
- Introduction with Related Work (AGZArank, Janusz 2026, SimBinder-IF/ProteinDPO).
- Results 2.1–2.6.
- Discussion and Limitations (seed sensitivity, homology proxy, no wet lab).
- Methods 5.1–5.4.
- Figures 1–4 legends.
- Data availability.
- References with DOIs.
- TODO list.

## Pending before submission

1. Author list, affiliations, corresponding email, CRediT.
2. Funding, conflicts of interest, acknowledgements.
3. Manuscript/data license confirmation.
4. Journal choice and formatting: *Bioinformatics* (primary) or *PLOS
   Computational Biology* (alternative).
5. Supplementary tables: per-record scores, seed summary, homology exclusion
   list, AAYL per-family scores.
6. Optional: epoch-20 seeds 2/3; AGZArank checkpoint request.
7. Companion-manuscript disclosure wording checked against journal policy.

## Claim boundary (must remain)

- ECLS is a likelihood contrast, not binding/affinity/specificity.
- M2 is the robust cross-architecture result; M3 is partial and
  seed-dependent.
- Peptide identity is not read; peptide backbone presence is.
- No wet-lab measurement.
