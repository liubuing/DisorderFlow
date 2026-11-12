# AGZArank head-to-head feasibility

Date: 2026-11-11. Update 2026-11-12: checkpoint access was requested through
a public GitHub issue,
`https://github.com/jenisbek-git/AGZArank/issues/1`; no response yet.

The closest preprint to the ECLS cross-architecture paper is AGZArank
(Sadykov et al., bioRxiv 2026, DOI 10.64898/2026.06.08.730711).

## What is public

- Code: `https://github.com/jenisbek-git/AGZArank` (cloned to
  `.tools/AGZArank`).
- Zenodo source data: `https://zenodo.org/records/20581998`, one file
  `AGZArank_data.zip` (1.34 GB), described as benchmark test data, source
  data, baseline inputs/outputs and reproduction files.
- Preprint: CC-BY-NC 4.0.

## What is not public

- The synthetic training dataset (proprietary Arlan Biotech structural data).
- Production model checkpoints (`checkpoints/agzarank_base.pt`). The README
  states that access "may be provided by Arlan Biotech upon reasonable
  request".

## Consequence

A true head-to-head (run AGZArank on the ECLS panel or on the same 45-interface
benchmark) is **not feasible locally** without the checkpoint. Training
AGZArank from scratch on the public test data would be invalid (the synthetic
training data are not released), and reproducing its architecture without its
supervision would not be AGZArank.

The paper should therefore:

1. Cite AGZArank as the closest learned-ranker work.
2. Compare **published** numbers only, with an explicit non-comparability
   statement: AGZArank uses structure-derived synthetic supervision on 45
   experimentally validated interfaces; our ECLS is a zero-shot paired
   likelihood contrast on a different panel.
3. State that a direct head-to-head requires author-provided weights.
4. Request the checkpoint before submission (filed 2026-11-12, issue #1).

## What was checked

- GitHub repository exists and contains training/inference/evaluation code.
- Zenodo record has no checkpoint; only the 1.34 GB source-data archive.
- README explicitly withholds the synthetic training data and production
  checkpoints.

## Status

**Blocked on external weights.** No numerical comparison is reported.
