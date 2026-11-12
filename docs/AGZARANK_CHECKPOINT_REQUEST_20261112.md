# AGZArank checkpoint request (GitHub issue)

Date: 2026-11-12. Status: **sent** as a public GitHub issue:
https://github.com/jenisbek-git/AGZArank/issues/1

The issue text is kept in
`docs/AGZARANK_CHECKPOINT_ISSUE_20261112.md`. It contains no personal contact
details; the public issue is the contact channel. The earlier email draft
(`publication/AGZARANK_CHECKPOINT_REQUEST_20261112.eml`) is superseded and is
not part of the submission.

## Why request

AGZArank (Sadykov et al., bioRxiv 2026,
DOI 10.64898/2026.06.08.730711) is the closest learned-ranker work to our
cross-architecture ECLS study. The code is public, but the README states that
the synthetic training data and the production checkpoint
(`checkpoints/agzarank_base.pt`) are not publicly released. A direct
head-to-head is therefore blocked on checkpoint access.

## Requested material

- `agzarank_base.pt` (inference checkpoint).
- The inference config or any preprocessing contract needed to reproduce the
  published ranking on the 45 experimentally validated interfaces.
- Optional: the exact evaluation split and candidate-pool definition.

## Intended use

- Non-commercial academic benchmark.
- Evaluate AGZArank on our frozen 31-structure / 15-antigen-cluster panel and,
  if compatible, on the published 45-interface benchmark.
- No redistribution of the checkpoint; no attempt to reverse-engineer the
  proprietary training data.
- Cite the AGZArank paper in any resulting manuscript and share the
  comparison results with the authors before submission.
- Willing to sign a data-use/MTA agreement if required.

## Fallback

If the checkpoint cannot be shared, ask for the expected inference input
format and the published evaluation split so that any comparison against
published numbers is described correctly.

## What to do with the checkpoint if received

1. Record SHA-256 and the license/DUA terms.
2. Run inference on the frozen panel with the published config.
3. Do not tune on the ECLS panel; freeze predictions before comparison.
4. Report both AGZArank and ECLS numbers, with the non-comparability caveat
   (different training supervision, different candidate pools).
