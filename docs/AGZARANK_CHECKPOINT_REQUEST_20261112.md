# AGZArank checkpoint request (draft)

Date: 2026-11-12. Status: **draft, not sent.** The public repository gives the
corresponding commit author as `zhenisbek@arlanbiotech.com`; the Zenodo
creator is Dauken Seitkali (Arlan Biotech Inc).

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

## Email draft

**To:** zhenisbek@arlanbiotech.com
**Subject:** Request for AGZArank checkpoint access for an independent academic benchmark

Dear Dr. Sadykov and colleagues,

I am [Name], [position] at [institution]. I work on computational antibody
design and on the evaluation of inverse-folding scores for antibody–peptide
complexes (project: DisorderFlow).

Your preprint "AGZArank: Investigating epitope-conditioned antibody binder
ranking with structure-derived synthetic supervision" is the closest work to
our current study. We are preparing an independent, non-commercial benchmark
of a paired complex-minus-stripped likelihood contrast (ECLS) across
inverse-folding models, and we would like to include AGZArank as a learned
ranker baseline rather than compare only to published numbers.

I saw that the code is available at github.com/jenisbek-git/AGZArank and that
the checkpoint and synthetic training data are not public. Would it be
possible to obtain the inference checkpoint (`agzarank_base.pt`) and the
inference configuration for research use? We would:

- use it only for non-commercial academic evaluation;
- not redistribute the checkpoint;
- cite your paper and describe the model as your work;
- share the comparison results with you before any submission;
- sign a data-use or MTA agreement if required.

If the full checkpoint cannot be shared, even a short description of the
expected inference input format and the published evaluation split would help
us avoid a misleading comparison.

Thank you for considering this request.

Best regards,
[Name]
[Position, institution]
[Email]
[Project URL]

## Alternative channel

If email is not appropriate, open a GitHub issue on
`jenisbek-git/AGZArank` titled "Request for inference checkpoint for academic
benchmark" with the same text, or use the Zenodo record contact. Do not post
the request publicly without the author's consent if it contains personal
contact details.

## What to do with the checkpoint if received

1. Record SHA-256 and the license/DUA terms.
2. Run inference on the frozen panel with the published config.
3. Do not tune on the ECLS panel; freeze predictions before comparison.
4. Report both AGZArank and ECLS numbers, with the non-comparability caveat
   (different training supervision, different candidate pools).
