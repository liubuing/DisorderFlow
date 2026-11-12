## Context

I maintain [DisorderFlow](https://github.com/liubuing/DisorderFlow), an open
computational study of inverse-folding scores for antibody–peptide complexes.
We are preparing an independent, non-commercial academic benchmark of a paired
complex-minus-stripped likelihood contrast (ECLS) across inverse-folding
models.

AGZArank is the closest learned-ranker work to our study. We would like to
include it as a baseline in that benchmark rather than compare only to the
published numbers, because the training supervision, candidate pool and
scoring target differ from ours.

## Request

Would it be possible to obtain, for research use:

- `checkpoints/agzarank_base.pt` (the inference checkpoint);
- the inference config or preprocessing contract needed to reproduce the
  published ranking;
- optionally, the exact evaluation split and candidate-pool definition.

## Intended use and commitments

- Non-commercial academic evaluation only.
- No redistribution of the checkpoint.
- No attempt to reverse-engineer the proprietary synthetic training data.
- Citation of the AGZArank paper, with the model clearly described as your
  work.
- Sharing the comparison results with the authors before any manuscript
  submission.
- Willingness to sign a data-use or MTA agreement if required.

## Fallback

If the checkpoint cannot be shared, even a short description of the expected
inference input format and the published evaluation split would help us avoid
a misleading comparison against published numbers.

Thank you for considering this request.
