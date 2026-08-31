# Candidate-Interface Confidence V1

## Status

`candidate_interface_v1` is an opt-in successor confidence architecture. It is
not enabled for legacy checkpoints and does not alter frozen BFN confidence
results. No trained or calibrated checkpoint is claimed yet.

## Architecture Contract

- pLDDT consumes candidate sequence embeddings plus candidate-to-antigen pair
  context.
- ipTM pools only the explicit candidate mask, rather than the full scaffold.
- PAE consumes residue identities and pair features directly.
- Fixed scoring requires nonempty candidate and antigen masks for every sample.
- Inference fails if a checkpoint declares this head but lacks any of its
  weights.
- Generated candidates are post-scored after the final sequence update. The
  reported PAE summary covers candidate-to-antigen pairs.

## Training Data Contract

A valid training record must contain the exact antibody-antigen complex used to
produce the target, an explicit candidate mask, an explicit antigen mask, and a
finite nonconstant full-complex PAE matrix. PAE normalization status must be
recorded explicitly and normalization is applied exactly once. PAE loss is
restricted to both candidate-to-antigen cross-chain directions.

The existing V14 confidence-regression records do not satisfy this contract:
they omit antigen coordinates, erase the candidate mask, and contain zero
`1 x 1` PAE placeholders. They must not be used to train this head.

## Evaluation Gate

Model selection must use scaffold-disjoint targets and report both between-
protein regression and within-scaffold variant discrimination. A successor is
not deployable unless it shows nontrivial within-scaffold output variance,
ranking correlation against held-out structural labels, and stability across
multiple antigen poses. Calibration and PPL may remain separate evidence
channels; they cannot substitute for this gate.
