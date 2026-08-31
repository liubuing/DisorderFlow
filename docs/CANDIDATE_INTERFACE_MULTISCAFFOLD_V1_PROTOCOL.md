# Candidate-Interface Multiscaffold V1 Protocol

## Status

Development-only successor lineage. This protocol does not alter or supersede
the frozen `multiscaffold_confirmatory_v2` holdout or its reported results.

## Frozen split

The split is defined by
`data/candidate_interface_multiscaffold_v1/split_manifest.json` (SHA-256
`f524411f8a975349a837f9f29ffd8565a6be950dbb5fbae52b400cba339cef58`).

- Train: 15 components.
- Calibration: `V2C010`, `V2C015` (shared `AVGIGAVF` antigen family).
- Final test: `V2C001`, `V2C002`, `V2C003` (GP2 antigen family).
- The `V2C011`/`V2C020` NANP-motif family remains together in train.

The final-test components must not determine checkpoint choice, hyperparameters,
calibration, thresholds, or stopping. Dataset construction omits the test split
unless `--include-test` is explicitly supplied.

## Candidate contract

- Candidate residues are exactly the frozen H3 indices in the source manifest.
- Heavy and light framework residues are context.
- The antigen chain is explicitly marked as antigen context.
- Existing selected variants and native/composition-shuffle controls are reused.
- Duplicate prediction identities are prohibited by the AF2 result contract.

## AF2 label contract

Config: `configs/benchmarks/candidate_interface_multiscaffold_v1_af2.yml`.

- AF2-Multimer v3, model 1, three recycles.
- Seeds: 7103, 7111, 7121.
- Full PAE is stored per prediction as a compressed `float32` NPZ sidecar.
- Each PAE and PDB file has a SHA-256 recorded in `results.json`.
- Per-residue pLDDT is recovered from PDB CA-atom B factors and checked against
  the runner's mean pLDDT within the two-decimal PDB precision tolerance.
- Dataset groups are `component_id+af2_seed`; variants are compared only within
  a shared component and AF2 seed.

Two sequence-only AF2 parameter protocols are frozen: Multimer v3 model 1 and
model 2, each with three seeds. Targets are means over all six replicates.
Replicate variance determines a precommitted stability weight and excludes
ranking pairs whose target difference does not exceed AF2 replicate noise.
These protocols satisfy the cross-protocol requirement but are not explicit
coordinate-level antigen poses. A strict two-pose deployment claim still
requires a separately frozen initial-guess or template pose protocol.

## Precommitted deployment gates

- pLDDT MAE <= 0.05.
- ipTM MAE <= 0.05.
- Normalized PAE MAE <= 0.10.
- Pair accuracy >= 0.65 for every output.
- Median component Spearman >= 0.50 for every output.
- Pair-accuracy bootstrap 95% lower bound > 0.50.
- Prediction/target within-group variance ratio between 0.5 and 2.0.
- Cross-protocol prediction variance <= 2 times target variance.
- No systematically negatively correlated test component.
- All three model-training seeds must pass.

These are deployment gates, not claims that the current development model has
passed. AF2-derived confidence is not evidence of binding, affinity,
specificity, efficacy, or binder probability.

## Current development outcome

Three independently initialized successor heads were trained with seeds 2041,
2053, and 2069. Pairwise supervision uses uncertainty of the six-replicate mean,
`sqrt(SEM_i^2 + SEM_j^2)`, while replicate SD remains the stability reference.
Balanced raw checkpoints selected without final-test access are iterations 400,
200, and 400, respectively. The machine-readable audit is
`results/confidence_candidate_interface_multiscaffold_dual_sem_v1/selection_summary.json`.

Deployment gates did not pass. Candidate pLDDT median calibration-scaffold
Spearman is 0.20-0.38, pair accuracy is 0.53-0.63, and entity-level prediction
variance is too small. Only two independent calibration ipTM pairs exceed the
aggregate-mean uncertainty threshold, so their perfect ordering is insufficient
evidence on its own. PAE is stronger: all seeds meet normalized MAE, median
scaffold Spearman is 0.70-0.75, and 180 qualified-pair accuracy is 0.83-0.85;
however, PAE entity-level variance remains under-dispersed. Cross-condition
prediction variability stays below the allowed 2x replicate-SD upper bound. The
final-test split remains unexported and unevaluated.
