# Follow-up Validation Protocol

Status: frozen before new follow-up inference on 2026-08-05.

## Claim Boundary

This follow-up tests contact-specific recovery, matched sequence generation,
AF2 uncertainty, external disorder prediction, alanine sensitivity,
developability, independent interface scoring, and runtime. It does not test
experimental binding, affinity, specificity, or therapeutic suitability.

The ECLS temporal final, T2.1 temporal-final outputs, legacy 4HIX ProteinMPNN
candidates, CAID2, and all previously exposed adaptation sets remain immutable.
They must not be rerun, reselected, or used for threshold tuning.

## Frozen Existing Results

- T2.1 source SHA-256: `ee33f4aa0c182fdbca3d3c63140f2036b64248162b949974574515740d6bd9d6`.
- T2.1 corrected analysis: supplied-minus-random `-0.036648`, 95% cluster
  bootstrap CI `[-0.063792, -0.010814]`. Contact-specific guidance is rejected.
- Proposed balanced-v4 checkpoint SHA-256:
  `ed9984504d799187b7660c1d40c3dc68c01a0ce9b3cf92e25af1273195931252`.
- Parent AntibodyDesignBFN checkpoint SHA-256:
  `f3e254773eca35bef98c3875544d0c114ef86547579e56de8f470b03ff977843`.
- 4HIX structure SHA-256:
  `2bb5404a3b66741f02e89b899f36982fb33f16f042149e3614435f4748004201`.
- 4HIX H3 is the anchor-bounded chain-local range `H:96-107`, sequence
  `VRYDHYSGSSDY`. Older 10- or 11-residue masks are invalid.

## Independent Contact Validation

The existing T2.1 result is a terminal negative result for the current
contact-specific mechanism. No additional run on that final is permitted, and
new data are not required to close that negative hypothesis. Testing a
mechanistically distinct successor requires a new preregistration and at least
12 antigen-homology components isolated from every previous set.

The primary endpoint is component-level supplied-minus-random held-out contact
recovery. Random restraints are matched by count, chain roles, and initial
distance bin. The claim passes only if at least 80% of components are valid,
at least 70% have a positive difference, and the paired component-bootstrap
95% CI excludes zero in the positive direction.

## Matched Generator Comparison

New-lineage BFN, the same BFN with disorder conditioning disabled, its Stage A
antibody checkpoint, ProteinMPNN, and ESM-IF receive identical structures,
context chains, design positions, three seeds, and eight candidates per seed.
Legacy balanced-v4 and parent-BFN checkpoints are prohibited as initializers for
the new lineage. Failed candidates remain failures. No post-generation
threshold may differ by arm.

The primary endpoint for a future multi-scaffold panel is worst-conformer
independent interface score. Native recovery, contact recovery, diversity,
developability, AF2 metrics, and runtime are secondary. A single 4HIX result is
only a runner diagnostic and cannot establish design success.

## AF2 Uncertainty

Candidate selection is frozen before AF2. Each candidate, native, and matched
shuffle is evaluated with the same AF2-Multimer model set, MSA/template policy,
recycles, and at least three seeds. Report median, range, and failure rate for
ipTM, interface PAE, and pLDDT. Candidates and seeds are averaged within each
antigen-homology component before inference.

## External Disorder Validation

CAID3 is the external label-unseen benchmark. Download and hash all five
official references before parsing labels. The balanced-v4 checkpoint and
Platt calibration above are immutable. No checkpoint, threshold, or feature is
selected after CAID3 access. CAID2 remains development-only.

Primary endpoint: macro per-target ROC-AUC among targets containing both
classes. Secondary endpoints: macro and pooled PR-AUC, MCC, F1, Brier, ECE,
and paired comparison with frozen metapredict and constant-prevalence controls.
Targets are resampled by UniRef50 component. External validation requires at
least 30 eligible independent targets and a 95% CI for macro ROC-AUC above 0.5.

## Secondary Endpoints

- Alanine sensitivity requires experimentally reported interface alanines;
  synthetic scans are labelled as computational controls.
- Sequence developability uses identical full VH/VL sequences and the frozen
  risk threshold `0.55`; it is a heuristic screen, not an assay.
- PRODIGY is the first independent open interface evaluator. Its predicted
  delta-G and Kd are reported separately from AF2 and model scores.
- Runtime records preprocessing, generation, AF2, and scoring wall time,
  hardware, failures, and peak accelerator memory per accepted candidate.

The confirmatory contact, generator, and external-disorder hypotheses use Holm
family-wise correction at 0.05. Secondary endpoint families use
Benjamini-Hochberg correction at 5%. There is no efficacy-based early stopping
or optional sample-size extension.

## Execution Status

- T2.1 remains frozen negative; no final-set rerun was performed.
- The 4HIX matched diagnostic completed 24 candidates per arm and three-seed AF2
  on three preselected candidates per arm. Native and composition shuffle were
  not separated, so no design-improvement claim passed.
- PRODIGY, heuristic developability, runtime, diversity, and computational Y3A,
  S9A, and S10A endpoints completed for 4HIX only.
- The existing five-seed 3STB panel failed the absolute multimer gate; the
  full-length Abeta42 de novo ceiling remains unbroken.
- CAID3 Disorder-NOX completed on 83 exact AFDB mappings after training-sequence
  and UniRef50 gates. Macro ROC-AUC was 0.7551, 95% CI [0.6945, 0.8119].
  Calibration failed and metapredict was not inferior to BFN.
- The current T2.1 contact-specific mechanism is terminal negative. Its final
  was not rerun; any successor mechanism requires a new protocol and data.
- Matched-position ESM-IF completed on seven exposed development antigen units:
  168/168 raw candidates, 31 unique non-native sequences, zero failures, and
  mean candidate native recovery 0.194. This is not confirmatory evidence.
- A fresh SAbDab metadata snapshot yielded 111 unexposed candidate instances;
  91 passed paired-chain, single-peptide, and ANARCII H3 gates. MMseqs2 against
  4,129 training/prior-exposure records found zero strict VH/VL/H3/antigen-axis
  independent records under the v1 50%-identity VH/VL gates. That v1 gate
  failed and generation was not run.
- Before any new-lineage training or candidate generation, the model-free v2
  matrix froze 20 prospective internal holdout components at 90% VH/VL, 70%
  paired-CDR, 50% H3, and 30% antigen identity with 80% coverage. This is not
  external confirmation. Stage A random-initialization and one-step optimization
  smoke tests passed on 2,952/328 antibody records; Stage B is restricted to the
  Stage A checkpoint and a 827-record disorder lookup after one antigen homolog
  was removed. Stage A completed 100,000 iterations; Stage B selected iteration
  2,200 by validation disorder ROC-AUC. Raw generation then completed all 2,400
  frozen attempt slots across 20 components, five arms, three seeds, and eight
  samples with no final failures. This is not a performance endpoint; candidate
  selection retained 280 unique candidates and 20 explicit ESM-IF shortfall
  slots. Three-seed AF2 completed 960/960 predictions, and PRODIGY scored
  959/960 structures; one no-contact structure remained a failure. Only 7/20
  components were valid across all five arms. The proposed arm's mean
  oracle-baseline-minus-proposed delta was -0.7635 kcal/mol (95% component
  bootstrap CI -1.5063 to -0.0403), with 28.6% positive components. All four
  preregistered gates failed. This is a terminal negative result for the v2
  internal computational hypothesis, not external or experimental evidence.
- The public experimental mutation audit retained six heterogeneous records
  (five point mutations) from two anti-Abeta antibodies. Only one point mutation
  had an exact uncensored numeric fold effect, so pooled quantitative
  correlation is prohibited; qualitative and censored checks are available.
- A mechanistically distinct successor-v3 was preregistered as development-only
  before training. It specializes on fixed-backbone H3 sequence design and uses
  shared-noise factual-versus-mismatched-antigen native-H3 NLL plus contact
  supervision. Only the frozen v2 Stage A checkpoint was permitted as an
  initializer; Stage B and the v2 holdout remained prohibited.
- Successor-v3 completed 5,000 iterations. On 52 previously exposed peptide
  antigen components, all development gates passed: factual-minus-mismatched
  native-H3 NLL was -0.2181 (95% component-bootstrap CI -0.3077 to -0.1389),
  factual native-H3 NLL improved by 0.3024 versus Stage A, contact AUROC was
  0.6125, and 52/52 components were valid. This is a positive development result,
  not confirmation, external validation, experimental binding evidence, or a
  reversal of the terminal-negative v2 result. These 52 components are now
  permanently exposed and prohibited for successor confirmation.
- A successor-v3 one-time internal computational confirmation was preregistered
  before accessing the live SAbDab summary. The 2026-08-06 SAbDab update added
  173 instances and 99 PDBs relative to the frozen 2026-07-13 baseline, but
  exact-diff peptide filtering retained only 19 instances from 9 PDBs. Because
  exact PDB isolation bounds the possible component count at 9, below the frozen
  minimum of 12, the metadata feasibility gate failed. No candidate structure
  was downloaded, no homology threshold was changed, and neither the successor
  nor Stage A checkpoint was accessed. This is a sample-size failure, not a
  successor-v3 performance result.
