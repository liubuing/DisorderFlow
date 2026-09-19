# ECLS Structural Sequence-Scoring Publication Protocol

## Target

Current manuscript: `publication/MANUSCRIPT_DRAFT.md`. Current release:
`disorderflow-ecls-v1`. See `docs/PUBLICATION_MAP.md` and
`release/ecls_v1/publication_line.json` for publication routing. The PAE surrogate
is a separate manuscript revision and does not replace this ECLS scope.

- Primary venue: Bioinformatics.
- Secondary venue: PLOS Computational Biology, conditional on a stronger
  biological analysis of flexible-epitope recognition.
- Study type: computational structural sequence-scoring and benchmark paper.
- Wet-lab validation: out of scope for the current study.

## Primary Claim

The primary claim is that epitope-conditioned likelihood shift (ECLS) assigns
the deposited native CDR-H3 a more favorable peptide-coordinate likelihood
contrast than composition-matched shuffled H3 controls on the frozen temporal
antibody-peptide structure panel.

This is a native-versus-counterfactual structural sequence-scoring claim. It is
not a claim that ECLS generally reranks generated candidates, improves antibody
design, predicts contacts without bound geometry, identifies causal hotspots,
or predicts binding.

The study will not claim experimentally validated binding, affinity,
specificity, or therapeutic efficacy.

## Method Under Test

The method under test is the fixed-coefficient ECLS contrast defined below.
Ensemble design, disorder conditioning, contact-v2, candidate generation, AF2
reranking, and wet-lab mutation testing are outside the primary paper claim.
They may appear only as explicitly labelled negative results, supplementary
development analyses, or future work.

External metapredict profiles and the internal disorder head are ablation arms,
not ground-truth disorder labels. The internal head must be described as a
latent routing feature unless independently validated.

### Current development candidate

The current scorer candidate is the epitope-conditioned likelihood shift
(ECLS):

`ECLS = H3 NLL(complex backbone) - H3 NLL(peptide-stripped complex-derived Fab backbone)`

For native-versus-composition-matched comparisons, the positive-direction
advantage is:

`ECLS advantage = mean ECLS(shuffled H3) - ECLS(native H3)`

This endpoint measures whether the deposited peptide geometry preferentially
supports the observed H3 over composition-matched controls. It does not measure
affinity. ProteinMPNN pretraining exposure remains a required temporal-analysis
and provenance limitation.

The fixed-coefficient ECLS scorer becomes a general reranking method only if it
improves candidate ranking across BFN, ProteinMPNN, and ESM-IF generation arms.
The seven-cluster development experiment rejected that generality claim. A
generator-calibrated likelihood contrast is therefore a separate exploratory
candidate:

`GCLC(lambda) = H3 NLL(complex) - lambda * H3 NLL(peptide-stripped reference)`

ProteinMPNN uses an adaptation-calibrated `lambda=1.75`; BFN and ESM-IF retain
the frozen ECLS value `lambda=1`. Native-backbone retrieval alone remains
insufficient because it can reward backbone preorganization or model
memorization.

## Benchmark Contract

The benchmark must contain antibody-peptide complexes with auditable
PDB/SAbDab provenance. Train, development, and final components must be
disjoint across all of the following:

- PDB entry and structure-derived duplicate group.
- Paired antibody identity.
- VH homology cluster.
- VL homology cluster.
- CDR-H3 homology cluster.
- Antigen/epitope homology cluster.

Splits must be generated as connected components over all isolation axes. A
split is invalid if any cross-split component remains. All source manifests,
split IDs, thresholds, tool versions, and SHA256 values must be saved.

The existing 728/187/230 routing split is not the publication benchmark: it
does not enforce VH/VL homology isolation and its final set belongs to a
different preregistered hypothesis. Its 230-case final remains sealed.

## Required Baselines

- Native-sequence and composition-matched controls.
- Parent BFN without disorder conditioning.
- Current single-profile metapredict routing model.
- ProteinMPNN with the same design positions and structures.
- ESM-IF or another established inverse-folding baseline.
- Single-conformation version of the proposed method.

Every arm receives the same structures, design positions, sample count, and
post-generation filters. A missing or failed baseline is reported as missing;
it cannot be silently omitted.

**Current completion status (2026-08-05):** a 4HIX diagnostic completed parent
BFN, current BFN with disorder conditioning on and off, and ProteinMPNN using
the same 12-residue H3 mask, three seeds, and eight candidates per seed. This
single pre-AF2 scaffold is insufficient for the publication baseline panel.
Matched-position ESM-IF is operational on seven exposed development antigen
units (168/168 raw candidates; zero failures), but it is not confirmatory.
A new SAbDab snapshot produced 91 structurally eligible candidates. The v1
audit found zero independent from 4,129 training/prior-exposure records under
50%-identity VH/VL gates, which are unsuitable for germline-related antibody
frameworks. Before accessing any generator or scorer output, the frozen v2
matrix selected 90% VH/VL, 70% paired-CDR, 50% H3, and 30% antigen identity at
80% coverage and yielded 20 connected-component representatives. They are a
prospective internal holdout, not external confirmation. A controlled Stage A
random-initialization and Stage B disorder-supervision lineage completed, and
raw generation preserved all 2,400 frozen attempts across five arms with no
final failed slots. Diversity-only selection retained 280 candidates and 20
explicit ESM-IF shortfall slots. Three-seed AF2 completed 960 predictions and
PRODIGY scored 959 structures, with one no-contact failure. Only 7/20 components
were valid across all arms; oracle-baseline-minus-proposed mean delta was
-0.7635 kcal/mol (95% component-bootstrap CI -1.5063 to -0.0403), and every
preregistered gate failed. The full design claim is rejected for this internal
v2 hypothesis.

## Evaluation

### Primary endpoint

The primary endpoint is antigen-cluster-level native ECLS advantage over 200
composition-matched H3 shuffles on the one-time temporal final. The frozen
summary is mean advantage `0.172281`, 95% cluster-bootstrap CI
`[0.059156, 0.291920]`, with 12/15 positive clusters. That temporal final is
terminal and must not be rerun.

### Mandatory secondary endpoints

- Native CDR-H3 recovery and native-versus-decoy retrieval.
- Interface contact recovery across conformers.
- Mean and lower-tail interface energy from an evaluator not used for training.
- Discrimination from composition-matched scrambled epitopes.
- Sensitivity to known interface alanine substitutions.
- Fab structural validity and developability pass rate.
- Sequence uniqueness and cluster-level diversity.
- Runtime and compute cost.

The 4HIX diagnostic now includes a computational contact-position alanine scan,
PRODIGY interface scoring, sequence developability, diversity, and runtime.
These are single-scaffold computational endpoints. A public mutation audit
retained six heterogeneous anti-Abeta records (five point mutations), but only
one point mutation had an exact uncensored numeric fold effect; pooled
quantitative validation is therefore prohibited. Multi-scaffold confirmatory
endpoints are now available as a negative internal computational result. They
do not establish non-binding, affinity, external generalization, or experimental
failure; an internal holdout is not external confirmation.

Synthetic mutations and shuffled sequences are counterfactual controls, not
experimental non-binders. Metrics must use complex-level estimates, not treat
residues or generated sequences from one complex as independent samples.

## Statistical Contract

- At least three training or generation seeds per learned arm.
- Paired bootstrap confidence intervals resampled by homology component.
- Effect sizes and confidence intervals reported with P values.
- Multiple secondary comparisons corrected with Benjamini-Hochberg.
- No claim based only on pooled residue-level AUROC.
- No final-set tuning, threshold changes, checkpoint selection, or reruns.

## Historical Development Gates

The already completed temporal final was eligible only after the following
development conditions. These gates are historical and cannot be reused to
justify a new final evaluation:

1. The primary endpoint improves over the strongest complete baseline and its
   paired 95% confidence interval excludes zero.
2. Improvement is present in at least three seeds and is not driven by one
   antigen family or CDR-H3 length bin.
3. Scrambled-epitope discrimination improves or remains non-inferior.
4. Fab validity and developability are non-inferior to the parent BFN.
5. The conclusion survives removal of sequences homologous to model pretraining
   data, or the remaining provenance limitation is explicitly bounded.
6. All required baselines and preregistered ablations completed successfully.

The broader multi-generator design gates subsequently failed. That failure does
not reverse the bounded ECLS native-versus-shuffle result, but it prohibits an
antibody-design-success claim in this paper.

## Existing Evidence Classification

- The repeatedly used 328-case validation set is development evidence only.
- The failed 717-case routing blind is development evidence only.
- The 187-case project development failures are negative development results.
- The 230-case routing final is sealed and cannot be repurposed silently.
- The CAID result with 42 disordered residues and AUROC 1.0 is a diagnostic,
  not publication-level disorder validation.
- AF2/Multimer ipTM is an absolute plausibility filter, not an affinity ranking
  target.
- A-beta/5CSZ is a case study and cannot establish generalization by itself.
- The corrected official-H3 development split contains 234 adaptation records,
  24 development records, and 29 still-sealed final records before selecting
  one representative per antigen cluster.
- The first ECLS discovery set has only seven antigen clusters and is
  hypothesis-generating evidence.
- The legacy disorder-conditioning ablation produced 2.83% Hamming change. The
  corrected production inference path, after explicit antigen-chain role
  assignment, produced 8.33% paired Hamming change on 4HIX; 17/24 pairs changed.
  This establishes pathway sensitivity, not beneficial design direction.
- Frozen CAID3 Disorder-NOX evaluation retained 187/204 targets after direct
  training-sequence homology screening and scored 83 exact AFDB mappings. Among
  71 mixed-label UniRef50 clusters, macro ROC-AUC was 0.7551 with bootstrap 95%
  CI [0.6945, 0.8119]. Metapredict reached 0.7927; paired BFN-minus-metapredict
  was -0.0376, 95% CI [-0.0786, 0.0056]. External discrimination passed, while
  calibration failed (Brier 0.2991; ECE 0.3794). The head may be described as
  externally discriminative but not superior or probability-calibrated.
- The exposed 46-antigen-cluster adaptation analysis passed its frozen development
  gates: mean advantage 0.217481, bootstrap 95% CI [0.146050, 0.289555], median
  advantage 0.203056, and 80.4348% positive clusters.
- The ECLS publication final was evaluated once under the temporal contract;
  generator reranking was not run on or tuned against that final.

### Temporal final result

The frozen post-ProteinMPNN-training temporal final was evaluated once on 31
structures aggregated into 15 official antigen clusters. It passed all frozen
ECLS gates: mean advantage 0.172281, median advantage 0.118699, bootstrap 95%
CI [0.059156, 0.291920], and 80% positive antigen clusters. The immutable
decision is recorded in
`results/publication/h3_ecls_temporal_final_v1/final_decision.json`.

This establishes a positive computational ECLS result. It does not yet satisfy
the full design-paper claim, which still requires multi-generator candidate
reranking and the required baseline panel without changing the final ECLS
definition or rerunning this temporal final.

### Candidate-reranking development result

The frozen seven-antigen-cluster pilot generated eight candidates at each of
three seeds from BFN, ProteinMPNN, and ESM-IF, with the native H3 spiked into
each pool. All three arms completed without generation failures. Universal
ECLS did not beat complex NLL across generators: mean normalized-native-rank
gain 0.055952 with cluster-bootstrap 95% CI [-0.191667, 0.271825]. This rejects
the universal-reranker development gate.

The failure exposed generator-specific score calibration. On 46 exposed
adaptation clusters, ProteinMPNN complex NLL placed its own generated
candidates ahead of native H3 almost universally. ECLS improved normalized
native rank by 0.368021 over that self-biased baseline, 95% CI [0.256568,
0.479522], but absolute ECLS rank remained below random. The smallest frozen
peptide-stripped-reference coefficient meeting adaptation mean-NNR >=0.95 and
top-1 >=0.90 was
`lambda=1.75`.

Applying the generator-aware coefficients to the seven development clusters
gave pooled mean normalized native rank 0.853175. Its improvement over random
expectation was 0.353175 with 95% CI [0.166667, 0.496032], and six of seven
clusters exceeded random (nominal exact paired permutation P=0.046875). Leave-one-
cluster-out mean NNR ranged from 0.828704 to 0.937500. ProteinMPNN remained
nominally positive after correction across generator-level comparisons (BH
q=0.046875), but this descriptive adjustment does not correct the adaptive
analysis selection. Mean gain over complex NLL was 0.186508, but the 95% CI
[-0.001984, 0.329365] and exact P=0.171875 did not establish superiority. This
is a development-stage native-retrieval result, not evidence that selected
novel H3s bind, and not confirmation on the immutable temporal final.

## Deliverables

1. Versioned public benchmark manifest and split audit.
2. Reproducible runners for every baseline and ablation.
3. Frozen metric implementation with unit tests and scorer sanity checks.
4. Development report with seed-level and family-level analyses.
5. One-time final report generated from the frozen protocol.
6. Tables, figures, model cards, limitations, and reproducibility package.

## Disorder Supervision Status (2026-08-05)

The confidence-weighted supervision implementation is documented in
`docs/DISORDER_SUPERVISION_PROTOCOL.md`. DisProt experimental regions are now
represented with per-residue source, confidence, and evidence masks; unknown
residues are not treated as ordered. CAID2 targets and their available UniRef50
clusters are excluded before a deterministic UniRef50 train/development split.

The current local structure intersection contains 61 train and 6 development
records (5,542 supervised residues; zero cross-split clusters). After enforcing
the model's length range, only 27 train and 5 development structures remain.
This was sufficient for pipeline smoke testing but not publication training.
Three balanced-v4 seeds were subsequently completed. Frozen CAID3 evaluation is
reported above; its positive discrimination and failed calibration replace the
earlier `post-evaluation remains unrun` status.

## Follow-up Diagnostic Results (2026-08-05)

The corrected 4HIX H3 is chain-local `H:96-107` (`VRYDHYSGSSDY`). The previous
10- and 11-residue masks are invalid. Each matched generation arm produced 24
unique candidates. Mean native recovery was 0.0799 for current disorder-on,
0.1528 for current disorder-off, 0.0625 for parent BFN, and 0.3264 for
ProteinMPNN. The result does not favor disorder conditioning by native recovery.

Three candidates per arm were selected before AF2 by each arm's native score and
evaluated with three identical AF2 seeds. Mean candidate-median ipTM was 0.4563
for current disorder-on, 0.4450 for disorder-off, 0.4524 for parent BFN, and
0.4557 for ProteinMPNN. Native and composition-shuffle medians were 0.4621 and
0.4635. AF2 therefore did not separate native from shuffle or establish design
improvement. PRODIGY scores had large seed spread and are descriptive only.

Developability pass fractions at the frozen 0.55 heuristic threshold were
0.708, 0.750, 0.792, and 1.000 for current-on, current-off, parent BFN, and
ProteinMPNN. Contact-position Y3A, S9A, and S10A scans did not show a consistent
AF2/PRODIGY loss and are computational counterfactuals, not mutation evidence.

The separate 3STB five-seed panel retained high monomer pLDDT but failed every
multimer gate. Native mean ipTM was 0.138 versus 0.110 for a composition-matched
CDR scramble; all designed arms were at or below 0.110. The de novo full-length
Abeta42 interface ceiling remains unbroken.

## Pose-Neighborhood Ensemble Extension

The flexibility extension is operationally defined as a `T1` local
pose-neighborhood ensemble, not experimental flexibility. Starting from each
deposited antibody-peptide pose, OpenMM 8.5.2 with Amber14 and GBN2 implicit
solvent generated four independently seeded restrained Langevin conformers.
Antibody heavy atoms were position restrained at 5000 kJ/mol/nm2 and peptide
backbone atoms at 25 kJ/mol/nm2. Accepted conformers required antibody-backbone
RMSD <=0.5 A, peptide-backbone RMSD between 0.05 and 3.0 A, at least 50% native
residue-contact retention, no nonlocal heavy-atom clash below 1.5 A, and no
worsening of prepared-reference CA or peptide C-N geometry outliers. A valid
ensemble required at least three of four accepted conformers and mean pairwise
peptide-backbone RMSD >=0.25 A.

The frozen seven-cluster development run produced valid ensembles for six
clusters (85.7%) with mean pairwise peptide-backbone RMSD 0.6299 A. Mean
ensemble native-versus-shuffle ECLS advantage was 0.1314, with cluster-bootstrap
95% CI [0.0220, 0.2324], but only four of six valid clusters were positive. The
frozen 70% positive-cluster gate therefore failed. Ensemble advantage was lower
than the deposited-single-pose advantage by 0.2004 on average, with 95% CI
[-0.3518, -0.0355]. This demonstrates a physical local ensemble layer and
quantifies native-pose sensitivity; it does not demonstrate that ensemble
averaging improves ranking.

The next permitted extension is a separately frozen `T2` moderate-perturbation
stress test with broad residue-level pose restraints and held-out contacts. It
must not access or rerun the immutable temporal final. A flexibility claim may
refer only to model-generated local conformational dispersion until independent
experimental or non-native-initialization evidence is available.

### T2 moderate-perturbation recovery result

The frozen `T2` development protocol started from 1.5-3.0 A antibody-aligned
peptide-backbone perturbations. Native CA residue contacts at 8 A were divided
deterministically into supplied and held-out halves. Generation received only
the supplied half as broad 8 A upper-bound restraints; held-out contacts were
used only for evaluation.

Only three of seven clusters met the requirement of at least three accepted
replicas, giving a valid-cluster fraction of 0.4286 versus the frozen 0.8 gate.
Across valid clusters, mean held-out contact recovery was -0.0171 (95% CI
[-0.0513, 0.0476]; exact sign-flip P=0.75), and mean RMSD recovery was -0.1415 A
(95% CI [-0.4146, 0.1135]; P=0.50). Mean final ECLS advantage was 0.1020, but
its 95% CI [-0.0609, 0.3729] included zero. The current `T2` recovery protocol
is therefore rejected. This negative result does not invalidate the `T1` local
stress-test implementation, but it prevents a claim of recovery from moderate
perturbations.
