# Flexible-Epitope Antibody Design Publication Protocol

## Target

- Primary venue: Bioinformatics.
- Secondary venue: PLOS Computational Biology, conditional on a stronger
  biological analysis of flexible-epitope recognition.
- Study type: computational methods and benchmark paper.
- Wet-lab validation: out of scope for the current study.

## Primary Claim

The study will test whether ensemble-aware, template-constrained CDR-H3 design
improves computational compatibility with flexible peptide epitopes on a
strictly homology-disjoint benchmark, relative to single-structure design and
established sequence-design baselines.

The study will not claim experimentally validated binding, affinity,
specificity, or therapeutic efficacy.

## Method Under Test

The proposed method must combine:

1. Multiple peptide-epitope conformations per antibody scaffold.
2. Mean-case and lower-tail interface objectives.
3. Preservation of experimentally observed epitope contacts.
4. Fab-fold and developability constraints.
5. Candidate diversity constraints.

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

## Evaluation

### Primary endpoint

The primary endpoint is paired improvement over the strongest baseline in
lower-tail ensemble interface compatibility on the development and final
complexes. The exact score and lower-tail quantile must be fixed after a
scorer sanity benchmark and before model comparison.

### Mandatory secondary endpoints

- Native CDR-H3 recovery and native-versus-decoy retrieval.
- Interface contact recovery across conformers.
- Mean and lower-tail interface energy from an evaluator not used for training.
- Discrimination from composition-matched scrambled epitopes.
- Sensitivity to known interface alanine substitutions.
- Fab structural validity and developability pass rate.
- Sequence uniqueness and cluster-level diversity.
- Runtime and compute cost.

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

## Development Gates

The publication final can be evaluated only if all conditions hold on the
development set:

1. The primary endpoint improves over the strongest complete baseline and its
   paired 95% confidence interval excludes zero.
2. Improvement is present in at least three seeds and is not driven by one
   antigen family or CDR-H3 length bin.
3. Scrambled-epitope discrimination improves or remains non-inferior.
4. Fab validity and developability are non-inferior to the parent BFN.
5. The conclusion survives removal of sequences homologous to model pretraining
   data, or the remaining provenance limitation is explicitly bounded.
6. All required baselines and preregistered ablations completed successfully.

If these gates fail, the final remains sealed and the method is revised using
development data only.

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
