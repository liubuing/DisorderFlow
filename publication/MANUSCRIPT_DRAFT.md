# Backbone-conditioned likelihood contrasts for computational scoring of antibody CDR-H3 sequences in antibody-peptide complex structures

## Abstract

### Motivation

Sequence scores conditioned on deposited antibody-peptide structures may
conflate peptide-coordinate context with compatibility of the same
complex-derived Fab backbone after peptide removal.

### Results

We evaluated epitope-conditioned likelihood shift (ECLS), defined as CDR-H3
negative log-likelihood (NLL) conditioned on the complex backbone minus CDR-H3
NLL conditioned on the same complex-derived Fab coordinates after peptide
removal. CDR-H3 residues were mapped using
official SAbDab2 annotations, and retained structures required a CDR-H3-peptide
minimum heavy-atom distance no greater than 4.5 A. Across 46 official antigen
clusters, native CDR-H3 sequences showed a mean ECLS advantage of 0.217 over
200 composition-matched shuffles (median 0.203; cluster-bootstrap 95% CI
0.146-0.290), with positive effects in 80.4% of clusters. A one-time
post-ProteinMPNN-training temporal final contained 31 structures from 15 antigen
clusters and yielded a mean advantage of 0.172 (median 0.119; 95% CI
 0.059-0.292), with 80% positive clusters. In a separate seven-cluster
development benchmark, universal ECLS did not outperform complex NLL for
reranking candidates from BFN, ProteinMPNN, and ESM-IF. The evidence therefore
supports a bounded native-versus-shuffle structural likelihood contrast, not a
general candidate-ranking or antibody-design claim.

### Availability

Code, frozen configs, tests, checksummed result evidence, and lightweight
reviewer reproduction commands are provided with the accompanying package.
Large derived datasets and third-party model weights are distributed
separately by checksum. No wet-lab validation was performed.

## 1 Introduction

CDR-H3 often contributes directly to antibody recognition, but scoring a
CDR-H3 sequence in an antibody-peptide structure is complicated by the
relationship between peptide-coordinate context and the deposited native
backbone. Backbone-conditioned inverse-folding likelihoods offer
a computational measure of sequence compatibility with a deposited structure,
but complex-conditioned likelihood alone does not isolate sensitivity to the
presence of peptide coordinates.

We therefore compare the likelihood of the same CDR-H3 sequence under two
coordinate contexts: the antibody-peptide complex and the same complex-derived
Fab after removing peptide coordinates without altering the Fab backbone. The
resulting ECLS score is a conditional likelihood contrast, not a binding energy
or affinity estimate. Its primary test is whether the deposited native CDR-H3
is preferentially supported relative to composition-matched counterfactual
sequences.

The study makes two distinctions. First, native-versus-shuffle ECLS was
developed on an exposed adaptation set and then evaluated once on structures
deposited after the ProteinMPNN training cutoff. Second, candidate reranking across BFN,
ProteinMPNN, and ESM-IF was treated as a separate development analysis. The
latter found descriptive differences between generator-specific pools and
motivated an explicitly exploratory generator-aware contrast.

### 1.1 Frozen scope and claim boundary

The primary paper contains one positive claim: native-versus-composition-shuffle
ECLS discrimination on the one-time temporal final. Candidate generation,
contact-v2, disorder prediction, molecular-dynamics recovery, AlphaFold
reranking, multiscaffold design, and planned SPR/BLI experiments do not support
that claim and are not presented as successful antibody design. Negative and
exploratory analyses are retained only to define limitations and prevent
selective reporting.

## 2 Materials and methods

### 2.1 Structural records and CDR-H3 mapping

Antibody-peptide structures were derived from the audited SAbDab2 resource
[1,2].
Publication records used the official SAbDab2 `VH_numerable_seq` and `CDRH3`
annotations, which were mapped exactly to parsed heavy-chain coordinates.
Approximate chain-length-scaled CDR labels were not used for publication
evaluation. Residue identifiers retained insertion-code information during
mapping.

Heavy chain, light chain when present, and peptide antigen roles were assigned
explicitly. A record was eligible only when at least one CDR-H3 and peptide
heavy-atom pair was separated by no more than 4.5 A. ProteinMPNN inputs were
then written with contiguous chain-local numbering to prevent insertion codes
or missing residue numbers from changing design masks.

### 2.2 Isolation and inference units

The publication audit considered PDB identity, official antibody and antigen
clusters, paired-antibody identity, VH sequence, VL sequence, exact CDR-H3
sequence, and peptide sequence. Near-duplicate isolation used identity/coverage
thresholds of 0.90/0.90 for VH and VL, 0.80/0.80 for CDR-H3, and 0.70/0.80 for
peptide antigen. Later-split records connected to earlier records on any frozen
exact, official-cluster, or near-duplicate axis were excluded. All reported
statistical summaries used the official antigen cluster as the inference unit. Multiple structures, generation seeds,
and candidate sequences within a cluster were averaged before inference.

### 2.3 Epitope-conditioned likelihood shift

ProteinMPNN `v_48_020` [3] was used in backbone-only conditional-probability mode.
Only the heavy chain was designated as the scored chain, its non-H3 positions
were fixed, and light-chain and peptide coordinates were retained as structural
context in the complex calculation. The peptide-stripped calculation retained
the same heavy- and light-chain coordinates. NLL was the mean per-residue
negative natural log probability in nats over official CDR-H3 positions. For
sequence `s`, ECLS was

`ECLS(s) = NLL_complex(s) - NLL_peptide-stripped(s)`.

Lower ECLS was treated as more favorable. For native sequence `s_native` and
composition-matched shuffles `S_shuffle`, the positive-direction endpoint was

`ECLS advantage = mean[ECLS(S_shuffle)] - ECLS(s_native)`.

Each record used 200 independently seeded composition-matched shuffles. These
sequences are counterfactual controls and are not experimental non-binders.

### 2.4 Exposed adaptation and temporal final

The exposed adaptation analysis selected one representative structure from
each of 46 official antigen clusters. Its frozen gates required median ECLS
advantage of at least 0.1, positive advantage in at least 70% of clusters, and a
cluster-bootstrap confidence interval with lower bound above zero.

ProteinMPNN documents a training cutoff of 2 August 2021. The temporal final
was constructed from structures after this cutoff while excluding previously
scored identifiers and exact, official-cluster, and sequence-homology
conflicts. Thirty-one structures from 15 official antigen clusters passed
official CDR-H3 mapping and structural contact requirements. The temporal final
was evaluated once. Record-level effects were averaged within each antigen
cluster before inference. Its source-result SHA256 and terminal no-rerun
decision are stored separately from the legacy result status field.

### 2.5 Cross-generator candidate development

The cross-generator development benchmark contained one structure from each of
seven official antigen clusters. BFN [5,6], ProteinMPNN [3], and ESM-IF [4] each generated
eight fixed-length CDR-H3 candidates under three predetermined seeds. BFN and
ProteinMPNN received Fab and peptide coordinate context. ESM-IF received
multichain coordinates but not context-chain sequences. Invalid sequences and
duplicates were recorded and removed; no ECLS-dependent filter was applied.
Across 21 generator-by-seed pools, BFN retained eight unique candidates per
pool, ProteinMPNN retained 2-8 (mean 5.43), and ESM-IF retained 1-2 (mean 1.48).
No invalid or native-identical generated sequence occurred. The native CDR-H3
was then inserted once, so every analyzed pool contained at least two unique
sequences.

For native rank `r` in a pool of size `N`, normalized native rank was

`NNR = 1 - (r - 1) / (N - 1)`.

NNR is one for the best rank and has random expectation 0.5. Ties used average
rank. NNR was averaged across seeds within each generator and then equally
across the three generators within each antigen cluster; the seven cluster
means entered pooled inference.

### 2.6 Generator-aware likelihood contrast

Universal ECLS corresponds to a peptide-stripped-reference coefficient of one.
The exploratory generator-calibrated likelihood contrast was

`GCLC_lambda(s) = NLL_complex(s) - lambda * NLL_peptide-stripped(s)`.

BFN and ESM-IF retained the frozen ECLS coefficient `lambda=1` because no
separate generator-specific adaptation pools were available. For ProteinMPNN,
the coefficient was selected on the 46 exposed adaptation clusters over a grid
from 0 to 3 in increments of 0.05. The frozen selection rule chose the smallest
coefficient with mean NNR at least 0.95 and top-1 fraction at least 0.90,
yielding `lambda=1.75`. The seven development clusters had previously been
observed under `lambda=1`; consequently, GCLC is exploratory rather than an
independent confirmation.

### 2.7 Statistical analysis

Means and 95% confidence intervals used 10,000 bootstrap resamples [13] of
official antigen-cluster summaries. The frozen ECLS decisions used these confidence intervals and
prespecified effect gates. Supplementary post hoc sensitivity used an exact
two-sided sign-flip test for the 15-cluster temporal final and 1,000,000 seeded
Monte Carlo sign flips for the 46-cluster adaptation set. These were two global
tests over cluster-level inference units, not 46 separate cluster tests. Their
supplementary P values were also adjusted together by Benjamini-Hochberg [14]. Exploratory GCLC used
exact sign-flip tests; generator-level descriptive comparisons were adjusted by
the Benjamini-Hochberg method, which does not correct the preceding adaptive
analysis selection. Leave-one-cluster-out
sensitivity recomputed the pooled mean after omitting each development cluster.
Candidates and generation seeds were not treated as biological replicates.

### 2.8 Local pose-neighborhood peptide ensembles

Sections 2.8 onward describe supplementary or negative-scope analyses and do
not contribute evidence to the primary ECLS claim.

As an exploratory flexibility extension, each of the seven development
complexes was subjected to four independently seeded OpenMM [8] restrained
Langevin simulations using Amber14 with GBN2 implicit solvent. Antibody heavy
atoms were restrained at 5000 kJ/mol/nm2 and peptide backbone atoms at
25 kJ/mol/nm2. This `T1` protocol samples local dispersion around a deposited
pose and is not a solution-state or equilibrium ensemble.

Conformers were accepted only if antibody-backbone RMSD was no greater than
0.5 A, antibody-aligned peptide-backbone RMSD was 0.05-3.0 A, at least 50% of
prepared-reference antibody-peptide residue contacts were retained, no
nonlocal heavy-atom clash was below 1.5 A, and prepared-reference CA and peptide
C-N geometry outlier counts were not worsened. A cluster required at least
three accepted conformers and mean pairwise peptide-backbone RMSD of at least
0.25 A. Each accepted conformer was scored against its own peptide-stripped Fab
coordinates. The primary ensemble score was the arithmetic mean ECLS across
accepted conformers.

### 2.9 Moderate-perturbation recovery

The `T2` development experiment generated 1.5-3.0 A antibody-aligned peptide
backbone perturbations using high-temperature restrained dynamics. Native
antibody-peptide CA residue contacts at 8 A were split deterministically by
record identifier into supplied and held-out halves. Recovery dynamics received
only the supplied contacts as broad 8 A upper-bound restraints. Held-out
contacts did not enter generation and were the primary structural endpoint.
Each cluster required at least three of four physically accepted recovery
replicas. Perturbed and recovered structures were also scored by paired
complex-versus-peptide-stripped ECLS.

### 2.10 Deterministic torsion perturbation and recovery (T2.1 v2)

The T2.1 protocol replaces the stochastic high-temperature perturbation of T2
with a deterministic, length-adaptive phi/psi torsion rotation to reach the
1.5-3.0 A peptide RMSD tier while keeping the antibody frame fixed. The
antibody backbone RMSD under torsion perturbation is effectively zero (machine
epsilon). Recovery simulations use 2000-step Langevin dynamics at 300 K with
implicit solvent (Amber14, GBN2). Native antibody-peptide Cgamma residue
contacts at 8 A are split deterministically into supplied and held-out halves.
Recovery receives only the supplied contacts as broad 8 A upper-bound
restraints.

The v2 protocol addresses parameter calibration issues identified in the initial
T2.1 evaluation. Peptide backbone restraints were increased from 25 to
100 kJ/mol/nm2, the maximum peptide RMSD QC gate was relaxed from 3.5 to
5.0 A, and geometry outlier comparison was disabled for perturbed structures.
Selection was performed per-structure (31 records) rather than per-cluster.
All four control arms were evaluated: supplied contacts, all contacts, random
peptide CA restraints, and null structural restraints.

### 2.11 4HIX computational case study

We performed a single-scaffold computational diagnostic using the 4HIX
humanized 3D6/bapineuzumab Fab [12] and the six-residue Abeta N-terminal
fragment DAEFRH. The pipeline extracts the Fab framework (VH + VL),
identifies the anchor-bounded CDR-H3 as chain-local positions 96-107, fixes all
other residues, and compares current BFN with disorder conditioning on and off,
parent BFN, and ProteinMPNN [3]. Each arm used three predetermined seeds and
eight candidates per seed.

Three candidates per arm were selected before structure prediction using each
generator's own score and evaluated using three seeds of AlphaFold-Multimer [7]
(V3, recycle=3)
with antibody chains passed using the colon separator (VH:VL) rather than
concatenation, which was found to reduce interface pTM by a factor of 4.5
when omitted. Interface quality is assessed by ipTM, pLDDT, and interface
predicted aligned error (iPAE). Native and composition-matched shuffle controls
were run identically. PRODIGY [17] provided an independent computational interface
score. Full-chain sequence developability and contact-position alanine
counterfactuals were secondary diagnostics.

### 2.12 Frozen CAID3 external disorder evaluation

The balanced-v4 checkpoint and weighted Platt calibration were frozen before
downloading CAID3 Disorder-NOX [15]. MMseqs2 excluded targets matching any of the 828
actual training sequences at 30% identity and 80% coverage. Targets with
unresolved UniRef50 membership were also excluded. Exact DisProt-to-AFDB
sequence mapping provided structures for 83 eligible targets. Statistics used
UniRef50 clusters; metapredict 3.0.2 [16] and constant prevalence were frozen
baselines. No model, feature, calibration, or threshold was changed after label
access.

## 3 Results

### 3.1 Exposed adaptation supported native-versus-shuffle ECLS discrimination

Across 46 adaptation antigen clusters, mean ECLS advantage was 0.217481 and the
median was 0.203056. The 95% cluster-bootstrap confidence interval for the mean
was 0.146050-0.289555, and 80.4348% of clusters had positive advantage on
deposited native complex-derived backbones. The frozen adaptation gates were
therefore satisfied. A post hoc Monte Carlo sign-flip sensitivity analysis gave
nominal P<1e-6.

### 3.2 The ECLS signal transferred to the one-time temporal final

The post-training temporal final comprised 31 structures aggregated into 15
antigen clusters. Mean ECLS advantage was 0.172281, median advantage was
0.118699, and the 95% confidence interval was 0.059156-0.291920. Twelve of 15
clusters, or 80%, had positive advantage. All frozen final gates passed on the
single permitted evaluation. A post hoc exact sign-flip sensitivity analysis
gave nominal P=0.014221; leave-one-cluster-out mean advantage ranged from
0.141769 to 0.199818.

This result supports temporal transfer of computational native-versus-shuffle
discrimination on deposited native complex-derived backbones. It can reflect
native-backbone compatibility or residual model exposure and does not establish
binding or affinity. Potential ProteinMPNN exposure through sources other than
the documented date cutoff cannot be excluded completely.

### 3.3 Universal ECLS was not a universal candidate reranker

All BFN, ProteinMPNN, and ESM-IF development arms completed without generation
failures. Universal ECLS produced a pooled mean NNR gain of 0.055952 over
complex NLL, with 95% confidence interval -0.191667 to 0.271825. The interval
included zero, so the universal cross-generator reranking gate was rejected.

Performance differed descriptively among generator-specific pools. Complex NLL
ranked native sequences strongly in BFN and ESM-IF pools but placed
ProteinMPNN-generated alternatives
ahead of native in the ProteinMPNN arm. ECLS partially corrected the latter
self-generation bias, but on 46 adaptation clusters its absolute mean NNR was
0.373611, below random expectation. Improvement over a severely biased
baseline was therefore insufficient by itself.

### 3.4 Generator-aware calibration improved exploratory native retrieval

ProteinMPNN adaptation calibration selected `lambda=1.75`. With generator-aware
coefficients, ProteinMPNN native NNR was 1.0 on all seven development clusters.
BFN and ESM-IF retained mean ECLS NNR values of 0.845238 and 0.714286,
respectively. The ProteinMPNN comparison with random had nominal
Benjamini-Hochberg-adjusted `q=0.046875`; this value is descriptive because the
analysis was adaptively motivated. BFN and ESM-IF secondary comparisons did not
exclude random ranking.

Across generators, pooled mean NNR was 0.853175. Its improvement over random
was 0.353175 (95% CI 0.166667-0.496032; nominal exact permutation
`P=0.046875`), and six of seven clusters exceeded random. Leave-one-cluster-out pooled mean NNR ranged
from 0.828704 to 0.937500. Mean gain over complex NLL was 0.186508, but its 95%
confidence interval (-0.001984 to 0.329365) and exact permutation
`P=0.171875` did not establish superiority over that baseline.

### 3.5 Local physical ensembles exposed native-pose sensitivity

Six of seven development clusters produced valid `T1` ensembles, giving a
validity fraction of 0.857. Mean pairwise peptide-backbone RMSD among accepted
conformers was 0.6299 A. Mean ensemble native-versus-shuffle ECLS advantage was
0.1314 (95% cluster-bootstrap CI 0.0220-0.2324), indicating that a positive mean
signal remained under local physical perturbations. However, only four of six
valid clusters had positive ensemble advantage, below the frozen 70% gate.

The ensemble-minus-deposited-single-pose advantage was -0.2004 on average (95%
CI -0.3518 to -0.0355). Thus, local ensemble averaging did not improve the
single-pose result and instead exposed sensitivity to the deposited native
geometry. The `T1` development gate was rejected. These results establish an
auditable local conformational stress test, not successful ensemble-based
ranking or experimentally validated peptide flexibility.

### 3.6 Deterministic torsion perturbation recovery (T2.1 v2)

The initial T2 protocol using stochastic high-temperature perturbation yielded
only 3 of 7 valid clusters, with poor held-out contact recovery (-0.017, 95% CI
-0.051 to 0.048; P=0.75). The T2.1 v2 protocol addresses this through
deterministic torsion perturbation and recalibrated physical parameters.

Across 31 structures from the temporal final, 26 of 31 (83.9%) reached the
1.5-3.0 A torsion perturbation tier. After recovery with the v2 protocol
(100 kJ/mol/nm2 peptide restraint, 5.0 A RMSD gate), 25 of 26 perturbed
structures (96.2%) passed revised QC criteria, corresponding to 25 of all 31
selected structures (80.6%). Mean held-out contact recovery was
0.581 (95% CI 0.533-0.635), and all 25 valid structures (100%) showed positive
held-out contact recovery. Mean peptide-backbone RMSD recovery was -0.801 A
(95% CI -1.014 to -0.556), indicating slight peptide displacement during the
2000-step recovery simulation, consistent with the expected behavior of
restrained Langevin dynamics starting from a perturbed pose.

The control arms contradicted a contact-specific interpretation. Mean recovery
was 0.611 for all-contact restraints, 0.619 for random restraints, and 0.533 for
the null structural arm. After averaging structures within official antigen
clusters, supplied-minus-random recovery was -0.0366 (95% cluster-bootstrap CI
-0.0638 to -0.0108), and supplied-minus-all-contact recovery was -0.0258
(95% CI -0.0444 to -0.0069). The overall validity gate (>=0.70) passed at
25/31 (0.806), but the controls do not support recovery specifically driven by
informative native contacts. Because v2 parameters were recalibrated after the
initial failure, these descriptive results require independent confirmation.

### 3.7 Matched 4HIX computational diagnostic

Each matched arm produced 24 unique 12-residue candidates. Mean native recovery
was 0.080 for current disorder-on BFN, 0.153 for current disorder-off BFN, 0.063
for parent BFN, and 0.326 for ProteinMPNN. Paired current on/off sequences had
mean Hamming fraction 0.083; 17/24 pairs changed. Thus disorder input affected
production inference, but the direction was not favorable by native recovery.

For the three pre-AF2-selected candidates per arm, mean candidate-median ipTM
was 0.456 for current disorder-on, 0.445 for disorder-off, 0.452 for parent BFN,
and 0.456 for ProteinMPNN. Native median ipTM was 0.462; the composition-matched
shuffle was 0.464. Median interface PAE values remained near 26 A and pLDDT near
0.18. PRODIGY showed large seed-to-seed variation. Developability pass fractions
were 0.708, 0.750, 0.792, and 1.000, respectively. Computational Y3A, S9A, and
S10A scans did not show a consistent loss across AF2 and PRODIGY.

A critical methodological finding during validation was that AlphaFold 2 multimer
requires antibody chains to be passed with a colon separator (VH:VL) rather
than as a concatenated sequence (VH+VL). Without the chain break, ipTM dropped
from approximately 0.45 to 0.10, a 4.5-fold underestimation of interface
quality. This fix was incorporated into the design pipeline and is essential
for accurate AF2 evaluation of antibody-antigen complexes.

The matched diagnostic does not establish improvement. The 4HIX epitope is only 6 residues, AF2
interface metrics are less discriminative for very short peptides, the 4HIX Fab
was in AF2's training set (PDB deposited 2012), and no experimental binding
validation was performed. This is a single-scaffold computational case study,
not validation of BFN design or general IDP-targeting design.

### 3.8 External disorder discrimination transferred but calibration failed

Of 204 CAID3 Disorder-NOX targets, 17 were removed by direct training-sequence
homology screening and three otherwise independent targets had unresolved
UniRef50 assignments. Eighty-three exact AFDB mappings were scored; 71 clusters
contained both residue classes. Macro ROC-AUC was 0.7551 (cluster-bootstrap 95%
CI 0.6945-0.8119), passing the frozen discrimination gate above 0.5. Pooled
ROC-AUC was 0.8196. Mean Brier score was 0.2991 and ECE was 0.3794, failing the
calibration gate. Metapredict macro ROC-AUC was 0.7927. Paired BFN-minus-
metapredict ROC-AUC was -0.0376 (95% CI -0.0786 to 0.0056). The head therefore
has external discrimination but did not outperform the mature sequence-only
baseline and its probabilities are not externally calibrated.

### 3.9 The full-length Abeta42 de novo ceiling remained unbroken

A separate 3STB VHH panel used five ColabFold seeds for native, composition-
scrambled, legacy design, and graft controls against full-length Abeta42. All six
arms passed the monomer foldability gate, but none passed the multimer gate of
mean ipTM at least 0.25 and mean interface PAE at most 20 A. Native mean ipTM was
0.138, the composition scramble was 0.110, and all designed or grafted arms were
at or below 0.110. These results distinguish preserved VHH foldability from the
unresolved antibody-IDP interface problem and leave the de novo design ceiling
unbroken.

## 4 Discussion

The independent temporal evaluation found a positive native-versus-shuffle
ECLS effect after development on an exposed 46-cluster adaptation set. The
analysis tests sensitivity to peptide-coordinate context on deposited native
complex-derived backbones; it does not isolate an experimentally observed
unbound Fab state.

Candidate-generation experiments exposed a limitation that was not visible in
native-versus-shuffle controls. A universal contrast coefficient did not
generalize uniformly across generators. ProteinMPNN candidates were sampled
from the same complex-conditioned model used for evaluation, producing a strong
self-generation likelihood bias. Increasing the peptide-stripped-reference
coefficient corrected native retrieval in development. BFN and ESM-IF retained
`lambda=1` because no generator-specific adaptation pools were available; their
optimal coefficients were not assessed. These observations motivate
independent evaluation of prespecified generator-specific calibration.

The GCLC result remains exploratory. It used only seven antigen clusters, the
clusters had already been observed at the uncalibrated coefficient, and the
ProteinMPNN coefficient was selected on exposed adaptation data. Moreover,
native-spike retrieval asks whether a score recognizes the crystallographically
observed sequence among generated alternatives. It does not demonstrate that
the highest-ranked non-native candidate binds or improves an antibody.

The physical-ensemble extension further narrows the interpretation. A positive
mean ECLS effect survived across accepted local conformers, but the fraction of
positive clusters failed the frozen gate and deposited single poses performed
better. Because every trajectory started from the native deposited backbone,
the result characterizes a local native neighborhood and cannot establish
conformer recovery from sequence or coarse pose information.

The `T2` result shows that adding broad native-derived residue restraints did
not solve this limitation. The fixed perturbation protocol also placed 13 of 28
trajectories outside the prespecified RMSD tier, indicating heterogeneous
sampling across targets. The subsequent T2.1 v2 protocol used deterministic
torsion perturbation and recalibrated parameters. Although 25 of 31 selected
structures passed the overall validity criterion, random restraints
outperformed supplied contacts in the paired cluster-level analysis. T2.1
therefore does not rescue the proposed contact-specific mechanism; it provides
an auditable deterministic stress test whose positive within-arm recovery is
not mechanistically specific.

The CAID3 result supports external disorder discrimination, but not superiority
to metapredict or calibrated probabilities. Structure availability also limited
scoring to 83 of 184 cluster-resolved homology-independent targets, so the
result applies to the exact AFDB-mapped subset rather than all CAID3 proteins.

The corrected 4HIX comparison includes BFN and ProteinMPNN under matched design
positions and seeds. Native and composition shuffle were indistinguishable by
AF2, no arm exceeded native on mean candidate-median ipTM, and the six-residue
fragment and pre-AF2 deposition prevent a generalization or design-success claim.

The preregistered design study remains incomplete. Parent BFN, matched-position
ProteinMPNN, computational alanine scans, PRODIGY, developability, diversity,
and runtime were completed only for the 4HIX diagnostic. Matched-position
ESM-IF completed on seven exposed development antigen units, but this was not
confirmatory. A fresh SAbDab snapshot yielded 91 structurally eligible peptide
complexes. A v1 MMseqs2 audit against 4,129 training and prior-exposure records
yielded zero records under 50%-identity VH/VL gates, which were judged
inappropriate for germline-related antibody frameworks. Before new model
training or generator access, a v2 feasibility matrix froze 20 prospective
internal connected-component representatives at 90% VH/VL, 70% paired-CDR, 50%
H3, and 30% antigen identity with 80% coverage. The random-initialization
lineage completed Stage A and Stage B training. Raw generation subsequently
recorded all 2,400 frozen attempts across five arms without final failed slots,
followed by diversity-only selection of 280 candidates and 20 explicit ESM-IF
shortfall slots. Three-seed AF2 completed 960 predictions; PRODIGY scored 959,
with one no-contact failure. Only 7/20 components were valid across all arms.
The proposed arm did not exceed the componentwise oracle baseline (mean
baseline-minus-proposed delta -0.7635 kcal/mol; 95% component-bootstrap CI
-1.5063 to -0.0403), and all preregistered gates failed. This internal holdout
is not external confirmation, and the full design-paper claim cannot be made.

The public mutation audit retained six heterogeneous anti-Abeta records, five
of which were point mutations, but only one point mutation had an exact
uncensored numeric fold effect. This supports qualitative or censored checks,
not pooled quantitative correlation. No independent non-binder panel or wet-lab
validation was available. Composition-matched shuffles cannot be called
non-binders. The analysis does not support claims of binding, affinity,
specificity, biological activity, therapeutic efficacy, or prospective design
success. A future independent generator-stratified benchmark should freeze
generator-specific calibration before evaluation and pair computational ranking
with experimental candidate measurements.

## 5 Conclusion

ECLS provided positive native-versus-shuffle discrimination on deposited native
backbones across 46 exposed adaptation clusters and a one-time 15-cluster
temporal final. The deterministic torsion perturbation recovery benchmark
(T2.1 v2) achieved 25/31 overall structural validity and mean supplied-arm
held-out contact recovery of 0.581, but random restraints performed better and
the experiment did not establish informative-contact guidance. The corrected
4HIX matched diagnostic did not beat native or distinguish native from shuffle
by multi-seed AF2. CAID3 established external disorder discrimination but not
superiority to metapredict or probability calibration.
Universal ECLS did not establish cross-generator reranking superiority.
Exploratory generator-aware calibration produced native retrieval above random
on seven development clusters, but did not establish improvement over complex
NLL. The completed evidence supports a bounded computational sequence-scoring
study, not the preregistered design claim.

## Data and code availability

The reviewer package contains frozen YAML contracts, scoring and analysis
scripts, focused tests, candidate-level result evidence, figures, tables, and a
SHA256 manifest. T2.1 v2 per-structure results are provided in
`results_t2.1_v2_final.json`; matched 4HIX generation, AF2, PRODIGY, alanine,
and CAID3 results are provided under `results/ablation/`. An
integrative analysis document (`INTEGRATIVE_ANALYSIS.md`) summarizes the
complete evidence chain from ECLS detection through T2.1 torsion recovery to
4HIX design validation. Derived LMDB datasets and model weights are excluded
from the lightweight archive because of size and third-party provenance. Their
expected locations and reproduction commands are documented in
`publication/REPRODUCIBILITY.md`.

## Author contributions

To be completed by the authors.

## Funding

To be completed by the authors.

## Conflict of interest

To be completed by the authors.

## References

1. Capel HL, et al. SAbDab2: The structural antibody database in the age of machine learning. Preprint. 2026. doi:10.64898/2026.06.16.732554.
2. Dunbar J, et al. SAbDab: the structural antibody database. Nucleic Acids Res. 2014;42:D1140-D1146. doi:10.1093/nar/gkt1043.
3. Dauparas J, et al. Robust deep learning-based protein sequence design using ProteinMPNN. Science. 2022;378:49-56. doi:10.1126/science.add2187.
4. Hsu C, et al. Learning inverse folding from millions of predicted structures. Proc Mach Learn Res. 2022;162:8946-8970.
5. Graves A, et al. Bayesian Flow Networks. arXiv:2308.07037. 2023.
6. Hu Y, et al. AntibodyDesignBFN: High-Fidelity Fixed-Backbone Antibody Design via Discrete Bayesian Flow Networks. arXiv:2601.05605. 2026.
7. Evans R, et al. Protein complex prediction with AlphaFold-Multimer. bioRxiv. 2021. doi:10.1101/2021.10.04.463034.
8. Eastman P, et al. OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. PLoS Comput Biol. 2017;13:e1005659. doi:10.1371/journal.pcbi.1005659.
9. Maier JA, et al. ff14SB: Improving the accuracy of protein side chain and backbone parameters from ff99SB. J Chem Theory Comput. 2015;11:3696-3713. doi:10.1021/acs.jctc.5b00255.
10. Nguyen H, Roe DR, Simmerling C. Improved Generalized Born solvent model parameters for protein simulations. J Chem Theory Comput. 2013;9:2020-2034. doi:10.1021/ct3010485.
11. Regep C, et al. The H3 loop of antibodies shows unique structural characteristics. Proteins. 2017;85:1311-1318. doi:10.1002/prot.25291.
12. Miles LA, et al. Bapineuzumab captures the N-terminus of the Alzheimer's disease amyloid-beta peptide in a helical conformation. Sci Rep. 2013;3:1302. doi:10.1038/srep01302.
13. Efron B. Bootstrap methods: another look at the jackknife. Ann Stat. 1979;7:1-26. doi:10.1214/aos/1176344552.
14. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B. 1995;57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x.
15. Mehdiabadi M, et al. Critical Assessment of Protein Intrinsic Disorder Round 3: Predicting disorder in the era of protein language models. Proteins. 2026;94:414-424. doi:10.1002/prot.70045.
16. Emenecker RJ, Griffith D, Holehouse AS. Metapredict: a fast, accurate, and easy-to-use predictor of consensus disorder and structure. Biophys J. 2021;120:4312-4319. doi:10.1016/j.bpj.2021.08.039.
17. Xue LC, et al. PRODIGY: a web server for predicting the binding affinity of protein-protein complexes. Bioinformatics. 2016;32:3676-3678. doi:10.1093/bioinformatics/btw514.
