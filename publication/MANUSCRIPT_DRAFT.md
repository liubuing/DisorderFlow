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
clusters, native CDR-H3 sequences showed a mean ECLS advantage of 0.217481 over
200 composition-matched shuffles (median 0.203056; cluster-bootstrap 95% CI
0.146050-0.289555), with positive effects in 80.4348% of clusters. A one-time
post-ProteinMPNN-training temporal final contained 31 structures from 15 antigen
clusters and yielded a mean advantage of 0.172281 (median 0.118699; 95% CI
0.059156-0.291920), with 80% positive clusters. In exploratory development on
seven antigen clusters, a universal ECLS reranker did not outperform complex
NLL. In a post hoc generator-aware analysis, pooled normalized native rank was
0.853175 and the nominal improvement over random was 0.353175 (95% CI
0.166667-0.496032; exact paired permutation P=0.046875), but gain over complex
NLL remained uncertain.

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

## 2 Materials and methods

### 2.1 Structural records and CDR-H3 mapping

Antibody-peptide structures were derived from the audited SAbDab2 resource.
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

ProteinMPNN `v_48_020` was used in backbone-only conditional-probability mode.
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
seven official antigen clusters. BFN, ProteinMPNN, and ESM-IF each generated
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

Means and 95% confidence intervals used 10,000 bootstrap resamples of official
antigen clusters. The frozen ECLS decisions used these confidence intervals and
prespecified effect gates. Supplementary post hoc sensitivity used an exact
two-sided sign-flip test for the 15-cluster temporal final and 1,000,000 seeded
Monte Carlo sign flips for the 46-cluster adaptation set. Exploratory GCLC used
exact sign-flip tests; generator-level descriptive comparisons were adjusted by
the Benjamini-Hochberg method, which does not correct the preceding adaptive
analysis selection. Leave-one-cluster-out
sensitivity recomputed the pooled mean after omitting each development cluster.
Candidates and generation seeds were not treated as biological replicates.

### 2.8 Local pose-neighborhood peptide ensembles

As an exploratory flexibility extension, each of the seven development
complexes was subjected to four independently seeded OpenMM restrained
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

### 3.6 Broad residue restraints did not recover moderate perturbations

Only three of seven clusters met the frozen `T2` replica-validity requirement.
Across those clusters, mean held-out contact recovery was -0.0171 (95% CI
-0.0513 to 0.0476; exact sign-flip `P=0.75`). Mean peptide-backbone RMSD
recovery was -0.1415 A (95% CI -0.4146 to 0.1135; `P=0.50`), indicating no
consistent movement toward the deposited pose. Mean final ECLS advantage was
0.1020, but its 95% CI (-0.0609 to 0.3729) included zero. The frozen `T2`
recovery gate was rejected.

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
sampling across targets. A future recovery study requires a separately frozen,
length-adaptive or torsion-targeted perturbation protocol and an independent
evaluation set; it cannot reinterpret the present negative result.

No experimental mutation-effect set or non-binder panel was available, and no
wet-lab validation was performed. Composition-matched shuffles cannot be called
non-binders. The analysis does not support claims of binding, affinity,
specificity, biological activity, therapeutic efficacy, or prospective design
success. A future independent generator-stratified benchmark should freeze
generator-specific calibration before evaluation and pair computational ranking
with experimental candidate measurements.

## 5 Conclusion

ECLS provided positive native-versus-shuffle discrimination on deposited native
backbones across 46 exposed adaptation clusters and a one-time 15-cluster
temporal final. Universal ECLS did not
establish cross-generator reranking superiority. Exploratory generator-aware
calibration produced native retrieval above random on seven development
clusters, but did not establish improvement over complex NLL. The resulting
method and benchmark are suitable for computational sequence-scoring studies,
with claims bounded to the evaluated retrospective tasks.

## Data and code availability

The reviewer package contains frozen YAML contracts, scoring and analysis
scripts, focused tests, candidate-level result evidence, figures, tables, and a
SHA256 manifest. Derived LMDB datasets and model weights are excluded from the
lightweight archive because of size and third-party provenance. Their expected
locations and reproduction commands are documented in
`publication/REPRODUCIBILITY.md`.

## Author contributions

To be completed by the authors.

## Funding

To be completed by the authors.

## Conflict of interest

To be completed by the authors.

## References

References to SAbDab/SAbDab2, ProteinMPNN, ESM-IF, BFN, antibody CDR-H3
modeling, bootstrap inference, and Benjamini-Hochberg correction must be added
from verified bibliographic records before submission.
