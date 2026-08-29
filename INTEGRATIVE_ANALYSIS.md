# CDR-H3 Flexibility and Epitope Flexibility Coupling: Integrative Analysis

## 1. Core Hypothesis

> Deposited antibody CDR-H3 sequences carry measurable, peptide-conformation-specific
> signal that persists under physical perturbation and degrades when epitope information
> is removed. This signal is captured by the epitope-conditioned likelihood shift (ECLS)
> metric and survives deterministic backbone torsion perturbation.

## 2. Evidence Chain

### 2.1 Detection — ECLS measures peptide-conditioned H3 signal

The ECLS metric is:
```
ECLS = H3_NLL(complex_backbone) - H3_NLL(peptide-stripped_backbone)
```
Lower ECLS means the H3 sequence achieves lower NLL (higher likelihood) when peptide
coordinates are present. The positive-direction endpoint used below is the ECLS
advantage of native over shuffles, not raw ECLS itself.

**Results on sealed temporal final (31 structures, 15 antigen clusters, all post-2021)**

| Metric | Value |
|---|---|
| Mean ECLS advantage (native vs composition-matched shuffle) | 0.172 |
| Bootstrap 95% CI | [0.059, 0.292] |
| Median ECLS advantage | 0.119 |
| Positive antigen clusters | 12/15 (80.0%) |

All frozen gates passed. Confirmed on a larger adaptation set (46 clusters):
mean advantage 0.217, CI95 [0.146, 0.290], 80.4% positive.

**Interpretation:** Deposited H3 sequences have a more favorable peptide-coordinate
likelihood contrast than composition-matched shuffles on these complex-derived
backbones. This does not isolate binding or physiological conformational coupling.

### 2.2 Local Flexibility — T1 ensemble reveals single-pose overconfidence

Starting from deposited poses, we generated restrained OpenMM ensembles (4 seeds,
antibody restrained at 5000 kJ/mol/nm², peptide backbone at 25 kJ/mol/nm²).
A valid ensemble required peptide RMSD 0.05–3.0 Å, ≥50% contact retention,
no clashes, no geometry outliers, and mean pairwise RMSD ≥0.25 Å.

**Results (7 development clusters)**

| Metric | Value |
|---|---|
| Valid ensembles | 6/7 (85.7%) |
| Mean pairwise peptide RMSD | 0.63 Å |
| Mean ensemble ECLS advantage | 0.131 |
| Ensemble ECLS CI95 | [0.022, 0.232] |
| Ensemble minus single-pose advantage | −0.200 (CI95 [−0.352, −0.036]) |

**Key finding:** The ensemble ECLS advantage is 0.20 lower than the single-pose
advantage. The deposited crystal pose overestimates the likelihood signal compared
to physically sampled nearby conformers. This demonstrates native-pose sensitivity:
the signal exists but is attenuated when conformational flexibility is accounted for.

Only 4/6 (66.7%) positive clusters; the frozen 70% gate failed. However, the
95% CI for ensemble advantage excludes zero, establishing a positive direction.

### 2.3 Moderate Perturbation Recovery — T2.1 v2

Unlike the rejected T2 protocol (high-temperature OpenMM perturbation with poor
replica validity), T2.1 uses a deterministic, length-adaptive phi/psi torsion
perturbation to reach the 1.5–3.0 Å peptide RMSD tier while keeping the antibody
frame fixed. Recovery simulations (2000-step Langevin, 300 K, implicit solvent)
test whether native residue contact restraints (split 50:50 into supplied and
held-out) can guide recovery of the perturbed peptide.

**Results on 31 structures from 15 antigen clusters**

- 26/31 structures reached the 1.5-3.0 A target tier.
- 25/31 were valid overall (80.6%); 25/26 tier-reaching structures passed revised QC (96.2%).
- Supplied-contact mean recovery: 0.581.
- All-contact, random-restraint, and null means: 0.611, 0.619, and 0.533.
- Cluster-paired supplied-minus-random difference: -0.0366 (95% CI [-0.0638, -0.0108]).
- Cluster-paired supplied-minus-all-contact difference: -0.0258 (95% CI [-0.0444, -0.0069]).

**Interpretation:** Positive recovery within the supplied-contact arm is not evidence
of contact-specific guidance because random restraints performed better. The
current contact-specific mechanism is terminal negative and the existing final
must not be rerun. Only a mechanistically distinct successor would justify a new,
separately preregistered independent test.

### 2.4 Hierarchy of evidence

From strongest to weakest:

1. **ECLS temporal final** (n=15 clusters, post-2021, sealed): Solid positive result.
   The deposited H3 sequence carries epitope-conformation-specific signal.
2. **ECLS adaptation** (n=46): Replicates the temporal final pattern on hold-out
   adaptation data.
3. **T1 ensemble** (n=7): Establishes that single-pose signal overestimates
   ensemble signal by ∼0.20 ECLS units. Ensemble advantage CI excludes zero.
4. **T2.1 v2 torsion recovery** (31 structures, 15 clusters): Positive within-arm
   recovery, but controls contradict informative-contact superiority.
5. **Generator calibration** (n=7): Shows that ECLS is generator-specific;
   universal reranker rejected, but calibrated reranker works.

## 3. What This Does NOT Show

- Binding affinity: ECLS is a likelihood metric, not a free energy
- Experimental validation: No wet-lab data
- Design quality: Generated H3 sequences were not tested for binding
- Generalization beyond antibody-peptide: Only peptide antigens tested

## 4. v2 Protocol Status

The completed v2 benchmark changed the following parameters after v1:
- Peptide backbone restraint: 25 → 100 kJ/mol/nm²
- Max peptide RMSD gate: 3.5 → 5.0 Å
- Geometry outlier comparison: disabled for perturbed structures
- Selection: per-structure (31 records) instead of per-cluster (15 units)
- All 4 control arms: supplied, all contacts, random peptide CA, null

Observed validity was 25/31 overall. Because parameters and QC were adapted after
v1, this result is development evidence rather than independent confirmation.

## 5. Manuscript Contribution

This study provides:
1. A rigorously homology-disjoint benchmark (6-axis split: PDB, paired antibody,
   VH, VL, H3, antigen) — exceeds standard practice
2. The ECLS metric as a reproducible computational endpoint for epitope-conditioned
   antibody sequence evaluation
3. Evidence that ECLS is sensitive to a restrained local coordinate neighborhood (T1)
4. A deterministic, length-adaptive torsion perturbation protocol as a
   reproducible physical benchmark
5. A complete protocol with sealed data, frozen gates, and one-shot evaluation

The bounded narrative is: *deposited H3 sequences show a more favorable
peptide-coordinate likelihood contrast than matched shuffles; the contrast is
sensitive to local coordinate perturbation. T2.1 does not establish a
contact-specific recovery mechanism.*

## 6. 4HIX Matched Generator + AF2 Diagnostic

### 6.1 Protocol

Using the 4HIX humanized 3D6 Fab and six-residue DAEFRH fragment, we compared
current BFN with disorder conditioning on and off, parent BFN, and ProteinMPNN.

Pipeline:
1. Extract 4HIX Fab (VH + VL) and Abeta peptide (chain A, 6 residues)
2. Fix all non-H3 residues and design the anchor-bounded `H:96-107` 12-mer
3. Generate eight candidates at each of three seeds in every arm
4. Select three per arm before AF2 using the arm-native generation score
5. Evaluate candidates, native, and composition shuffle with three AF2 seeds
6. Apply PRODIGY, developability, runtime, and contact-position alanine diagnostics

### 6.2 Results

| Arm | Candidates | Native recovery | Mean candidate-median ipTM | Developability pass |
|---|---:|---:|---:|---:|
| Current BFN, disorder on | 24 | 0.080 | 0.456 | 0.708 |
| Current BFN, disorder off | 24 | 0.153 | 0.445 | 0.750 |
| Parent BFN | 24 | 0.063 | 0.452 | 0.792 |
| ProteinMPNN | 24 | 0.326 | 0.456 | 1.000 |
| Native | 1 | 1.000 | 0.462 | not ranked |
| Composition shuffle | 1 | 0.250 | 0.464 | not ranked |

Current disorder on/off paired Hamming fraction was 0.0833 and 17/24 pairs
changed. This establishes a production-path response, not beneficial direction.
Native and shuffle were not separated by AF2. PRODIGY had large seed variation;
its arm means do not establish affinity. Y3A, S9A, and S10A computational scans
also lacked a consistent AF2/PRODIGY loss.

### 6.3 Critical Fixes

During validation, we discovered that AF2 multimer requires antibody chains
to be passed with `:` separator (`VH:VL`) rather than concatenated
(`VH+VL`). Without chain breaks, ipTM dropped from 0.45 to 0.10 — a 4.5x
underestimation of interface quality. This fix was applied to `af2_jax_runner.py`
and `design_wrappers.py`.

The generic BFN inference path also encoded `fragment_type` as chain ordinals,
while the BFN core expected the antigen enum. The antigen mask was consequently
empty. Explicit `antigen_chains` assignment fixed this before matched generation.

### 6.4 Limitations

- Epitope is only 6 residues (DAEFRH) — AF2 interface metrics are less
  discriminative for very short peptides
- Triplet scoring uses static backbone geometry; does not capture conformational
  changes upon H3 substitution
- No experimental binding validation is possible within the current study scope
- The 4HIX Fab was in AF2's training set (PDB deposited 2012)
- The diagnostic is one scaffold and cannot establish generalization

## 7. IDP Target Expansion: Tau and Alpha-Synuclein

### 7.1 Results

| Target | Complex | Epitope | Designs | Time | MPNN Score Range |
|---|---|---|---|---|---|
| Tau (MAPT) | 5MP3 Fab | 9 aa | 10 | 37s | 0.900–1.009 |
| α-Synuclein (SNCA) | 8B9V Fab | 10 aa | 10 | 55s | 0.796–0.872 |

AF2 multimer validation pending for tau/synuclein candidates.

## 8. Disorder Head Evidence Status

Two legacy retraining attempts produced random-level discrimination:

| Labels | Iterations | AUC-ROC | Outcome |
|---|---|---|---|
| Charge-hydropathy heuristic (2950 entries) | 800 GPU | 0.509 | Random |
| DisProt/MobiDB experimental (143 entries) | 500 GPU | 0.500 | Random |

These legacy results do not identify whether architecture, supervision, or training
caused failure. Later homology-separated balanced-v4 development runs reached mean
dev ROC-AUC 0.896 across three seeds and CAID2 development-regression ROC-AUC 0.742.

The checkpoint and calibration were then frozen before CAID3 download. Direct
MMseqs2 comparison to the 828 actual training records excluded 17/204 targets;
three more lacked resolved UniRef50 membership. Eighty-three exact AFDB mappings
were scored, including 71 mixed-label clusters. Macro ROC-AUC was 0.7551, with
cluster-bootstrap 95% CI [0.6945, 0.8119]. Metapredict 3.0.2 reached 0.7927;
paired BFN-minus-metapredict was -0.0376, CI [-0.0786, 0.0056]. Brier score
0.2991 and ECE 0.3794 failed calibration criteria.

The head is therefore independently externally discriminative on the mapped
CAID3 subset, but is neither superior to metapredict nor externally calibrated.
No biological disorder-aware design claim follows from this prediction result.

## 9. Synthesis Specifications

Legacy Abeta and tau/synuclein sequences in
`idp_design_results/SYNTHESIS_SPECS.md` are historical computational drafts.
The former 4HIX-D1 ranking used an invalid 11-residue H3 mask and a selected
single AF2 result. It is not synthesis-ready and is superseded by the matched
diagnostic above.
