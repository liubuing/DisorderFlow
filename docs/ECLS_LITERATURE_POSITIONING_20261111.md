# ECLS cross-architecture paper: closest work and positioning

Date: 2026-11-11. Search sources: Europe PMC (published + preprints),
Crossref/DOI records. No exact match for the full ECLS package was found.

## Closest work

### 1. AGZArank — closest preprint

**Sadykov Z, Khamidullina A, Sultankulov B, Seitkali D.** *AGZArank:
Investigating epitope-conditioned antibody binder ranking with
structure-derived synthetic supervision.* bioRxiv 2026.
DOI: 10.64898/2026.06.08.730711.

- A geometric deep-learning **ranker** trained with structure-derived
  synthetic supervision (normalized pseudo-energy targets).
- Benchmark: 45 experimentally validated antibody–antigen interfaces;
  top-10 retrieval 44.4%; better post-2021 generalization than ProteinMPNN,
  ESM-IF and PRODIGY.
- Ablation: ranking depends on training scale and objective–evaluation
  alignment, not architecture alone.

**Overlap:** the phrase "epitope-conditioned antibody binder ranking" and the
post-2021 comparison to ProteinMPNN/ESM-IF.

**Difference:** AGZArank *trains a ranker*; it does not test the paired
complex-minus-stripped likelihood contrast, native-versus-shuffle controls,
or a controlled pre-2021 training regime. Our paper is a replication/boundary
study of the contrast, not a new ranker. Its scale/objective ablation is
consistent with our M3 finding that a 3-epoch model is null while a 10-epoch
model replicates.

### 2. Antigen-aware inverse folding benchmark — closest published work

**Janusz B, Chomicz D, Wrobel S, Dudzic P, Polasa A, Martin K, Darnell S,
Comeau SR, Krawczyk K.** *Benchmarking antigen-aware inverse folding methods
for antibody design.* Bioinformatics Advances 2026.
DOI: 10.1093/bioadv/vbag081. PMC13140557.

- Benchmarks inverse-folding methods (including AntiFold/ProteinMPNN lineage)
  for binder enrichment and affinity correlation.
- Key conclusions: inverse folding can enrich binders from a pool, but the
  methods make **very limited use of the antigen**; affinity correlations are
  weak; structural validity acts as a statistical shortcut; models are largely
  insensitive to the target's chemical environment.

**Overlap:** the general boundary claim "inverse folding is weak for
antigen-specific binding/affinity" is already published here. We must not
present that as novel.

**Difference:** this benchmark tests binder enrichment and affinity
correlation across methods; it does not test the **paired
complex-minus-stripped contrast**, the **native-versus-shuffle** estimand, or
a **controlled pre-2021 from-scratch replication**. Our contribution is the
replication of that specific contrast and its dissociation from affinity, not
the broad claim that inverse folding is weak.

### 3. Aβ conformational selection — closest to the CSN Aβ paper

**Xu H, Zheng Y, Bai G, Wang Y, Ma B.** *Differential conformational
selections of three therapeutic antibodies binding to polymorphic Aβ
oligomers.* Int J Biol Macromol 2026.
DOI: 10.1016/j.ijbiomac.2026.150222.

- Simulation study of aducanumab, lecanemab and donanemab binding Aβ
  oligomers; reports different recognition modes and conformational
  selection.

**Overlap:** the concept "different Aβ antibodies select different
conformations" is already in the literature.

**Difference:** this is a simulation study of three therapeutic antibodies
against oligomers. The CSN Aβ paper uses **deposited crystal structures from
nine independent lineages**, a preregistered register-aware RMSD analysis,
and a free-state NMR comparison. The structural evidence and the lineage-level
claim are different; cite and differentiate.

### 4. Task-specific affinity/kinetics models — adjacent, not competing

- **SimBinder-IF** (Zhao X et al., Bioinformatics 2026,
  DOI 10.1093/bioinformatics/btag548): fine-tunes ESM-IF with preference
  optimization for affinity; reports a modest affinity-rank improvement
  (Spearman 0.26 → 0.32; assay-level p = 0.188).
- **ProteinDPO** (Nat Methods 2026, DOI 10.1038/s41592-026-03137-3): aligns
  generative models to experimental fitness.
- **AlphaBind** (mAbs 2025, DOI 10.1080/19420862.2025.2534626) and
  **AbDist** (mAbs 2026, DOI 10.1080/19420862.2026.2644655): affinity
  prediction/benchmarking.
- **MAHLER** (Teng D et al., bioRxiv 2026,
  DOI 10.64898/2026.06.10.731383): inverse folding + metadynamics for
  antibody–antigen kinetics.

**Overlap:** they show that task-specific training can extract some
affinity/kinetics signal. This is consistent with our AAYL null: the
**zero-shot ECLS contrast** is not an affinity ranker; a model trained for
affinity is a different task.

**Difference:** we do not claim that no model can rank affinity. We claim that
the ECLS contrast, as defined and replicated, does not.

### 5. Sequence-recovery benchmarks

- **Benchmarking inverse folding models for antibody CDR sequence design**
  (PLoS One 2025, DOI 10.1371/journal.pone.0324566) — sequence recovery and
  mutation-effect metrics; background for M2.
- **Evidence-aware comparison of sequence-centric machine learning for
  antibody discovery and optimization** (Brief Bioinform 2026,
  DOI 10.1093/bib/bbag459) — benchmark rigor; useful framing reference.

## What is genuinely novel in our paper

1. **The paired complex-minus-stripped contrast (ECLS)** is not the same
   estimand as binder enrichment or affinity prediction. We found no paper or
   preprint that tests it across architectures.
2. **Cross-architecture replication** with AntiFold/ESM-IF1 on the frozen
   temporal panel (M2).
3. **Controlled pre-2021 from-scratch replication** on 4,506 PDBs / 8,585
   chains (M3), with zero PDB overlap and an explicit H3-overlap sensitivity.
4. **A quantitative dissociation from affinity**: 508 AAYL variants, M3 ECLS
   Spearman +0.009 / −0.002 (both CIs span zero).

## What is no longer novel by itself

- "Inverse folding makes limited use of antigen and correlates weakly with
  affinity." This is already the conclusion of the 2026 antigen-aware
  benchmark. Do not claim it as a new discovery; cite it as prior work and
  position our result as a **replication of a different estimand** plus a
  **controlled-exposure test**.
- "Different Aβ antibodies select different conformations." Already studied
  by simulation (IJBM 2026). Our CSN Aβ contribution is the deposited-structure
  lineage analysis, not the concept itself.

## Required manuscript changes

1. Add a Related Work paragraph naming AGZArank and the antigen-aware
   benchmark; state explicitly that the broad "inverse folding is weak"
   claim is prior work.
2. Frame the novelty as: **paired contrast + cross-architecture +
   controlled exposure + affinity dissociation**.
3. In the Discussion, distinguish zero-shot likelihood contrasts from
   task-specific learned rankers (AGZArank, SimBinder-IF, ProteinDPO). Do not
   overgeneralize the AAYL null.
4. In the CSN Aβ manuscript, cite IJBM 2026 and differentiate the
   deposited-structure/lineage analysis from the simulation study.
5. Add a sentence to the abstract: prior benchmarks show weak antigen use;
   here we ask whether the paired contrast itself replicates and whether it
   tracks affinity.

## Search boundary

- Europe PMC (MED + PPR), Crossref/DOI. Semantic Scholar was rate-limited.
- Search terms included "epitope-conditioned", "peptide-stripped",
  "complex/apo", "native/shuffle", "inverse folding + affinity", "Aβ
  conformer", "AGZArank".
- No exact match for the ECLS contrast/replication package was found.
- Preprint status can change; re-check before submission.
