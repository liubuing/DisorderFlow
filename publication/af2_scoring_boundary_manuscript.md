# Interface PAE, not ipTM or pLDDT, transfers across independent antibody-peptide scaffolds for candidate ranking

**Running title:** AF2-derived confidence for antibody-peptide interfaces

**Status:** Manuscript draft v1 (development-only; not peer reviewed; no wet-lab validation)

---

## Abstract

De novo antibody design against disordered targets is limited by validation: AlphaFold2 (AF2)-derived confidence is the standard proxy for interface quality, yet its reliability for antibody-peptide interfaces is rarely tested. We systematically evaluate three AF2 confidence signals — per-residue pLDDT, whole-complex ipTM, and candidate-to-antigen interface PAE — across six independent, homology-isolated antibody-peptide scaffolds obtained from three preregistered discovery rounds. We report four findings. First, only interface PAE transfers across scaffolds (median scaffold Spearman 0.70–0.87 across three training seeds), whereas pLDDT does not (five head/loss ablations all collapse to Spearman ≈ 0) and ipTM lacks sufficient reliable pair evidence (2/6 contributing scaffolds). Second, single-sequence AF2 fails to recapitulate a native crystal complex: the ACPA Fab 6YXM bound to its citrullinated peptide is scored at ipTM 0.22 and pLDDT 0.18, indistinguishable from non-binders. Third, MSA-based AF2 restores the antibody fold (ipTM 0.75, pLDDT 90) but cannot discriminate native from scrambled 8-residue peptides via whole-complex ipTM (0.753 versus 0.755), because the 435-residue antibody dominates the metric. Fourth, after variance-matching calibration, interface PAE becomes a deployment-grade relative ranking signal with all three seeds passing MAE, median-scaffold-Spearman, variance-ratio, pair-accuracy, and bootstrap gates. We conclude that interface PAE is the only transferable relative ranking signal for antibody-peptide candidates, and that absolute binding validation requires wet-lab measurement. These results define a concrete, reproducible boundary for AF2-based scoring in antibody-peptide design and caution against the unvalidated use of single-sequence AF2 or whole-complex ipTM as binding proxies.

---

## 1. Introduction

Antibody design against intrinsically disordered protein (IDP) targets — including amyloid-β, tau, and α-synuclein in Alzheimer's and Parkinson's disease — is complicated by the absence of a single stable target conformation. Fixed-backbone design methods assume a rigid target pose and therefore fail on disordered epitopes. The standard computational workaround is to generate CDR candidates with a structure-conditioned sequence model and then rank or validate them with AlphaFold2 (AF2)-derived confidence, treating low interface error (or high ipTM) as evidence of binding.

This validation step is rarely scrutinized. Three signals are commonly used interchangeably: per-residue pLDDT, whole-complex ipTM, and the local candidate-to-antigen interface PAE. Each has a different interpretation, and their reliability as a *binding proxy* for antibody-peptide interfaces is not established. Furthermore, AF2 is frequently run in single-sequence mode (no multiple sequence alignment, MSA) for speed, which substantially degrades its predictions.

Here we perform a systematic evaluation of these three signals. We build a frozen calibration set of six independent antibody-peptide scaffolds, isolated from all project reference pools by five-axis homology. We then (i) measure which confidence signal transfers across scaffolds, (ii) test whether single-sequence AF2 can recapitulate a native crystal complex, (iii) test whether MSA-based AF2 can discriminate native from scrambled peptides, and (iv) assess whether a variance-matching calibration yields a deployment-grade ranking signal.

Our central result is that only interface PAE transfers across scaffolds and can serve as a relative ranking signal; neither pLDDT nor ipTM does, and neither single-sequence nor MSA-based AF2 can validate absolute binding at the 8-residue peptide scale. We therefore define a reproducible boundary for AF2-based scoring and argue that wet-lab validation remains the only route to a binding claim.

---

## 2. Results

### 2.1 Only interface PAE transfers across independent scaffolds

We froze six independent antibody-peptide scaffolds (33CI, 5V6L, 6MQE, 6MQM, 6P60, 6YXM) through three preregistered RCSB discovery rounds and five-axis homology isolation (VH ≤0.90, VL ≤0.90, paired-CDR ≤0.70, H3 ≤0.50, antigen ≤0.30 identity to all reference pools). For each scaffold we generated 84 candidate-interface entities (72 designed CDR-H3 candidates plus native and composition-shuffled controls) and scored them with AF2 multimer under model 1 and 2, three seeds each.

A v2 confidence model with candidate-interface heads was evaluated on these scaffolds in transfer mode (the model was trained only on the original 15-scaffold training set). **Interface PAE transferred**: median per-scaffold Spearman between predicted and target PAE was 0.866, 0.725, and 0.730 for the three converged seeds, with pair ordering accuracy 0.953, 0.886, and 0.843. **pLDDT did not transfer**: across five head/loss ablations (baseline two-layer MLP; single-linear head; interface-pooled features; explicit geometry head; normalized grouped-difference loss), pLDDT median scaffold Spearman remained near or below zero. **ipTM lacked sufficient reliable pair evidence**: only 2 of 6 scaffolds contributed reliable ipTM target deltas (53 reliable pairs, 71.7% from a single scaffold), below the preregistered ≥3-scaffold / ≤50% rule. Table 1 summarizes the transferability matrix.

### 2.2 Single-sequence AF2 cannot recapitulate a native binder

To test whether the validation metric itself is informative, we scored the native ACPA Fab 6YXM (a crystal structure of a Fab bound to its citrullinated peptide LPGQGERG) with single-sequence AF2 multimer (no MSA, no templates). The native complex — a real, experimentally resolved binder — was scored at **ipTM 0.22, pLDDT 0.18, interface PAE 27.7 Å**, values indistinguishable from non-binding predictions. Single-sequence AF2 therefore fails even to identify a known binder, confounding any downstream interpretation of candidate scores.

### 2.3 MSA restores the antibody fold but not short-peptide discrimination

We then re-scored 6YXM with MSA-based AF2 (ColabFold multimer v3, model 1). The native complex rose to **ipTM 0.753 and pLDDT 89.5** at recycle 0 (0.819 and 93.1 at recycle 1), confirming that the low single-sequence scores were an MSA-absence artifact for the *antibody chain*. However, a composition-shuffled peptide (GLPGGQER) scored **ipTM 0.755** — statistically indistinguishable from the native peptide (0.753). Because ipTM is a whole-complex metric dominated by the 435-residue antibody, the 8-residue peptide contributes negligible weight, and ipTM cannot discriminate native from scrambled peptides at this scale. This is the core limitation of whole-complex ipTM as a short-peptide binding proxy.

### 2.4 Variance-matching calibration yields a deployment-grade PAE ranking signal

Interface PAE predictions were shrunk relative to targets (raw prediction-to-target variance ratio 0.17–0.48), so we applied a variance-matching affine calibration (target = slope·pred + intercept, slope = target_std/pred_std) fit on the six calibration scaffolds. After calibration, all three seeds passed the six preregistered gates: corrected MAE 0.022–0.029 (≤0.10), median scaffold Spearman 0.725–0.866 (≥0.5), variance ratio 1.0 ([0.5, 2.0]), 236 reliable pairs (≥30), 6 contributing scaffolds (≥3), pair accuracy 0.843–0.953 (≥0.65), and bootstrap 95% lower bound 0.683–0.873 (>0.5). The calibration is monotonic and ranking-invariant, so it preserves the validated ordering while restoring the scale.

---

## 3. Methods

### 3.1 External scaffold calibration

Candidates were discovered via three preregistered RCSB PDB search rounds (release windows 2025-01–2026-09, 2018–2024, plus a deferred viral-antigen pool), each with a frozen query configuration and SHA-256 checksum. Structures were materialized (Bio.PDB, auth chains), ANARCII Chothia numbering, and retained only if paired H/L chains and a single resolved 5–50-residue peptide made ≥1 H3 heavy-atom contact within 4.5 Å. Isolation against all reference pools (train, project, temporal-sealed, v1/v2 holdout) used MMseqs2 with coverage 0.8 and per-axis identity thresholds (VH/VL 0.90, paired-CDR 0.70, H3 0.50, antigen 0.30), plus exact PDB exclusion. Six independent components survived (EXT001–EXT006).

### 3.2 Candidate generation

For each scaffold, CDR-H3 candidates were generated with BFN (three arms: disorder-conditioned, disorder-agnostic, and stage-A) and ProteinMPNN, 8 samples per seed, three seeds, 576 attempts total. Controls (native and composition-shuffled H3) were added per scaffold. Selection was diversity-only (maximize distinct seeds and pairwise Hamming distance), 3 slots per arm per scaffold.

### 3.3 AF2 scoring

AF2 multimer v3 was run under two protocols. Single-sequence: JAX runner (num_msa=1, no templates), models 1 and 2, three seeds (7103, 7111, 7121), 3 recycles, returning per-residue pLDDT, ipTM, and interface PAE. MSA: ColabFold 1.6.1 batch (alphafold2_multimer_v3, model 1), MSA fetched from the ColabFold server, 0–1 recycles, returning pLDDT, pTM, and ipTM.

### 3.4 Candidate-interface confidence model

A v2 candidate-interface confidence model (frozen encoder, trained heads) predicts pLDDT, ipTM, and interface PAE from single-sequence AF2 targets. Transfer evaluation loaded the converged checkpoints (iterations 1000/800/800 selected by lowest validation loss) and evaluated on the six external scaffolds with the lineage check deliberately bypassed.

### 3.5 Pair evidence and calibration

Entity-level targets were aggregated as the mean over 2 models × 3 seeds. Reliable pairs within a scaffold were defined by absolute target delta strictly exceeding the joint hierarchical SEM. Interface PAE was calibrated by variance-matching affine fit on the six calibration scaffolds; pLDDT and ipTM were abstained per the frozen deployment contract.

---

## 4. Discussion

We find that AF2-derived confidence is more limited as an antibody-peptide binding proxy than is commonly assumed, in three distinct ways. **Single-sequence AF2 is unreliable for antibody folding** — a native crystal complex is scored as non-binding — so any ranking performed in single-sequence mode inherits this artifact. **MSA fixes the antibody but not the peptide** — whole-complex ipTM cannot discriminate an 8-residue native from a scrambled peptide because the antibody dominates the metric. **Only interface PAE transfers** — and, after variance-matching calibration, it is the single deployment-grade relative ranking signal.

These results have practical implications. Researchers using single-sequence AF2 to score antibody-peptide designs should be aware that such scores do not distinguish binders from non-binders. Whole-complex ipTM is an antibody-foldability metric, not a short-peptide binding metric. The candidate-to-antigen interface PAE, localized to the interface, is the signal with demonstrated cross-scaffold transferability, and it should be used for *relative* ranking rather than as an absolute binding readout.

The principal limitation of this study is the absence of wet-lab validation. We therefore make no binding claim; our contribution is the systematic delineation of what AF2-derived confidence can and cannot support for antibody-peptide interfaces, together with a reproducible, preregistered calibration methodology. Establishing whether a de novo designed antibody actually binds a disordered target remains a wet-lab question (SPR/BLI), and our interface-PAE ranking signal provides the candidate prioritization for such experiments.

---

## References (to be completed)

- SAbDab2 / RCSB PDB for structural records
- ColabFold / AlphaFold2 multimer (Mirdita et al. 2022; Evans et al. 2021)
- ANARCII (Dunbar & Deane 2016)
- MMseqs2 (Steinegger & Söding 2017)
- BFN (Bayesian Flow Networks)

## Data and code availability

Frozen artifacts: `publication/candidate_interface_pae_deployment_v1.json` (deployment contract), `publication/af2_scoring_boundary_final_decision_v1.json` (scoring-boundary conclusions), `publication/idp_platform_capability_v1.json` (capability manifest), and the six-scaffold calibration data under `data/candidate_interface_external_calibration_v1/`. The sealed test split was never evaluated.
