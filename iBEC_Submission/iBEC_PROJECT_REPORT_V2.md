# DisorderFlow: An Auditable Platform for Structure-Conditioned Antibody Design against Disordered Epitopes

## Team 28 - Analyzing Disorder

**Members:** Chen Haoyang (Captain), Hu Jing, Dai Tanyu, Sun Yuchao  
**Institution:** Qilu University of Technology  
**Primary track:** Bioinformatics Resources / Platforms  
**Secondary alignment:** AI-driven Life Science Discovery  
**Submission date:** September 2026

## 1. Executive Summary

Intrinsically disordered proteins and flexible peptide epitopes are central to Alzheimer's disease, Parkinson's disease, and other protein-misfolding disorders. Their conformational heterogeneity creates a difficult engineering problem: a candidate antibody sequence should not be judged from one rigid antigen pose or one uncalibrated confidence score.

DisorderFlow is an auditable computational platform for structure-conditioned CDR-H3 analysis and candidate prioritization against flexible peptide antigens. It integrates a Bayesian Flow Network (BFN), epitope-conditioned likelihood shift (ECLS), multi-conformation scoring, ProteinMPNN and ESM-IF baselines, AlphaFold-Multimer structural triage, developability filters, and frozen evidence contracts in a Gradio application and command-line workflow.

The strongest general result is a one-time temporal ECLS final on 31 antibody-peptide structures aggregated into 15 antigen clusters. Native CDR-H3 sequences achieved a mean advantage of 0.1723 over 200 composition-matched shuffles per record, with a cluster-bootstrap 95% confidence interval of [0.0592, 0.2919] and 12/15 positive clusters. The result supports a bounded structural sequence-scoring claim, not experimental binding.

We also completed an end-to-end amyloid-beta engineering case study. A frozen funnel consolidated 352 unique candidates including the native control, retained 77 after computational gates, selected 24 diverse candidates, completed 81/81 three-seed AlphaFold-Multimer predictions for 24 candidates and three controls, and produced 12 sequence-diverse final computational candidates. All 12 were converted into auditable secreted scFv construct plans. Because no wet-lab labels are available, the binder-confidence module correctly abstains instead of converting AF2 or PRODIGY scores into false binding probabilities.

Two iBEC-specific audits further strengthened the project. A frozen 20-component analysis of 2,400 completed generation attempts showed that the BFN arm substantially increased position entropy and pairwise Hamming diversity relative to ProteinMPNN, but reduced the developability pass fraction by 16.5 percentage points. We therefore report broader exploration with a quality tradeoff, not unconditional superiority or an isolated disorder-conditioning effect. Evidence auditing of the 1,289-record, five-conformation dataset found 219 records with exact-sequence DisProt disorder-region evidence; prediction and AF2 proxy labels are reported separately. A correctly masked Tau/5MP3 extension then evaluated 275 unique H3 candidates including the native control across five poses, retained 163 after all preregistered gates, and produced eight diverse computational candidates.

## 2. Challenge and Significance

Many antibody-design systems assume a stable antigen structure. This assumption is poorly matched to disordered or conformationally heterogeneous peptide targets such as amyloid-beta, tau, and alpha-synuclein. The core challenge is not merely generating diverse sequences. It is preserving antigen context, comparing multiple conformations, preventing evaluator leakage, and reporting uncertainty when computational evidence is insufficient.

DisorderFlow addresses this gap as an engineering platform rather than a single score. The platform separates generation, structural conditioning, counterfactual controls, multi-seed structural triage, developability, evidence classification, and experimental handoff. This separation is important for safe AI-assisted life-science discovery because it prevents software availability from being mistaken for biological validation.

## 3. Originality and Innovation

### 3.1 Epitope-Conditioned Likelihood Shift

ECLS tests whether peptide coordinates preferentially support an observed H3 sequence:

`ECLS = NLL(complex backbone) - NLL(peptide-stripped backbone)`

The positive-direction native advantage is the mean ECLS of composition-matched shuffled H3 controls minus native ECLS. The comparison preserves sequence composition and changes only residue order, reducing trivial composition effects.

### 3.2 Disorder- and Position-Sensitive Routing

The BFN receiver contains antigen-aware pair routing, position-sensitive disorder conditioning, contact decoding, and a state-compatibility output. Explicit antigen-chain assignment is required. The software refuses to merge antibody and antigen structures from unrelated coordinate frames.

### 3.3 Fail-Closed Confidence Design

Legacy BFN pLDDT, ipTM, and PAE outputs are not used as primary within-scaffold ranking features because their variant-level dynamic range was found to collapse. The new binder-confidence contract freezes computational features but returns `abstain_not_trained` until blinded expression and binding labels pass quality gates. This abstention mechanism is a deliberate engineering contribution.

### 3.4 Evidence Contracts

Every confirmatory or prospective workflow freezes input hashes, checkpoints, random seeds, masks, baselines, thresholds, and claim boundaries. Negative controls and failed gates remain visible rather than being silently removed.

## 4. Platform Architecture

DisorderFlow provides the following modules:

1. BFN fixed-backbone sequence generation and fixed-candidate scoring.
2. Explicit heavy, light, and antigen chain-role assignment.
3. ECLS native-versus-counterfactual benchmarking.
4. ProteinMPNN and ESM-IF matched baselines.
5. Multi-pose and multi-seed aggregation.
6. AlphaFold-Multimer and PRODIGY computational structural triage.
7. Sequence developability, glycosylation, charge, hydrophobicity, and diversity gates.
8. Gradio interface, batch processing, command-line runners, frozen YAML contracts, and SHA256 manifests.
9. Experimental-handoff schemas and binder-confidence abstention.

The default application can be launched through `python manage.py start`. The core auditable BFN entry point is `modules/bfn_loader.py::run_bfn_design`.

## 5. Technical Rigor

### 5.1 Data Isolation

Publication-grade splits use connected components over PDB identity, paired-antibody identity, VH homology, VL homology, H3 homology, and antigen homology. Candidate-level rows from one component are never treated as independent biological replicates.

### 5.2 Controls

Controls include native H3, 200 composition-matched shuffles, antigen-stripped structures, ProteinMPNN, ESM-IF, parent BFN, multi-seed AF2, and independent sequence-developability checks. Shuffled sequences are explicitly described as counterfactual controls, not experimental non-binders.

### 5.3 Statistics

The temporal ECLS result is aggregated by antigen cluster and uses 10,000 cluster-bootstrap resamples. The final was evaluated once and is marked terminal; reruns are prohibited.

### 5.4 Negative Findings

The seven-cluster cross-generator experiment did not establish ECLS as a universal reranker. A 20-component internal multiscaffold design experiment failed all preregistered design gates. Random restraints outperformed supplied contacts in the terminal T2.1 mechanism test. These findings define the system's applicability boundary and guided the fail-closed candidate workflow.

## 6. Result 1: Frozen ECLS Temporal Final

| Metric | Result |
|---|---:|
| Structures | 31 |
| Independent antigen clusters | 15 |
| Shuffles per record | 200 |
| Mean native advantage | 0.172281 |
| Median native advantage | 0.118699 |
| Cluster-bootstrap 95% CI | [0.059156, 0.291920] |
| Positive clusters | 12/15 (80%) |

All frozen primary gates passed. The allowed interpretation is that deposited native H3 sequences have a more favorable peptide-coordinate likelihood contrast than composition-matched shuffles on the frozen temporal panel. ECLS does not measure affinity or experimental binding.

## 7. Result 2: Amyloid-Beta Engineering Case Study

The updated 4HIX/3D6 case study uses the correct 12-residue mask `H:96-107` with native H3 `VRYDHYSGSSDY`. The antigen coordinates enter the model through an explicit antigen chain. Five template-transferred Aβ42 poses plus the deposited six-residue complex were used as computational structural states.

### 7.1 Candidate Funnel

| Stage | Count |
|---|---:|
| Source memberships | 481 |
| Unique candidates | 352 |
| Pass frozen computational gates | 77 |
| Diverse pre-AF2 shortlist | 24 |
| AF2 entities including controls | 27 |
| AF2 predictions completed | 81/81 |
| Pass AF2/PRODIGY gates | 16 |
| Final computational shortlist | 12 |

The 12 final candidates are unique and have minimum pairwise Hamming distance 2. Native 3D6 had median AF2 ipTM 0.4762. Final candidates had median ipTM 0.4763-0.5034 across three seeds, with seed ranges of 0.0030-0.0087. Median interface PAE remained approximately 26, so the result is treated as computational prioritization rather than binding validation.

### 7.2 Engineering Honesty

All 12 final candidates came from the conservative mutation library; none of the 216 direct BFN generation attempts reached the final shortlist. We report this result because it identifies the current role of BFN: antigen-state scoring and hypothesis support are more reliable than unconstrained generation for this scaffold.

## 8. Result 3: External Disorder-Head Evaluation

The frozen external CAID3 evaluation retained 71 mixed-label UniRef50 clusters. The BFN disorder head achieved macro ROC-AUC 0.7551 with bootstrap 95% CI [0.6945, 0.8119]. Metapredict achieved 0.7927. External calibration failed (Brier 0.2991; ECE 0.3794), so the head is exposed as a routing score rather than a calibrated biological probability.

## 9. Result 4: Generator Behavior and Dataset Evidence

### 9.1 BFN versus ProteinMPNN

The frozen behavior panel contained 20 independent antibody-peptide components, five generator arms, three seeds, and 2,400/2,400 successful attempts.

| Metric | BFN | ProteinMPNN | Paired difference, 95% CI |
|---|---:|---:|---:|
| Unique fraction | 0.833 | 0.750 | 0.083 [-0.008, 0.190] |
| Position entropy, nats | 2.405 | 0.346 | 2.059 [2.009, 2.113] |
| Pairwise Hamming fraction | 0.934 | 0.217 | 0.717 [0.687, 0.750] |
| Developability pass fraction | 0.825 | 0.990 | -0.165 [-0.248, -0.088] |

Self-perplexity was not compared across model families. The supported interpretation is that BFN expands H3 sequence exploration but requires developability and external structural filtering.

### 9.2 Five-Conformation Dataset Audit

All 1,289 retained records contain five conformations. Exact sequence comparison to a local 3,337-entry DisProt snapshot identified 219 records with curated disorder regions: 23 highly disordered, 36 partially disordered, 99 containing an IDR of at least 20 residues, and 61 with shorter disorder regions. Nine additional records had prediction-only MobiDB-lite evidence, five had AF2 pLDDT proxy evidence, and 1,055 had no source disorder evidence. The prior phrase “800+ natural IDPs” is deprecated.

The source LMDB contained 1,301 records. The missing 12 records were source keys 1289-1300, all folded records of length 495-500; the build mechanically stopped at 1,289 rather than applying biological filtering.

### 9.3 Tau Second-Target Extension

The 5MP3 DC8E8 Fab-Tau complex provided a real bound geometry. Independent IMGT and structural-anchor checks identified the complete 13-residue H3 `ARDYYGTSFAMDY`. Five transferred Tau NMR poses passed geometry checks with zero severe clashes and 19-27 contacts per pose. All five showed antigen-conditioned BFN response; mean complex-minus-stripped state compatibility was 0.0634 and the across-pose output range was 0.0480. Historical candidates using an incorrect 10-position mask were invalidated.

The frozen extension combined 200 conservative variants, 40 composition-shuffled controls, three-seed ProteinMPNN generation, and three-seed BFN generation. It yielded 275 unique sequences including native; 163 passed all source, 1-4 mutation, five-pose BFN noninferiority, contact-map, and developability gates. Diversity selection produced eight candidates with minimum pairwise Hamming distance 2. Their mean complex-minus-stripped scores ranged from 0.0615 to 0.0779, compared with 0.0634 for native. All eight came from the conservative library, so this is a second-target computational prioritization result, not evidence that either generator improves binding.

## 10. Engineering Design and Translation Potential

The software is modular, runnable on Windows plus WSL2 GPU, and provides web and command-line interfaces. The Aβ shortlist was translated into a secreted scFv construct contract: signal peptide-VH-(G4S)3-VL-G4S-His6. Twelve candidates, native 3D6, and a composition-matched counterfactual passed sequence-boundary and translation checks. Blinded BLI layouts, expression-batch tracking, acceptance criteria, and measurement import schemas were generated.

These artifacts demonstrate a complete computational-to-experimental handoff, although no synthesis or measurement is claimed. Reference DNA is intentionally marked not order-ready because its GC content requires vendor optimization.

## 11. Reproducibility and Transparency

- Frozen YAML contracts define every prospective run.
- SHA256 hashes bind checkpoints, structures, configs, scripts, and outputs.
- Focused workflow tests and package-integrity checks pass.
- The frozen ECLS reviewer manifest lists 68 checksummed files.
- GPU and external-tool runs preserve one result row per requested prediction slot.
- The 81-slot AF2 panel completed with zero missing results.
- `binder_confidence_v1` currently reports zero experimental labels, null probability output, and `abstain_not_trained`.

## 12. Limitations and Responsible Claims

There is no wet-lab validation. AF2 and PRODIGY are computational plausibility filters, not affinity measurements. The Aβ poses are computational template transfers, not experimental monomer or oligomer complexes. The final candidates are hypotheses. DisorderFlow does not claim therapeutic efficacy, binding, specificity, or calibrated binder probability.

## 13. Competition Deliverables and Next Milestones

Current deliverables include the Gradio platform, BFN and baseline runners, frozen ECLS benchmark, Aβ candidate funnel, 81 AF2 structures, 12 scFv-ready protein designs, the eight-candidate Tau computational shortlist, reproducibility manifests, and a fail-closed experimental feedback schema.

The evidence-tier audit, 20-component generator benchmark, and correctly masked Tau second-target candidate protocol are complete. If selected for the final, we will prepare a 5-8 minute live software demonstration using the frozen 4HIX workflow and cached results.

## References

1. Dauparas et al. Robust deep learning-based protein sequence design using ProteinMPNN. Science, 2022.
2. Jumper et al. Highly accurate protein structure prediction with AlphaFold. Nature, 2021.
3. Evans et al. Protein complex prediction with AlphaFold-Multimer. bioRxiv, 2021.
4. Dunbar et al. SAbDab: the structural antibody database. Nucleic Acids Research, 2014.
5. Eastman et al. OpenMM 7. PLoS Computational Biology, 2017.
