# Candidate-Interface PAE Deployment — Reproducibility

This document describes how to reproduce the **PAE single-axis candidate-interface
confidence deployment**: six independent antibody-peptide scaffolds are discovered
and isolated, candidate interfaces are generated and scored with AlphaFold2, and a
deployment-grade interface-PAE ranking signal is produced via variance-matching
calibration. The frozen contract is
`publication/candidate_interface_pae_deployment_v1.json`.

## Scope

- **Deployment signal**: normalized candidate-to-antigen interface PAE (only).
- **Abstained**: pLDDT (non-transferable across scaffolds) and ipTM (insufficient
  reliable pair evidence).
- **Sealed test**: never evaluated.

## Pipeline overview

1. **Scaffold discovery** (three preregistered RCSB rounds → 6 independent scaffolds)
2. **Isolation audit** (five-axis homology vs all reference pools)
3. **Candidate generation** (BFN + ProteinMPNN CDR-H3 redesign)
4. **AF2 scoring** (multimer v3, models 1/2 × 3 seeds)
5. **Pair-evidence evaluation** (reliable target deltas vs hierarchical SEM)
6. **PAE calibration** (variance-matching affine)
7. **Deployment freeze**

## Scripts

| Step | Script |
|---|---|
| RCSB snapshot | `scripts/build/acquire_rcsb_candidate_interface_extension.py` |
| Discovery | `scripts/build/discover_rcsb_candidate_interface_extension.py` |
| Structure materialization | `scripts/build/materialize_rcsb_candidate_interface_extension.py` |
| Viral-pool filter | `scripts/build/filter_viral_candidates.py` |
| Manifest merge | `scripts/build/merge_structural_manifests.py` |
| Isolation audit | `scripts/audit_successor_v3_isolation.py` |
| Reference union | `scripts/build/build_candidate_interface_extension_reference_union.py` |
| Pipeline config builder | `scripts/build/build_calibration_extension_pipeline.py` |
| Extension manifest freeze | `scripts/build/freeze_candidate_interface_extension_manifest.py` |
| Pair-evidence evaluation | `scripts/evaluate_candidate_interface_calibration_extension.py` |
| Transfer evaluation | `scripts/evaluate_candidate_interface_calibration_extension_transfer.py` |
| Checkpoint reselection | `scripts/reselect_candidate_interface_checkpoints.py` |
| PAE calibration | `scripts/calibrate_candidate_interface_pae.py` |
| Deployment freeze | `scripts/freeze_candidate_interface_pae_deployment.py` |

Candidate generation, AF2, and LMDB build reuse the existing pipeline:
`generate_multiscaffold_confirmatory_v2.py`, `run_multiscaffold_v2_af2.py`,
`build_candidate_interface_multiscaffold_v1.py` (with `--uncertainty-version v2`).

## Prerequisites

- Python environment with `torch`, `biopython`, `anarcii`, `numpy`, `yaml`, `lmdb`
- BFN candidate-interface confidence checkpoint (v2 lineage)
- AlphaFold2 multimer v3 parameters (WSL; `~/.cache/colabfold/params/`)
- MMseqs2 (for the five-axis isolation audit)
- WSL2 + GPU (AF2 inference is JAX/WSL-only)

## Frozen artifacts

- `publication/candidate_interface_pae_deployment_v1.json` — deployment contract
  (per-seed calibration maps, six gates, abstentions)
- `configs/candidate_interface_calibration_extension_v1.json` — discovery/isolation protocol
- `configs/candidate_interface_extension_rcsb_query_v1.json` / `_v2.json` — frozen RCSB queries
- `configs/benchmarks/*calibration_ext*` — generation/selection/AF2 pipeline configs
- `publication/multiscaffold_v2_calibration_ext_checkpoint_lineage.json` — checkpoint lineage

## Key results (reproducible)

| Signal | Result |
|---|---|
| Interface PAE transfer | median scaffold Spearman 0.725–0.866 (3 seeds) |
| Post-calibration MAE | 0.022–0.029 (≤0.10) |
| Post-calibration variance ratio | 1.0 |
| Pair accuracy | 0.843–0.953 (≥0.65) |
| Bootstrap 95% lower | 0.683–0.873 (>0.5) |

## Claim boundary

This deployment provides a **relative ranking** signal for candidate antibody-peptide
interfaces. It does not claim binding, affinity, or efficacy; absolute binding
validation requires wet-lab measurement. The sealed test split is not evaluated.
