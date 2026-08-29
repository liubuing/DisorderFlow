# ECLS/GCLC Reproducibility Guide

## Scope

The lightweight reviewer package reproduces tests, summary statistics, tables,
and the generator-calibration figure from saved candidate-level results. It
does not bundle LMDB datasets, third-party model weights, checkpoints, or model
work directories.

The post-ProteinMPNN temporal final is immutable. Authors must not rerun it.
Its saved result is verified against the SHA256 recorded in
`results/publication/h3_ecls_temporal_final_v1/final_decision.json`.

## Lightweight Reproduction

Create a Python environment and install the analysis dependencies:

```powershell
py -3.11 -m venv .venv-review
.\.venv-review\Scripts\python -m pip install --upgrade pip
.\.venv-review\Scripts\python -m pip install numpy pyyaml matplotlib pytest openmm pdbfixer
```

Run the focused tests:

```powershell
.\.venv-review\Scripts\python -m pytest `
  tests/test_h3_epitope_delta.py `
  tests/test_h3_candidate_reranking.py `
  tests/test_h3_generator_calibration.py `
  tests/test_h3_ecls_statistics.py `
  tests/test_h3_publication_package.py `
  tests/test_peptide_conformer_ensemble.py `
  tests/test_h3_t1_ensemble.py `
  tests/test_h3_t1_ensemble_analysis.py `
  tests/test_peptide_t2_recovery.py `
  tests/test_h3_t2_recovery.py `
  tests/test_h3_t2_recovery_analysis.py `
  tests/test_h3_interface_contacts.py `
  tests/test_peptide_h3_publication_split.py `
  tests/test_peptide_h3_temporal_split.py -q
```

Regenerate the calibration statistics, CSV tables, and figure from frozen
candidate results:

```powershell
.\.venv-review\Scripts\python scripts/analyze_h3_generator_calibration.py `
  --config configs/benchmarks/peptide_h3_generator_calibration_v1.yml `
  --out-dir reviewer_outputs/h3_generator_calibration_v1
```

Regenerate the post hoc ECLS sign-flip sensitivity analysis from saved results
without model inference:

```powershell
.\.venv-review\Scripts\python scripts/analyze_h3_ecls_statistics.py `
  --out-dir reviewer_outputs/h3_ecls_statistical_summary_v1
```

Regenerate the T1 ensemble analysis and figures from saved conformer-level
scores without rerunning OpenMM or ProteinMPNN:

```powershell
.\.venv-review\Scripts\python scripts/analyze_h3_t1_ensemble.py `
  --results results/publication/h3_t1_ensemble_dev_v3/results.json `
  --out-dir reviewer_outputs/h3_t1_ensemble_analysis_v1
```

Full T1 development regeneration requires OpenMM 8.5.2, PDBFixer 1.12.0, the
development LMDB, and ProteinMPNN weights:

```powershell
python scripts/benchmark_h3_t1_ensemble.py `
  --config configs/benchmarks/peptide_h3_t1_ensemble_dev_v3.yml `
  --out-dir reviewer_outputs/h3_t1_ensemble_dev_v3
```

Regenerate the T2 negative-result analysis from saved results:

```powershell
python scripts/analyze_h3_t2_recovery.py `
  --results results/publication/h3_t2_recovery_dev_v1/results.json `
  --out-dir reviewer_outputs/h3_t2_recovery_analysis_v1
```

Full T2 regeneration uses the frozen development config and does not access the
temporal final:

```powershell
python scripts/benchmark_h3_t2_recovery.py `
  --config configs/benchmarks/peptide_h3_t2_recovery_dev_v1.yml `
  --out-dir reviewer_outputs/h3_t2_recovery_dev_v1
```

Build and verify the lightweight archive:

```powershell
.\.venv-review\Scripts\python scripts/build_h3_publication_package.py `
  --out-dir reviewer_outputs/h3_submission_package_v1
```

## Full Model Reproduction

Full rescoring additionally requires the SAbDab2-derived LMDB files and
ProteinMPNN `v_48_020` weights. Candidate regeneration requires the BFN
checkpoint, fair-esm ESM-IF1 weights, `torch-geometric`, and `biotite`.
External assets must be restored at the paths frozen in each YAML config and
verified against the separately distributed checksums.

The author-permitted adaptation command is:

```powershell
python scripts/benchmark_h3_epitope_delta.py `
  --config configs/benchmarks/peptide_h3_ecls_adaptation_v1.yml `
  --out-dir reviewer_outputs/h3_ecls_adaptation_v1
```

An independent reviewer may evaluate the temporal config into a new directory
after obtaining the external temporal LMDB. This does not authorize an author
rerun or replacement of the immutable reported temporal result.

## Statistical Unit

All confidence intervals and permutation tests use the official antigen
cluster as the inference unit. Generation seeds and candidates are averaged
within cluster and are not treated as independent biological replicates.

Normalized native rank is

`NNR = 1 - (native_rank - 1) / (pool_size - 1)`.

NNR is one for the best rank and has random expectation 0.5. Ties use average
rank. Bootstrap intervals use 10,000 cluster resamples. Reported P values use
two-sided exact paired sign-flip tests; generator-level secondary comparisons
use Benjamini-Hochberg correction.
