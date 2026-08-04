#!/usr/bin/env python
"""Build a lightweight, checksummed reviewer package for the ECLS/GCLC study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PACKAGE_FILES = [
    "PUBLICATION_PROTOCOL.md",
    "publication/REPRODUCIBILITY.md",
    "publication/MANUSCRIPT_DRAFT.md",
    "configs/benchmarks/peptide_h3_ecls_adaptation_v1.yml",
    "configs/benchmarks/peptide_h3_ecls_temporal_final_v1.yml",
    "configs/benchmarks/peptide_h3_candidate_reranking_dev_v1.yml",
    "configs/benchmarks/peptide_h3_candidate_reranking_adaptation_mpnn_v1.yml",
    "configs/benchmarks/peptide_h3_generator_calibration_v1.yml",
    "configs/benchmarks/peptide_h3_t1_ensemble_dev_v3.yml",
    "configs/benchmarks/peptide_h3_t2_recovery_dev_v1.yml",
    "configs/benchmarks/peptide_h3_publication_split_v4.yml",
    "configs/benchmarks/peptide_h3_temporal_split_v3.yml",
    "modules/esmif_compat.py",
    "modules/h3_interface_contacts.py",
    "modules/peptide_conformer_ensemble.py",
    "modules/peptide_t2_recovery.py",
    "scripts/benchmark_h3_epitope_delta.py",
    "scripts/benchmark_h3_candidate_reranking.py",
    "scripts/analyze_h3_generator_calibration.py",
    "scripts/analyze_h3_ecls_statistics.py",
    "scripts/generate_h3_peptide_ensemble.py",
    "scripts/benchmark_h3_t1_ensemble.py",
    "scripts/analyze_h3_t1_ensemble.py",
    "scripts/generate_h3_peptide_t2.py",
    "scripts/benchmark_h3_t2_recovery.py",
    "scripts/analyze_h3_t2_recovery.py",
    "scripts/build_h3_publication_package.py",
    "scripts/build/build_peptide_h3_publication_split.py",
    "scripts/build/build_peptide_h3_temporal_split.py",
    "tests/test_h3_epitope_delta.py",
    "tests/test_h3_candidate_reranking.py",
    "tests/test_h3_generator_calibration.py",
    "tests/test_h3_ecls_statistics.py",
    "tests/test_peptide_conformer_ensemble.py",
    "tests/test_h3_t1_ensemble.py",
    "tests/test_h3_t1_ensemble_analysis.py",
    "tests/test_peptide_t2_recovery.py",
    "tests/test_h3_t2_recovery.py",
    "tests/test_h3_t2_recovery_analysis.py",
    "tests/test_h3_publication_package.py",
    "tests/test_h3_interface_contacts.py",
    "tests/test_peptide_h3_publication_split.py",
    "tests/test_peptide_h3_temporal_split.py",
    "results/publication/h3_ecls_adaptation_v1/results.json",
    "results/publication/h3_ecls_temporal_final_v1/results.json",
    "results/publication/h3_ecls_temporal_final_v1/final_decision.json",
    "results/publication/h3_candidate_reranking_dev_v1/results.json",
    "results/publication/h3_candidate_reranking_adaptation_mpnn_v1/results.json",
    "results/publication/h3_generator_calibration_v1/analysis.json",
    "results/publication/h3_generator_calibration_v1/per_unit.csv",
    "results/publication/h3_generator_calibration_v1/main_table.csv",
    "results/publication/h3_generator_calibration_v1/proteinmpnn_calibration_curve.csv",
    "results/publication/h3_generator_calibration_v1/generator_calibration.pdf",
    "results/publication/h3_generator_calibration_v1/generator_calibration.png",
    "results/publication/h3_generator_calibration_v1/milestone_decision.json",
    "results/publication/h3_ecls_statistical_summary_v1/analysis.json",
    "results/publication/h3_t1_ensemble_dev_v3/results.json",
    "results/publication/h3_t1_ensemble_analysis_v1/analysis.json",
    "results/publication/h3_t1_ensemble_analysis_v1/main_table.csv",
    "results/publication/h3_t1_ensemble_analysis_v1/t1_ensemble_summary.pdf",
    "results/publication/h3_t1_ensemble_analysis_v1/t1_ensemble_summary.png",
    "results/publication/h3_t1_ensemble_analysis_v1/milestone_decision.json",
    "results/publication/h3_t2_recovery_dev_v1/results.json",
    "results/publication/h3_t2_recovery_analysis_v1/analysis.json",
    "results/publication/h3_t2_recovery_analysis_v1/main_table.csv",
    "results/publication/h3_t2_recovery_analysis_v1/t2_recovery_summary.pdf",
    "results/publication/h3_t2_recovery_analysis_v1/t2_recovery_summary.png",
    "results/publication/h3_t2_recovery_analysis_v1/milestone_decision.json",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_manifest(paths):
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def environment_report():
    packages = {}
    for name in (
        "torch", "numpy", "lmdb", "PyYAML", "matplotlib", "biopython",
        "fair-esm", "torch-geometric", "biotite", "openmm", "pdbfixer", "pytest",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_commit = None
    git_dirty = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "captured_at_package_build_not_original_model_run": True,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit,
        "git_worktree_dirty": git_dirty,
    }


def verify_temporal_final():
    decision_path = ROOT / "results/publication/h3_ecls_temporal_final_v1/final_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    source = ROOT / decision["source_result"]
    observed = sha256(source)
    if observed != decision["source_result_sha256"]:
        raise RuntimeError("Immutable temporal-final result hash does not match its decision")
    if decision.get("rerun_permitted") is not False or not decision.get("terminal"):
        raise RuntimeError("Temporal-final decision is not terminal")
    return {
        "decision": decision["decision"],
        "source_result_sha256": observed,
        "rerun_permitted": False,
    }


def main_results():
    adaptation = json.loads((
        ROOT / "results/publication/h3_ecls_adaptation_v1/results.json"
    ).read_text(encoding="utf-8"))["aggregate"]
    temporal = json.loads((
        ROOT / "results/publication/h3_ecls_temporal_final_v1/final_decision.json"
    ).read_text(encoding="utf-8"))["primary_endpoint"]
    reranking = json.loads((
        ROOT / "results/publication/h3_candidate_reranking_dev_v1/results.json"
    ).read_text(encoding="utf-8"))["aggregate"]
    calibrated = json.loads((
        ROOT / "results/publication/h3_generator_calibration_v1/analysis.json"
    ).read_text(encoding="utf-8"))["development"]
    ensemble = json.loads((
        ROOT / "results/publication/h3_t1_ensemble_analysis_v1/analysis.json"
    ).read_text(encoding="utf-8"))["summary"]
    t2 = json.loads((
        ROOT / "results/publication/h3_t2_recovery_analysis_v1/analysis.json"
    ).read_text(encoding="utf-8"))["summary"]
    return [
        {
            "analysis": "ECLS adaptation native-vs-shuffle",
            "n_inference_units": adaptation["n_inference_units"],
            "effect": adaptation["mean_ecls_advantage"],
            "ci95_low": adaptation["ecls_advantage_mean_ci95"][0],
            "ci95_high": adaptation["ecls_advantage_mean_ci95"][1],
            "interpretation": "positive exposed adaptation confirmation",
        },
        {
            "analysis": "ECLS temporal-final native-vs-shuffle",
            "n_inference_units": temporal["n_inference_units"] if "n_inference_units" in temporal else 15,
            "effect": temporal["mean"],
            "ci95_low": temporal["bootstrap_mean_ci95"][0],
            "ci95_high": temporal["bootstrap_mean_ci95"][1],
            "interpretation": "positive one-time temporal final",
        },
        {
            "analysis": "Universal ECLS gain over complex NLL",
            "n_inference_units": reranking["n_inference_units"],
            "effect": reranking["overall_mean_gain"],
            "ci95_low": reranking["overall_gain_ci95"][0],
            "ci95_high": reranking["overall_gain_ci95"][1],
            "interpretation": "development gate rejected",
        },
        {
            "analysis": "Generator-aware NNR over random",
            "n_inference_units": calibrated["n_inference_units"],
            "effect": calibrated["mean_generator_aware_nnr"] - 0.5,
            "ci95_low": calibrated["generator_aware_over_random_ci95"][0],
            "ci95_high": calibrated["generator_aware_over_random_ci95"][1],
            "interpretation": "positive exploratory development result",
        },
        {
            "analysis": "T1 ensemble native-vs-shuffle ECLS",
            "n_inference_units": ensemble["n_valid_ensembles"],
            "effect": ensemble["mean_ensemble_ecls_advantage"],
            "ci95_low": ensemble["ensemble_ecls_advantage_ci95"][0],
            "ci95_high": ensemble["ensemble_ecls_advantage_ci95"][1],
            "interpretation": "positive mean but frozen development gate rejected",
        },
        {
            "analysis": "T1 ensemble gain over deposited single pose",
            "n_inference_units": ensemble["n_valid_ensembles"],
            "effect": ensemble["mean_ensemble_minus_single_advantage"],
            "ci95_low": ensemble["ensemble_minus_single_advantage_ci95"][0],
            "ci95_high": ensemble["ensemble_minus_single_advantage_ci95"][1],
            "interpretation": "ensemble underperformed deposited pose",
        },
        {
            "analysis": "T2 held-out contact recovery",
            "n_inference_units": t2["n_valid_clusters"],
            "effect": t2["mean_held_out_contact_recovery"],
            "ci95_low": t2["held_out_contact_recovery_ci95"][0],
            "ci95_high": t2["held_out_contact_recovery_ci95"][1],
            "interpretation": "moderate-perturbation recovery gate rejected",
        },
        {
            "analysis": "T2 RMSD recovery angstrom",
            "n_inference_units": t2["n_valid_clusters"],
            "effect": t2["mean_rmsd_recovery_angstrom"],
            "ci95_low": t2["rmsd_recovery_ci95"][0],
            "ci95_high": t2["rmsd_recovery_ci95"][1],
            "interpretation": "no consistent recovery toward deposited pose",
        },
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/publication/h3_submission_package_v1"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / relative for relative in PACKAGE_FILES]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Reviewer package inputs are missing: {missing}")
    temporal_verification = verify_temporal_final()
    manifest = {
        "schema_version": 1,
        "package": "ECLS_GCLC_reviewer_package_v1",
        "temporal_final": temporal_verification,
        "files": package_manifest(paths),
        "excluded_large_assets": [
            "LMDB datasets", "ProteinMPNN weights", "BFN checkpoint",
            "ESM-IF weights", "model work directories",
        ],
    }
    manifest_path = out_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    environment_path = out_dir / "software_environment.json"
    environment_path.write_text(
        json.dumps(environment_report(), indent=2) + "\n", encoding="ascii")
    result_rows = main_results()
    results_path = out_dir / "main_results.csv"
    with results_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    zip_path = out_dir / "ECLS_GCLC_reviewer_package_v1.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(manifest_path, "reviewer_package/evidence_manifest.json")
        archive.write(environment_path, "reviewer_package/software_environment.json")
        archive.write(results_path, "reviewer_package/main_results.csv")
    print(json.dumps({
        "n_files": len(paths),
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "temporal_final": temporal_verification,
    }, indent=2))


if __name__ == "__main__":
    main()
