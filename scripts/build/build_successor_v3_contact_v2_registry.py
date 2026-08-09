#!/usr/bin/env python
"""Seal the successor-v3 contact-v2 artifact registry."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = {
    "development_protocol": "configs/benchmarks/successor_v3_contact_v2_development.yml",
    "future_confirmation_policy": "configs/benchmarks/successor_v3_contact_v2_future_confirmation.yml",
    "training_config": "configs/train/bfn_successor_v3_contact_v2_exposed.yml",
    "dataset_manifest": "data/successor_v3_contact_dev_v1/manifest.json",
    "dataset_train_lmdb": "data/successor_v3_contact_dev_v1/train.lmdb",
    "dataset_eval_lmdb": "data/successor_v3_contact_dev_v1/eval.lmdb",
    "baselines": "results/successor_v3_contact_v2/baselines.json",
    "development_evaluation": "results/successor_v3_contact_v2/development_evaluation.json",
    "candidate_checkpoint": "logs/bfn_successor_v3_contact_v2_exposed_2026_08_09__09_06_40_successor_v3_contact_v2_cb/checkpoints/best.pt",
    "experiment_panel": "experiments/successor_v3_contact_v2/panel.json",
    "experiment_constructs": "experiments/successor_v3_contact_v2/constructs.csv",
    "experiment_protocol": "experiments/successor_v3_contact_v2/PROTOCOL.md",
    "experiment_template": "experiments/successor_v3_contact_v2/measurements_template.csv",
    "pair_head": "disorderflow/modules/bfn/receiver.py",
    "contact_loss": "disorderflow/modules/bfn/core.py",
    "hard_negative_model": "disorderflow/models/bfn_model.py",
    "dataset_builder": "scripts/build/build_successor_v3_contact_dev_dataset.py",
    "baseline_runner": "scripts/benchmark_successor_v3_contact_baselines.py",
    "evaluation_runner": "scripts/evaluate_successor_v3_contact_v2.py",
    "experiment_runner": "scripts/prepare_successor_v3_contact_v2_experiments.py",
    "confirmatory_evaluation_runner": "scripts/evaluate_successor_v3_contact_v2_confirmatory.py",
    "lineage_validator": "scripts/validate_release_lineage.py",
    "ecls_scope_freeze": "publication/ECLS_SCOPE_FREEZE.yml",
    "artifact_bundle_manifest": "release/ecls_v1/artifact_bundle_manifest.json",
    "pipeline": "scripts/pipelines/run_successor_v3_contact_v2_pipeline.py",
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("publication/successor_v3_contact_v2_registry.json"))
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    missing = [value for value in ARTIFACTS.values() if not (ROOT / value).is_file()]
    if missing:
        raise FileNotFoundError(f"Registry inputs missing: {missing}")
    entries = {
        name: {
            "path": relative, "sha256": digest(ROOT / relative),
            "bytes": (ROOT / relative).stat().st_size,
        } for name, relative in ARTIFACTS.items()
    }
    evaluation = json.loads((ROOT / ARTIFACTS["development_evaluation"]).read_text())
    panel = json.loads((ROOT / ARTIFACTS["experiment_panel"]).read_text())
    payload = {
        "schema_version": 1,
        "status": "contact_v2_candidate_frozen_waiting_for_12_plus_independent_components",
        "classification": "mixed registry; each claim remains bounded by its source artifact",
        "artifacts": entries,
        "development_summary": {
            "held_out_records": evaluation["held_out_records"],
            "residue_contact_auroc": evaluation["residue_contact"]["auroc"],
            "pair_contact_auroc": evaluation["pair_contact"]["auroc"],
            "pair_contact_average_precision": evaluation["pair_contact"]["average_precision"],
            "residue_component_macro_auroc": evaluation["component_level"]
            ["residue_contact"]["macro_auroc"],
            "residue_component_macro_auroc_ci95": evaluation["component_level"]
            ["residue_contact"]["macro_auroc_component_bootstrap_ci95"],
            "eligible_for_future_confirmation": evaluation["eligible_for_future_confirmation"],
        },
        "wet_experiment_summary": {
            "status": panel["status"], "targets": len(panel["panel"]),
            "constructs": sum(len(row["construct_ids"]) for row in panel["panel"]),
            "independent_targets": panel["power_analysis"]["independent_targets"],
            "planned_target_level_power": panel["power_analysis"]
            ["target_level_paired_t_test_power"],
            "measurements_present": False,
        },
        "future_readiness": {
            "eligible_exact_new_structures": 0,
            "independent_homology_components": 0,
            "required_independent_homology_components": 12,
            "ready": False,
        },
        "claim_boundary": (
            "The candidate passed exposed development gates only. No confirmatory "
            "or wet-experiment result exists."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": payload["status"], "artifacts": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
