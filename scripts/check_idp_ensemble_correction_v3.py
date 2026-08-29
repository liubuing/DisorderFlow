#!/usr/bin/env python
"""Fail-closed readiness check for the four ensemble-pilot corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_status(path, expected_hash=None):
    path = Path(path)
    if not path.is_file():
        return {"present": False, "hash_matches": False, "path": str(path)}
    observed = sha256(path)
    return {
        "present": True,
        "hash_matches": expected_hash is None or observed == expected_hash,
        "path": str(path),
        "sha256": observed,
    }


def latest_versioned(prefix):
    directory = ROOT / "reviewer_outputs/idp_ensemble_correction_v3"
    versions = sorted(directory.glob(f"{prefix}_v*.json"))
    return artifact_status(versions[-1]) if versions else artifact_status(
        directory / f"{prefix}_v1.json"
    )


def check(config_path, output_path):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite correction status: {output_path}")
    adapter = artifact_status(
        ROOT / "reviewer_outputs/idp_ensemble_correction_v3/model_adapter_audit_v6.json"
    )
    adapter_payload = (
        json.loads(Path(adapter["path"]).read_text(encoding="ascii"))
        if adapter["present"] else {}
    )
    current_readiness = artifact_status(
        ROOT / "publication/successor_v3_future_readiness_after_exposure_2026_08_15.json"
    )
    panel = artifact_status(
        ROOT / "reviewer_outputs/idp_ensemble_correction_v3/panel_plan.json"
    )
    admission = artifact_status(
        ROOT / "reviewer_outputs/idp_ensemble_correction_v3/untouched_cohort_admission_v4.json"
    )
    panel_payload = (
        json.loads(Path(panel["path"]).read_text(encoding="ascii"))
        if panel["present"] else {}
    )
    admission_payload = (
        json.loads(Path(admission["path"]).read_text(encoding="ascii"))
        if admission["present"] else {}
    )
    generation = latest_versioned("generation_readiness")
    generation_payload = (
        json.loads(Path(generation["path"]).read_text(encoding="ascii"))
        if generation["present"] else {}
    )
    candidates = artifact_status(
        ROOT / "reviewer_outputs/idp_ensemble_correction_v3/candidates.json"
    )
    candidates_payload = (
        json.loads(Path(candidates["path"]).read_text(encoding="ascii"))
        if candidates["present"] else {}
    )
    pose = latest_versioned("pose_status")
    pose_payload = (
        json.loads(Path(pose["path"]).read_text(encoding="ascii"))
        if pose["present"] else {}
    )
    measurements = artifact_status(
        ROOT / "reviewer_outputs/idp_ensemble_correction_v3/measurement_status.json"
    )
    measurements_payload = (
        json.loads(Path(measurements["path"]).read_text(encoding="ascii"))
        if measurements["present"] else {}
    )
    readiness_payload = (
        json.loads(Path(current_readiness["path"]).read_text(encoding="ascii"))
        if current_readiness["present"] else {}
    )
    untouched_ready = (
        readiness_payload.get("ready") is True
        and readiness_payload.get("observed_exact_new_entries", 0) >= 12
        and admission_payload.get("status") == "untouched_cohort_admitted"
        and admission_payload.get("observed", {}).get(
            "independent_lineage_components", 0
        ) >= 12
        and admission_payload.get("observed", {}).get("targets", 0) >= 3
    )
    mutation = config["mutation_space"]
    panel_mutation = panel_payload.get("mutation_space", {})
    panel_ready = (
        panel_payload.get("status") == "v3_panel_plan_frozen_before_candidate_access"
        and panel_mutation.get("substitution_buckets") == mutation["substitution_buckets"]
        and panel_mutation.get("maximum_h3_substitutions") == mutation["maximum_h3_substitutions"]
    )
    passed = {
        "autoregressive_generator": adapter_payload.get("status") == "adapter_eligible",
        "untouched_cohort": untouched_ready,
        "minimum_independent_components": (
            panel_payload.get("status")
            == "v3_panel_plan_frozen_before_candidate_access"
            and len(panel_payload.get("components", [])) >= 12
        ),
        "broader_mutation_space": (
            panel_ready
            and pose_payload.get("status") == "pose_panel_ready"
            and generation_payload.get("status") == "generation_ready"
            and candidates_payload.get("status") == "complete"
        ),
        "experimental_measurements": (
            measurements_payload.get("status") == "measurements_complete"
        ),
    }
    eligible = all(passed.values())
    payload = {
        "schema_version": 1,
        "status": "correction_ready_for_execution" if eligible else "correction_blocked",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "checks": passed,
        "passed_count": sum(passed.values()),
        "required_count": len(passed),
        "decision": (
            "freeze_execution_manifest_before_model_or_coordinate_access"
            if eligible else
            "do_not_score_or_claim_ensemble_improvement; acquire_missing_inputs"
        ),
        "claim_boundary": config["claim_boundary"],
        "artifact_audits": {
            "adapter": adapter,
            "adapter_payload_status": adapter_payload.get("status"),
            "current_readiness": current_readiness,
            "admission": admission,
            "admission_status": admission_payload.get("status"),
            "generation": generation,
            "generation_status": generation_payload.get("status"),
            "candidates": candidates,
            "candidates_status": candidates_payload.get("status"),
            "pose_panel": pose,
            "pose_panel_status": pose_payload.get("status"),
            "pose_ready_components": pose_payload.get("ready_components"),
            "measurements": measurements,
            "measurements_status": measurements_payload.get("status"),
            "panel": panel,
            "panel_status": panel_payload.get("status"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_correction_v3.yml"
    )
    parser.add_argument(
        "--output", default="reviewer_outputs/idp_ensemble_correction_v3/status.json"
    )
    args = parser.parse_args()
    print(json.dumps(check(ROOT / args.config, ROOT / args.output), indent=2))


if __name__ == "__main__":
    main()
