#!/usr/bin/env python
"""Audit whether expanded IDP artifacts can support StateContrast-v2 training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disorderflow.statecontrast_v2_contract import validate_statecontrast_v2_records



def audit(candidate_path, pose_path, output_path, explicit_states_path=None,
          training_manifest_path=None):
    candidates = json.loads(candidate_path.read_text(encoding="ascii"))
    poses = json.loads(pose_path.read_text(encoding="ascii"))
    pose_by_component = {
        row["component_id"]: row["poses"] for row in poses["components"]
    }
    records = []
    for component_id, component in candidates["components"].items():
        for pose in pose_by_component.get(component_id, []):
            # Existing artifacts contain bound target poses only. Materializing
            # this fact through the normal validator prevents accidental claims
            # that sampled target poses are apo or off-target supervision.
            records.append({
                "group_id": component_id,
                "state": {
                    "type": "target",
                    "source": pose["source"],
                    "prior_weight": 1.0,
                },
                "targets": {
                    "state_training_weight": 1.0,
                    "independent_teacher_score": None,
                },
            })
    explicit = None
    if explicit_states_path is not None and explicit_states_path.is_file():
        explicit = json.loads(explicit_states_path.read_text(encoding="ascii"))
        explicit_records = [
            {
                "group_id": (
                    f"{component['antibody_lineage_cluster']}|"
                    f"{component['candidate_sequence']}"),
                "state": {
                    "type": state["state_type"],
                    "source": state["source"],
                    "prior_weight": state["prior_weight"],
                },
                "targets": {
                    "state_training_weight": state.get("state_training_weight", 1.0),
                    "independent_teacher_score": state.get(
                        "independent_teacher_score"),
                    "independent_teacher_required": state.get(
                        "independent_teacher_required", True),
                },
            }
            for component in explicit["components"] for state in component["states"]
        ]
        validation = validate_statecontrast_v2_records(
            explicit_records, minimum_teacher_coverage=0.80)
        available_states = set(explicit.get("available_state_types", []))
    else:
        validation = validate_statecontrast_v2_records(
            records, minimum_teacher_coverage=0.80)
        available_states = {"target"} if records else set()
    training_manifest = (
        json.loads(training_manifest_path.read_text(encoding="ascii"))
        if training_manifest_path is not None and training_manifest_path.is_file()
        else {}
    )
    manifest_ready = (
        training_manifest.get("status") == "statecontrast_v2_manifest_ready"
        and training_manifest.get("validation", {}).get("valid") is True
        and bool(training_manifest.get("records"))
    )
    off_target_ready = "off_target" in available_states
    training_ready = validation["valid"] and off_target_ready and manifest_ready
    payload = {
        "schema_version": 1,
        "status": (
            "statecontrast_v2_expanded_training_ready"
            if training_ready else
            "statecontrast_v2_expanded_training_blocked"
        ),
        "classification": "exposed_expanded_idp_training_readiness",
        "component_count": candidates["component_count"],
        "candidate_count": candidates["candidate_count"],
        "existing_pose_record_count": len(records),
        "checks": {
            "explicit_target_states": "target" in available_states,
            "explicit_apo_states": "apo" in available_states,
            "explicit_off_target_states": off_target_ready,
            "independent_teacher_coverage_80_percent": (
                validation["teacher_coverage"] >= 0.80),
            "complete_state_groups": validation["valid"],
            "direct_training_manifest_ready": manifest_ready,
        },
        "validation": validation,
        "required_next_artifacts": (
            [] if training_ready else
            ["complete_explicit_states_and_direct_training_manifest"]
        ),
        "decision": (
            "run_development_training_only" if training_ready else
            "build_explicit_state_artifacts_before_training; "
            "do_not_relabel_sampled_bound_poses_as_negative_states"
        ),
        "claim_boundary": (
            "Readiness audit only; no model training or performance claim"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("reviewer_outputs/idp_ensemble_expanded_development_v2/candidates.json"))
    parser.add_argument(
        "--poses", type=Path,
        default=Path("reviewer_outputs/idp_ensemble_expanded_development_v2/merged_pose_manifest_v2.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("reviewer_outputs/statecontrast_v2_expanded_readiness_v1/status.json"))
    parser.add_argument(
        "--explicit-states", type=Path,
        default=Path("reviewer_outputs/statecontrast_v2_expanded_states_v1/explicit_states.json"))
    parser.add_argument(
        "--training-manifest", type=Path,
        default=Path(
            "reviewer_outputs/statecontrast_v2_expanded_states_v1/training_manifest.json"))
    args = parser.parse_args()
    result = audit(
        ROOT / args.candidates, ROOT / args.poses, ROOT / args.output,
        ROOT / args.explicit_states, ROOT / args.training_manifest)
    print(json.dumps({
        "status": result["status"],
        "checks": result["checks"],
        "required_next_artifacts": result["required_next_artifacts"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
