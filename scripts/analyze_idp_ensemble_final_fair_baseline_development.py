#!/usr/bin/env python
"""Audit the frozen all-single-pose median baseline on development artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recover_omitted_pose_score(full_mean, retained_mean, pose_count):
    if pose_count < 2:
        raise ValueError("At least two poses are required")
    return pose_count * full_mean - (pose_count - 1) * retained_mean


def analyze(source_path, loo_path, abstention_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite fair baseline audit: {output}")
    source = json.loads(source_path.read_text(encoding="ascii"))
    loo = json.loads(loo_path.read_text(encoding="ascii"))
    abstention = json.loads(abstention_path.read_text(encoding="ascii"))
    full = {
        (row["component_id"], row["candidate_id"]): row[
            "mean_h3_log_probability"
        ]
        for row in source["records"]
        if row["source_panel"] == "mixed"
    }
    admitted = {
        (row["component_id"], row["substitution_bucket"]): row[
            "admitted_by_final_abstention"
        ]
        for row in abstention["records"]
    }
    records = []
    for component in loo["components"]:
        component_id = component["component_id"]
        pose_count = component["pose_count"]
        candidate_ids = list(component["omissions"][0]["candidate_scores"])
        individual = {candidate_id: [] for candidate_id in candidate_ids}
        for omission in component["omissions"]:
            for candidate_id, retained_mean in omission["candidate_scores"].items():
                individual[candidate_id].append({
                    "pose_id": omission["omitted_pose_id"],
                    "source": omission["omitted_source"],
                    "score": recover_omitted_pose_score(
                        full[(component_id, candidate_id)], retained_mean, pose_count
                    ),
                })
        for bucket in (2, 4, 6, 8):
            ensemble_id = f"ensemble_{bucket}"
            single_id = f"single_state_{bucket}"
            ensemble_mixed = full[(component_id, ensemble_id)]
            single_pose_scores = [row["score"] for row in individual[single_id]]
            single_median = statistics.median(single_pose_scores)
            fair_difference = ensemble_mixed - single_median
            records.append({
                "component_id": component_id,
                "substitution_bucket": bucket,
                "ensemble_fixed_candidate_mixed_mean_score": ensemble_mixed,
                "single_state_fixed_candidate_pose_scores": individual[single_id],
                "all_single_pose_median_score": single_median,
                "ensemble_minus_all_single_pose_median": fair_difference,
                "ensemble_favored_by_model": fair_difference > 0,
                "admitted_by_final_abstention": admitted[(component_id, bucket)],
            })
    admitted_rows = [row for row in records if row["admitted_by_final_abstention"]]
    payload = {
        "schema_version": 1,
        "status": "fair_single_pose_baseline_development_audit_complete",
        "classification": "exposed_non_idp_development_rule_audit",
        "source_sensitivity": str(source_path),
        "leave_one_pose_out": str(loo_path),
        "abstention": str(abstention_path),
        "matched_pair_count": len(records),
        "admitted_pair_count": len(admitted_rows),
        "ensemble_favored_all_pairs": sum(
            row["ensemble_favored_by_model"] for row in records
        ),
        "ensemble_favored_admitted_pairs": sum(
            row["ensemble_favored_by_model"] for row in admitted_rows
        ),
        "median_fair_difference_all_pairs": statistics.median(
            row["ensemble_minus_all_single_pose_median"] for row in records
        ),
        "median_fair_difference_admitted_pairs": statistics.median(
            row["ensemble_minus_all_single_pose_median"] for row in admitted_rows
        ),
        "records": records,
        "decision": "freeze_all_single_pose_median_without_further_baseline_search",
        "claim_boundary": (
            "Fair-baseline rule audit on exposed non-IDP development data. "
            "Model likelihood is not binding or confirmatory evidence."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_correction_v3_robustness/"
            "source_sensitivity.json"
        ),
    )
    parser.add_argument(
        "--loo", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_correction_v3_robustness/"
            "leave_one_pose_out.json"
        ),
    )
    parser.add_argument(
        "--abstention", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_final_confirmation_v1/"
            "abstention_development_audit.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_final_confirmation_v1/"
            "fair_baseline_development_audit.json"
        ),
    )
    args = parser.parse_args()
    result = analyze(
        ROOT / args.source, ROOT / args.loo, ROOT / args.abstention,
        ROOT / args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "matched_pairs": result["matched_pair_count"],
        "admitted_pairs": result["admitted_pair_count"],
        "ensemble_favored_all": result["ensemble_favored_all_pairs"],
        "ensemble_favored_admitted": result["ensemble_favored_admitted_pairs"],
        "median_fair_difference_all": result["median_fair_difference_all_pairs"],
        "median_fair_difference_admitted": result[
            "median_fair_difference_admitted_pairs"
        ],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
