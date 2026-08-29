#!/usr/bin/env python
"""Audit two-axis source and leave-one-pose stability without reranking.

An abstained pair is excluded from conditional summaries; it is never recorded
as a zero effect. This diagnostic cannot change the main level-2 decision
because the rule was examined after candidate results existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(config_path, output_path=None):
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    payloads = {}
    provenance = {}
    for name, item in config["inputs"].items():
        path = ROOT / item["path"]
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Frozen robustness hash mismatch: {path}")
        payloads[name] = json.loads(path.read_text(encoding="ascii"))
        provenance[name] = {"path": item["path"], "sha256": digest}

    # Join by the scientific unit (component, exact mutation bucket), not by
    # array order, because independently generated artifacts may sort records
    # differently.
    source = {
        (row["component_id"], int(row["substitution_bucket"])): row
        for row in payloads["source_sensitivity"]["matched_bucket_contrasts"]
    }
    leave_one_out = {
        (component["component_id"], int(row["substitution_bucket"])): row
        for component in payloads["leave_one_pose_out"]["components"]
        for row in component["matched_bucket_stability"]
    }
    if set(source) != set(leave_one_out):
        raise ValueError("Robustness artifacts contain different matched pairs")

    records = []
    for key in sorted(source):
        source_row = source[key]
        loo_row = leave_one_out[key]
        # Both axes are required. Passing only source sensitivity or only LOO
        # stability is insufficient for a robust candidate recommendation.
        eligible = (
            source_row["source_panel_sign_consistent"]
            and loo_row["direction_stable_across_all_omissions"]
        )
        mixed_effect = float(
            source_row["ensemble_minus_single_state_log_probability"]["mixed"]
        )
        records.append({
            "component_id": key[0],
            "substitution_bucket": key[1],
            "source_panel_stable": source_row["source_panel_sign_consistent"],
            "leave_one_pose_out_stable": loo_row[
                "direction_stable_across_all_omissions"
            ],
            "eligible": eligible,
            "mixed_ensemble_minus_single_state": mixed_effect,
            "eligible_direction": (
                1 if mixed_effect > 0 else -1 if mixed_effect < 0 else 0
            ) if eligible else None,
        })
    eligible = [row for row in records if row["eligible"]]
    output = {
        "schema_version": 1,
        "status": "two_axis_stability_abstention_diagnostic_complete",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "provenance": provenance,
        "attempted_pairs": len(records),
        "eligible_pairs": len(eligible),
        "coverage": len(eligible) / len(records),
        "eligible_ensemble_favored_pairs": sum(
            row["eligible_direction"] == 1 for row in eligible
        ),
        "eligible_single_state_favored_pairs": sum(
            row["eligible_direction"] == -1 for row in eligible
        ),
        "eligible_tied_pairs": sum(row["eligible_direction"] == 0 for row in eligible),
        "records": records,
        "decision": "diagnostic_only_do_not_change_level2_status",
        "claim_boundary": config["claim_boundary"],
    }
    output_path = Path(output_path or ROOT / config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_level2_stability_abstention_diagnostic_v1.yml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(ROOT / args.config, ROOT / args.output if args.output else None)
    print(json.dumps({
        key: result[key] for key in (
            "status", "attempted_pairs", "eligible_pairs", "coverage",
            "eligible_ensemble_favored_pairs",
            "eligible_single_state_favored_pairs", "eligible_tied_pairs", "decision",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
