#!/usr/bin/env python
"""Prepare readiness to sample components that currently have only experimental poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(readiness_path, donor_audit_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite resampling readiness: {output}")
    readiness = json.loads(readiness_path.read_text(encoding="ascii"))
    donor = json.loads(donor_audit_path.read_text(encoding="ascii"))
    experimental_ready = {
        row["component_id"] for row in donor["components"]
        if row["experimental_pose_ready"]
    }
    components = []
    for row in readiness["components"]:
        component = dict(row)
        component["multi_pose_ready"] = row["component_id"] not in experimental_ready
        component["sampling_requested_for_source_balance"] = (
            row["component_id"] in experimental_ready
        )
        components.append(component)
    payload = {
        "schema_version": 1,
        "status": "expanded_source_balance_resampling_readiness",
        "classification": "retrospective_expanded_idp_pose_development",
        "component_count": len(components),
        "sampling_requested_count": sum(
            row["sampling_requested_for_source_balance"] for row in components
        ),
        "components": components,
        "future_confirmation_eligible_count": 0,
        "claim_boundary": (
            "Readiness for sampled-pose source balancing on exposed development "
            "components only; no confirmation claim"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "generation_readiness.json"
        ),
    )
    parser.add_argument(
        "--donor-audit", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "donor_sequence_audit.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "source_balance_resampling_readiness.json"
        ),
    )
    args = parser.parse_args()
    result = prepare(
        ROOT / args.readiness, ROOT / args.donor_audit, ROOT / args.output
    )
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "sampling_requested": result["sampling_requested_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
