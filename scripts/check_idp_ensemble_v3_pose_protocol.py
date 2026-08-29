#!/usr/bin/env python
"""Check that a frozen v3 pose manifest contains distinct accepted poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def check(protocol_path, panel_path, pose_manifest, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pose status: {output}")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="ascii"))
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    manifest = (
        json.loads(pose_manifest.read_text(encoding="ascii"))
        if pose_manifest.is_file() else {"components": []}
    )
    expected = {row["component_id"] for row in panel.get("components", [])}
    minimum = protocol["requirements"]["minimum_poses_per_component"]
    observed = {}
    for row in manifest.get("components", []):
        accepted = [pose for pose in row.get("poses", []) if pose.get("accepted") is True]
        hashes = {pose.get("coordinate_sha256") for pose in accepted if pose.get("coordinate_sha256")}
        observed[row.get("component_id")] = {
            "accepted_poses": len(accepted),
            "distinct_coordinate_hashes": len(hashes),
            "ready": len(accepted) >= minimum and len(hashes) >= minimum,
        }
    ready_ids = {component_id for component_id, row in observed.items() if row["ready"]}
    ready = ready_ids == expected
    payload = {
        "schema_version": 1,
        "status": "pose_panel_ready" if ready else "pose_panel_blocked",
        "classification": "v3_pose_panel_readiness",
        "required_components": len(expected),
        "ready_components": len(ready_ids & expected),
        "minimum_poses_per_component": minimum,
        "components": observed,
        "decision": "allow_candidate_generation" if ready else "generate_missing_distinct_poses",
        "claim_boundary": protocol["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.protocol, args.panel, args.poses, args.output), indent=2))


if __name__ == "__main__":
    main()
