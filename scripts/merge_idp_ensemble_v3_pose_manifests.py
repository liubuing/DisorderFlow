#!/usr/bin/env python
"""Merge accepted experimental and sampled v3 pose manifests for checking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(panel_plan, experimental_manifest, sampled_manifest, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged pose manifest: {output}")
    panel = json.loads(panel_plan.read_text(encoding="ascii"))
    experimental = json.loads(experimental_manifest.read_text(encoding="ascii"))
    sampled = (
        json.loads(sampled_manifest.read_text(encoding="ascii"))
        if sampled_manifest.is_file() else {"components": []}
    )
    by_component = {}
    for manifest in (experimental, sampled):
        for row in manifest.get("components", []):
            by_component.setdefault(row["component_id"], []).extend(
                pose for pose in row["poses"] if pose.get("accepted") is True
            )
    expected = [row["component_id"] for row in panel.get("components", [])]
    missing = [
        component_id for component_id in expected
        if component_id not in by_component
    ]
    if missing:
        raise RuntimeError(f"Pose manifests miss components: {missing}")
    duplicate_hashes = 0
    components = []
    for component_id in expected:
        poses = by_component[component_id]
        hashes = [pose["coordinate_sha256"] for pose in poses]
        if len(set(hashes)) != len(hashes):
            duplicate_hashes += len(hashes) - len(set(hashes))
        components.append({
            "component_id": component_id,
            "poses": poses,
            "accepted_pose_count": len(poses),
            "distinct_coordinate_hashes": len(set(hashes)),
        })
    payload = {
        "schema_version": 1,
        "status": "merged_pose_manifest_complete",
        "classification": "v3_combined_pose_panel",
        "panel_plan": str(panel_plan),
        "panel_plan_sha256": sha256(panel_plan),
        "experimental_manifest": str(experimental_manifest),
        "experimental_manifest_sha256": sha256(experimental_manifest),
        "sampled_manifest": str(sampled_manifest),
        "sampled_manifest_sha256": sha256(sampled_manifest),
        "duplicate_pose_hashes_across_sources": duplicate_hashes,
        "component_count": len(components),
        "components": components,
        "claim_boundary": (
            "merged experimental and restrained-sampled poses for a general "
            "antibody multi-conformer panel; no candidate or performance result"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-plan", type=Path, required=True)
    parser.add_argument("--experimental", type=Path, required=True)
    parser.add_argument("--sampled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.panel_plan, args.experimental, args.sampled, args.output)
    print(json.dumps({
        "status": result["status"],
        "accepted_pose_counts": {
            row["component_id"]: row["accepted_pose_count"]
            for row in result["components"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
