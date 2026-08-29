#!/usr/bin/env python
"""Emit an opaque 108-construct v3 assay manifest and sealed blind key."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blind(candidates_path, panel_path, manifest_path, key_path):
    if manifest_path.exists() or key_path.exists():
        raise FileExistsError("Refusing to overwrite blinded v3 outputs")
    candidates = json.loads(candidates_path.read_text(encoding="ascii"))
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    if candidates.get("status") != "complete":
        raise RuntimeError("Candidate panel is not complete")
    selected, key = [], {}
    for component_id in sorted(candidates["components"]):
        component = candidates["components"][component_id]
        for arm_name in ("ensemble", "single_state"):
            for row in component["arms"][arm_name]["candidates"]:
                code = f"V3-{len(selected) + 1:03d}"
                key[code] = {
                    "component_id": component_id,
                    "arm": arm_name,
                    "sequence": row["sequence"],
                    "substitutions": row["substitutions"],
                    "seed": row["seed"],
                }
                selected.append({
                    "construct_id": code,
                    "component_id": component_id,
                    "substitutions": row["substitutions"],
                    "native_control": False,
                })
        code = f"V3-{len(selected) + 1:03d}"
        key[code] = {
            "component_id": component_id,
            "arm": "native_control",
            "sequence": component["native_h3"],
            "substitutions": 0,
            "seed": None,
        }
        selected.append({
            "construct_id": code,
            "component_id": component_id,
            "substitutions": 0,
            "native_control": True,
        })
    if len(selected) != 108:
        raise RuntimeError(f"Expected 108 constructs, found {len(selected)}")
    random.Random(12041).shuffle(selected)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "blinded_construct_manifest_ready_for_experimental_qc",
        "classification": "v3_blinded_assay_handoff",
        "randomization_seed": 12041,
        "candidate_input": str(candidates_path),
        "candidate_input_sha256": sha256(candidates_path),
        "panel_plan": str(panel_path),
        "panel_plan_sha256": sha256(panel_path),
        "construct_count": len(selected),
        "constructs": selected,
        "claim_boundary": "opaque assay manifest; no sequence, arm, or performance result",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    key_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "sealed_blind_key_do_not_share_with_assay_analyst",
        "manifest_sha256": sha256(manifest_path),
        "construct_count": len(key),
        "constructs": key,
    }, indent=2) + "\n", encoding="ascii")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    result = blind(args.candidates, args.panel, args.manifest, args.key)
    print(json.dumps({
        "status": result["status"],
        "construct_count": result["construct_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
