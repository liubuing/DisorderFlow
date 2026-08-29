#!/usr/bin/env python
"""Validate and merge all v3 per-component candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def substitutions(sequence, native):
    if len(sequence) != len(native):
        raise ValueError("Candidate and native H3 lengths differ")
    return sum(left != right for left, right in zip(sequence, native, strict=True))


def finalize(panel_path, candidate_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidates: {output}")
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    buckets = panel["mutation_space"]["substitution_buckets"]
    seeds = panel["seeds"]
    expected = [row["component_id"] for row in panel["components"]]
    merged = {}
    source_artifacts = []
    for component_id in expected:
        path = candidate_dir / f"{component_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing component candidates: {path}")
        payload = json.loads(path.read_text(encoding="ascii"))
        if payload.get("status") != "component_candidates_complete":
            raise RuntimeError(f"Incomplete component artifact: {component_id}")
        if payload.get("seeds") != seeds or payload.get("substitution_buckets") != buckets:
            raise RuntimeError(f"Frozen budget mismatch: {component_id}")
        native = payload["native_h3"]
        if set(payload["arms"]) != {"ensemble", "single_state"}:
            raise RuntimeError(f"Arm mismatch: {component_id}")
        for arm_name, arm in payload["arms"].items():
            rows = arm["candidates"]
            if len(rows) != len(buckets):
                raise RuntimeError(f"Candidate count mismatch: {component_id}/{arm_name}")
            if len({row["sequence"] for row in rows}) != len(rows):
                raise RuntimeError(f"Duplicate candidate: {component_id}/{arm_name}")
            for index, row in enumerate(rows):
                bucket = buckets[index]
                if (
                    row["seed"] != seeds[index]
                    or row["substitution_bucket"] != bucket
                    or row["substitutions"] != bucket
                    or substitutions(row["sequence"], native) != bucket
                ):
                    raise RuntimeError(
                        f"Bucket validation failed: {component_id}/{arm_name}/{bucket}"
                    )
            control = arm["native_control"]
            if control["sequence"] != native or control["substitutions"] != 0:
                raise RuntimeError(f"Native control mismatch: {component_id}/{arm_name}")
        if payload["arms"]["ensemble"]["pose_count"] < 2:
            raise RuntimeError(f"Ensemble arm has fewer than two poses: {component_id}")
        if payload["arms"]["single_state"]["pose_count"] != 1:
            raise RuntimeError(f"Single-state arm pose count is not one: {component_id}")
        merged[component_id] = payload
        source_artifacts.append({"path": str(path), "sha256": sha256(path)})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "status": "complete",
        "classification": "v3_untouched_panel_computational_candidates",
        "panel_plan": str(panel_path),
        "panel_plan_sha256": sha256(panel_path),
        "component_count": len(merged),
        "candidate_count": len(merged) * 2 * len(buckets),
        "native_control_count": len(merged),
        "source_artifacts": source_artifacts,
        "components": merged,
        "claim_boundary": (
            "computational candidates for a general antibody multi-conformer "
            "panel; no binding, IDP-specific, or therapeutic claim"
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.panel, args.candidate_dir, args.output)
    print(json.dumps({
        key: result[key]
        for key in ("status", "component_count", "candidate_count", "native_control_count")
    }, indent=2))


if __name__ == "__main__":
    main()
