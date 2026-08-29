#!/usr/bin/env python
"""Select a target-balanced, lineage-deduplicated expanded development panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolution_value(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed > 0 else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def select(config_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite expanded panel: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    registry_path = ROOT / config["input"]["registry"]
    if sha256(registry_path) != config["input"]["registry_sha256"]:
        raise RuntimeError("Expanded registry hash changed after selection freeze")
    registry = json.loads(registry_path.read_text(encoding="ascii"))
    best_by_lineage = {}
    for row in registry["records"]:
        key = (row["target"], row["lineage_proxy"])
        candidate_key = (
            resolution_value(row["resolution"]),
            row["pdb_id"],
            row["heavy_chain"],
        )
        if key not in best_by_lineage or candidate_key < best_by_lineage[key][0]:
            best_by_lineage[key] = (candidate_key, row)
    grouped = defaultdict(list)
    for (_target, _lineage), (sort_key, row) in best_by_lineage.items():
        grouped[row["target"]].append((sort_key, row))
    maximum = config["selection"]["maximum_components_per_target"]
    selected = []
    target_summary = {}
    for target in sorted(grouped):
        rows = [row for _, row in sorted(grouped[target])[:maximum]]
        selected.extend(rows)
        target_summary[target] = {
            "available_lineages": len(grouped[target]),
            "selected_components": len(rows),
        }
    target_count = len(target_summary)
    if target_count < config["selection"]["minimum_targets"]:
        raise RuntimeError("Expanded registry has too few eligible targets")
    payload = {
        "schema_version": 1,
        "status": "balanced_expanded_development_panel_frozen",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "component_count": len(selected),
        "target_count": target_count,
        "lineage_proxy_count": len({row["lineage_proxy"] for row in selected}),
        "target_summary": target_summary,
        "components": selected,
        "future_confirmation_eligible_count": 0,
        "decision": "materialize_coordinates_for_expanded_development_only",
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/benchmarks/idp_ensemble_expanded_development_v2_selection.yml"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = select(config_path, ROOT / config["output"])
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "targets": result["target_count"],
        "lineage_proxies": result["lineage_proxy_count"],
        "target_summary": result["target_summary"],
        "future_confirmation_eligible": result[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
