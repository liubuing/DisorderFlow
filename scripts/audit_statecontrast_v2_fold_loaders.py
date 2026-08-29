#!/usr/bin/env python
"""Instantiate every frozen fold and verify train/validation loader boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import disorderflow.datasets.statecontrast_v2_pose_manifest  # noqa: E402,F401
from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.utils.misc import load_config  # noqa: E402


def audit(contract_path, output_path):
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    rows = []
    all_validation_components = set()
    for item in contract["folds"]:
        config, _ = load_config(str(ROOT / item["config"]))
        train = get_dataset(config.dataset.train)
        validation = get_dataset(config.dataset.val)
        train_components = {row["component_id"] for row in train.records}
        validation_components = {row["component_id"] for row in validation.records}
        overlap = train_components & validation_components
        duplicate_validation = all_validation_components & validation_components
        all_validation_components.update(validation_components)
        rows.append({
            "fold": item["fold"],
            "train_records": len(train),
            "validation_records": len(validation),
            "train_components": len(train_components),
            "validation_components": len(validation_components),
            "component_overlap": sorted(overlap),
            "duplicate_validation_components": sorted(duplicate_validation),
            "maximum_train_group_size": max(map(len, train.group_indices)),
            "maximum_validation_group_size": max(map(len, validation.group_indices)),
            "batch_capacity": int(config.train.batch_size),
        })
    checks = {
        "no_train_validation_component_overlap": all(
            not row["component_overlap"] for row in rows),
        "validation_components_appear_once": all(
            not row["duplicate_validation_components"] for row in rows),
        "atomic_groups_fit_batch": all(
            row["maximum_train_group_size"] <= row["batch_capacity"]
            and row["maximum_validation_group_size"] <= row["batch_capacity"]
            for row in rows),
    }
    payload = {
        "schema_version": 1,
        "status": (
            "statecontrast_v2_fold_loader_audit_passed"
            if all(checks.values()) else "statecontrast_v2_fold_loader_audit_failed"),
        "checks": checks,
        "heldout_component_count": len(all_validation_components),
        "folds": rows,
        "decision": "execute_cv_only_when_all_loader_checks_pass",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/contract.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/fold_loader_audit.json"))
    args = parser.parse_args()
    result = audit(ROOT / args.contract, ROOT / args.output)
    print(json.dumps({
        "status": result["status"], "checks": result["checks"],
        "heldout_components": result["heldout_component_count"],
        "folds": result["folds"],
    }, indent=2))


if __name__ == "__main__":
    main()
