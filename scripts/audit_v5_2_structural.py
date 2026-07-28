#!/usr/bin/env python3
"""Audit v5.2 structural caches and the full-group FP16 training gate."""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from easydict import EasyDict


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disorderflow.data_factory import sha256_file
from disorderflow.datasets.statecontrast_structural import StateContrastStructuralDataset


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_summary(split, parent_lmdb):
    dataset = StateContrastStructuralDataset(EasyDict({
        "records_dir": str(ROOT / "data/statecontrast_factory_v2_1"),
        "parent_lmdb_path": str(parent_lmdb),
        "split": split,
        "expected_schema": "statecontrast_factory_v2_1",
    }))
    return {
        "records": len(dataset),
        "groups": len(dataset.group_indices),
        "group_sizes": dict(sorted(Counter(map(len, dataset.group_indices)).items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", default="data/statecontrast_structural_v2_1")
    parser.add_argument(
        "--output", default="data/statecontrast_structural_v2_1/cache_audit.json")
    args = parser.parse_args()

    cache_dir = ROOT / args.cache_dir
    train_lmdb = cache_dir / "train_parents.lmdb"
    validation_lmdb = cache_dir / "validation_parents.lmdb"
    train_ids_path = Path(str(train_lmdb) + "-ids")
    validation_ids_path = Path(str(validation_lmdb) + "-ids")
    train_report = read_json(train_lmdb.with_suffix(".build_report.json"))
    validation_report = read_json(validation_lmdb.with_suffix(".build_report.json"))
    gate = read_json(cache_dir / "fp16_training_gate.json")
    formal_gate = read_json(cache_dir / "formal_training_gate.json")
    production = read_json(cache_dir / "production_training_audit.json")
    factory = read_json(ROOT / "data/statecontrast_factory_v2_1/manifest.json")
    train_ids = set(pickle.loads(train_ids_path.read_bytes()))
    validation_ids = set(pickle.loads(validation_ids_path.read_bytes()))

    checks = {
        "factory_schema": factory["schema_version"] == "statecontrast_factory_v2_1",
        "train_build_pass": train_report["status"] == "pass",
        "validation_build_pass": validation_report["status"] == "pass",
        "train_lmdb_checksum": sha256_file(train_lmdb) == train_report["parent_lmdb_sha256"],
        "validation_lmdb_checksum": (
            sha256_file(validation_lmdb) == validation_report["parent_lmdb_sha256"]),
        "train_ids_checksum": sha256_file(train_ids_path) == train_report["parent_ids_sha256"],
        "validation_ids_checksum": (
            sha256_file(validation_ids_path) == validation_report["parent_ids_sha256"]),
        "parent_id_disjoint": not (train_ids & validation_ids),
        "fp16_gate_pass": gate["status"] == "pass" and gate.get("amp_dtype") == "fp16",
        "full_group_gate": gate.get("full_group") is True and gate["batch_shape"][0] in (13, 14),
        "finite_nonzero_update": (
            gate["grouped_contrastive_loss"] > 0
            and gate["gradient_l1_total"] > 0
            and gate["parameter_delta_l1"] > 0),
        "formal_train_checkpoint_gate": (
            formal_gate["status"] == "pass"
            and formal_gate["successful_optimizer_steps"] >= 2
            and formal_gate["checks"]["best_checkpoint_marked_ema"]),
        "production_training_pass": (
            production["status"] == "pass"
            and production["iteration"] == 2000
            and production["successful_optimizer_steps"] == 2000
            and production["checks"]["no_nonfinite_training_events"]),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "factory_fingerprint": factory["canonical_fingerprint"],
        "checks": checks,
        "train": {
            **dataset_summary("train", train_lmdb),
            "parent_cache_records": len(train_ids),
            "parent_lmdb_sha256": train_report["parent_lmdb_sha256"],
            "rejected_parents": train_report["failed_parents"],
        },
        "validation": {
            **dataset_summary("validation", validation_lmdb),
            "parent_cache_records": len(validation_ids),
            "parent_lmdb_sha256": validation_report["parent_lmdb_sha256"],
            "rejected_parents": validation_report["failed_parents"],
        },
        "parent_id_overlap": len(train_ids & validation_ids),
        "fp16_training_gate": gate,
        "formal_training_gate": formal_gate,
        "production_training": production,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "pass":
        raise RuntimeError("v5.2 structural audit failed")


if __name__ == "__main__":
    main()
