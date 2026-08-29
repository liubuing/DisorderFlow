#!/usr/bin/env python
"""Freeze five component-level StateContrast-v2 fold configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(base_config_path, manifest_path, output_dir, folds=5):
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    observed = int(manifest.get("cross_validation_folds", 0))
    if observed != int(folds):
        raise ValueError(f"Manifest has {observed} folds, expected {folds}")
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    unexpected = [path for path in existing if path.name != "isolation_audit.json"]
    if unexpected:
        raise FileExistsError(f"Refusing to overwrite CV contract: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    for fold in range(int(folds)):
        config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
        config["lineage"]["cross_validation_fold"] = fold
        config["lineage"]["cross_validation_folds"] = int(folds)
        config["dataset"]["train"].update({
            "heldout_fold": fold, "fold_role": "train"})
        config["dataset"]["val"].update({
            "heldout_fold": fold, "fold_role": "validation"})
        path = output_dir / f"fold_{fold}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs.append({
            "fold": fold,
            "config": path.relative_to(ROOT).as_posix(),
            "config_sha256": sha256(path),
        })
    contract = {
        "schema_version": 1,
        "status": "statecontrast_v2_cross_validation_frozen",
        "base_config": base_config_path.relative_to(ROOT).as_posix(),
        "base_config_sha256": sha256(base_config_path),
        "training_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "training_manifest_sha256": sha256(manifest_path),
        "folds": configs,
        "initializer": base["lineage"]["allowed_initializer"],
        "initializer_sha256": base["lineage"]["allowed_initializer_sha256"],
        "evaluation_split": "val",
        "claim_boundary": "Exposed component-level cross-validation only",
    }
    contract_path = output_dir / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="ascii")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path(
        "configs/train/bfn_statecontrast_v2_expanded_idp.yml"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_expanded_states_v1/training_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = prepare(
        ROOT / args.base_config, ROOT / args.manifest,
        ROOT / args.output_dir, args.folds)
    print(json.dumps({
        "status": result["status"],
        "folds": len(result["folds"]),
        "contract": str((ROOT / args.output_dir / "contract.json").relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
