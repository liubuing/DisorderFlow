#!/usr/bin/env python
"""One-shot contact-v2 evaluation on untouched independent future components."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import load_config  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from scripts.evaluate_successor_v3_contact_v2 import (  # noqa: E402
    component_metrics,
    metrics,
    sha256,
)
from scripts.evaluate_successor_v3_development import contact_labels  # noqa: E402


def validate_admission(policy, manifest, manifest_path):
    admission = policy["admission"]
    errors = []
    if admission["current_state"]["ready"] is not True:
        errors.append("future policy is not marked ready")
    if manifest.get("classification") != "future_confirmatory_untouched":
        errors.append("manifest is not classified as untouched future confirmation")
    if manifest.get("checkpoint_accessed_before_manifest_freeze") is not False:
        errors.append("checkpoint-access declaration is not false")
    if manifest.get("reference_union_sha256") != admission["reference_union"]["sha256"]:
        errors.append("manifest reference-union hash differs from frozen v3 exposure ledger")
    records = manifest.get("records", [])
    components = [row.get("component_id") for row in records]
    if None in components or len(set(components)) != len(components):
        errors.append("confirmation requires exactly one representative per component")
    required = admission["minimum_independent_homology_components"]
    if len(set(components)) < required:
        errors.append(f"confirmation requires at least {required} independent components")
    if manifest.get("all_pairwise_axes_isolated") is not True:
        errors.append("manifest does not assert pairwise isolation on every frozen axis")
    if errors:
        raise RuntimeError(f"Admission failed for {manifest_path}: {errors}")


def load_model(config_path, checkpoint_path, device):
    config, _ = load_config(config_path)
    model = get_model(copy.deepcopy(config.model)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, config


def paired_component_difference(candidate, baseline, seed=5261, draws=10000):
    candidate_rows = {row["component_id"]: row for row in candidate["components"]}
    baseline_rows = {row["component_id"]: row for row in baseline["components"]}
    common = sorted(candidate_rows.keys() & baseline_rows.keys())
    differences = np.asarray([
        candidate_rows[value]["auroc"] - baseline_rows[value]["auroc"]
        for value in common])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(common), size=(draws, len(common)))
    means = differences[indices].mean(axis=1)
    return {
        "components": len(common), "mean_auroc_difference": float(differences.mean()),
        "component_bootstrap_ci95": [
            float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--attempt-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    resolved = {name: ROOT / value for name, value in {
        "policy": args.policy, "manifest": args.manifest, "db": args.db_path,
        "attempt": args.attempt_record, "output": args.output}.items()}
    if resolved["attempt"].exists() or resolved["output"].exists():
        raise FileExistsError("One-shot attempt or result already exists; rerun is prohibited")
    policy = yaml.safe_load(resolved["policy"].read_text(encoding="ascii"))
    manifest = json.loads(resolved["manifest"].read_text(encoding="utf-8"))
    validate_admission(policy, manifest, resolved["manifest"])
    candidate_path = ROOT / policy["candidate_model"]["path"]
    predecessor_path = ROOT / policy["matched_baselines"]["predecessor_checkpoint"]["path"]
    if sha256(candidate_path) != policy["candidate_model"]["sha256"]:
        raise RuntimeError("Frozen candidate checkpoint hash mismatch")
    if sha256(predecessor_path) != policy["matched_baselines"]["predecessor_checkpoint"]["sha256"]:
        raise RuntimeError("Frozen predecessor checkpoint hash mismatch")

    resolved["attempt"].parent.mkdir(parents=True, exist_ok=True)
    resolved["attempt"].write_text(json.dumps({
        "status": "started_irreversible_one_shot_attempt",
        "manifest_sha256": sha256(resolved["manifest"]),
        "candidate_sha256": policy["candidate_model"]["sha256"],
        "policy_sha256": sha256(resolved["policy"]),
    }, indent=2) + "\n", encoding="ascii")

    candidate, config = load_model(
        "configs/train/bfn_successor_v3_contact_v2_exposed.yml", candidate_path, args.device)
    predecessor, _ = load_model(
        "configs/train/bfn_successor_v3_h3_specificity.yml", predecessor_path, args.device)
    dataset_config = copy.deepcopy(config.dataset.val)
    dataset_config.db_path = str(resolved["db"])
    dataset = get_dataset(dataset_config)
    records = manifest["records"]
    if len(dataset) != len(records):
        raise RuntimeError("Confirmation LMDB and frozen manifest length differ")

    stores = {
        "candidate_residue": {}, "candidate_pair": {},
        "predecessor_residue": {}, "geometry_ceiling": {}}
    pooled = {name: {"labels": [], "scores": []} for name in stores}
    for meta, item in zip(records, dataset, strict=True):
        batch = recursive_to(PaddingCollate()([item]), args.device)
        with torch.inference_mode():
            candidate_output = candidate.score(batch, fixed_t=policy["evaluation"]["fixed_t"])
            predecessor_output = predecessor.score(batch, fixed_t=policy["evaluation"]["fixed_t"])
        labels, generated = contact_labels(batch)
        antigen = batch["mask_antigen"][0].bool()
        ca = batch["pos_heavyatom"][0, :, 1]
        cb = batch["pos_heavyatom"][0, :, 4]
        cb_mask = batch["mask_heavyatom"][0, :, 4].bool()
        positions = torch.where(cb_mask.unsqueeze(-1), cb, ca)
        distances = torch.cdist(positions[generated[0]], positions[antigen])
        residue_labels = labels[0][generated[0]].cpu().int().tolist()
        pair_labels = (distances < 8.0).flatten().cpu().int().tolist()
        values = {
            "candidate_residue": (
                residue_labels, torch.sigmoid(candidate_output["contact"][0][generated[0]]).cpu().tolist()),
            "candidate_pair": (
                pair_labels, torch.sigmoid(candidate_output["contact_pair"][0][generated[0]][:, antigen]).flatten().cpu().tolist()),
            "predecessor_residue": (
                residue_labels, torch.sigmoid(predecessor_output["contact"][0][generated[0]]).cpu().tolist()),
            "geometry_ceiling": (
                residue_labels, torch.sigmoid(8.0 - distances.min(dim=1).values).cpu().tolist()),
        }
        for name, (target, score) in values.items():
            component = stores[name].setdefault(
                meta["component_id"], {"labels": [], "scores": []})
            component["labels"].extend(target)
            component["scores"].extend(score)
            pooled[name]["labels"].extend(target)
            pooled[name]["scores"].extend(score)

    summaries = {
        name: {
            "pooled_descriptive": metrics(values["labels"], values["scores"]),
            "component_level_primary": component_metrics(stores[name]),
        } for name, values in pooled.items()
    }
    comparison = paired_component_difference(
        summaries["candidate_residue"]["component_level_primary"],
        summaries["predecessor_residue"]["component_level_primary"])
    gates = {
        "candidate_residue_macro_auroc_ci95_lower_at_least_0_70": (
            summaries["candidate_residue"]["component_level_primary"]
            ["macro_auroc_component_bootstrap_ci95"][0] >= 0.70),
        "candidate_pair_macro_auroc_ci95_lower_at_least_0_70": (
            summaries["candidate_pair"]["component_level_primary"]
            ["macro_auroc_component_bootstrap_ci95"][0] >= 0.70),
        "candidate_minus_predecessor_component_ci95_lower_above_zero": (
            comparison["component_bootstrap_ci95"][0] > 0),
    }
    report = {
        "schema_version": 1, "status": "future_confirmation_complete",
        "classification": "one-shot untouched future confirmation",
        "manifest": {"path": args.manifest.as_posix(), "sha256": sha256(resolved["manifest"])},
        "policy": {"path": args.policy.as_posix(), "sha256": sha256(resolved["policy"])},
        "summaries": summaries, "candidate_minus_predecessor": comparison,
        "gates": gates, "all_confirmatory_gates_passed": all(gates.values()),
        "claim_boundary": (
            "This confirms bound-structure contact-map decoding only, not binding, "
            "affinity, hotspot causality, specificity, or design success."),
    }
    resolved["output"].parent.mkdir(parents=True, exist_ok=True)
    resolved["output"].write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"gates": gates, "all_confirmatory_gates_passed": all(gates.values())}, indent=2))


if __name__ == "__main__":
    main()
