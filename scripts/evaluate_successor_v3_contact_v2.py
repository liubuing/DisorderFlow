#!/usr/bin/env python
"""Evaluate a contact-v2 checkpoint on the held-out exposed component fold."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import load_config  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from scripts.evaluate_successor_v3_development import contact_labels  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(labels, scores):
    return {
        "n": len(labels), "positive_fraction": float(np.mean(labels)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def component_metrics(component_values, seed=5209, draws=10000):
    rows = []
    for component_id, values in sorted(component_values.items()):
        if len(set(values["labels"])) < 2:
            continue
        rows.append({"component_id": component_id, **metrics(
            values["labels"], values["scores"])})
    generator = np.random.default_rng(seed)
    aurocs = np.asarray([row["auroc"] for row in rows])
    average_precisions = np.asarray([row["average_precision"] for row in rows])
    indices = generator.integers(0, len(rows), size=(draws, len(rows)))
    return {
        "eligible_components": len(rows),
        "macro_auroc": float(aurocs.mean()),
        "macro_auroc_component_bootstrap_ci95": [
            float(value) for value in np.quantile(aurocs[indices].mean(axis=1), [0.025, 0.975])],
        "macro_average_precision": float(average_precisions.mean()),
        "macro_average_precision_component_bootstrap_ci95": [
            float(value) for value in np.quantile(
                average_precisions[indices].mean(axis=1), [0.025, 0.975])],
        "components": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/train/bfn_successor_v3_contact_v2_exposed.yml")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baselines", type=Path,
        default=Path("results/successor_v3_contact_v2/baselines.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint_path = ROOT / args.checkpoint
    manifest_path = ROOT / args.dataset_manifest
    config, _ = load_config(args.config)
    dataset = get_dataset(copy.deepcopy(config.dataset.val))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_rows = sorted(
        [row for row in manifest["records"] if row["fold"] == manifest["eval_fold"]],
        key=lambda row: row["instance"])
    if len(eval_rows) != len(dataset):
        raise RuntimeError("Evaluation LMDB and manifest disagree")

    model = get_model(copy.deepcopy(config.model)).to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    residue_labels, residue_scores = [], []
    pair_labels, pair_scores = [], []
    residue_by_component, pair_by_component = {}, {}
    rows = []
    for index, (meta, item) in enumerate(zip(eval_rows, dataset), 1):
        batch = recursive_to(PaddingCollate()([item]), args.device)
        with torch.inference_mode():
            output = model.score(batch, fixed_t=0.5)
        labels, generated = contact_labels(batch)
        residue_target = labels[0][generated[0]]
        residue_probability = torch.sigmoid(output["contact"][0][generated[0]])
        ca = batch["pos_heavyatom"][0, :, 1]
        cb = batch["pos_heavyatom"][0, :, 4]
        cb_mask = batch["mask_heavyatom"][0, :, 4].bool()
        positions = torch.where(cb_mask.unsqueeze(-1), cb, ca)
        antigen = batch["mask_antigen"][0].bool()
        pair_target = (torch.cdist(
            positions[generated[0]], positions[antigen]) < 8.0).float()
        pair_probability = torch.sigmoid(
            output["contact_pair"][0][generated[0]][:, antigen])
        residue_labels.extend(residue_target.cpu().int().tolist())
        residue_scores.extend(residue_probability.cpu().tolist())
        pair_labels.extend(pair_target.flatten().cpu().int().tolist())
        pair_scores.extend(pair_probability.flatten().cpu().tolist())
        residue_values = residue_by_component.setdefault(
            meta["component_id"], {"labels": [], "scores": []})
        residue_values["labels"].extend(residue_target.cpu().int().tolist())
        residue_values["scores"].extend(residue_probability.cpu().tolist())
        pair_values = pair_by_component.setdefault(
            meta["component_id"], {"labels": [], "scores": []})
        pair_values["labels"].extend(pair_target.flatten().cpu().int().tolist())
        pair_values["scores"].extend(pair_probability.flatten().cpu().tolist())
        rows.append({
            "instance": meta["instance"], "component_id": meta["component_id"],
            "residues": int(residue_target.numel()),
            "pairs": int(pair_target.numel()),
        })
        print(f"Evaluated {index}/{len(dataset)}", flush=True)

    residue = metrics(residue_labels, residue_scores)
    pair = metrics(pair_labels, pair_scores)
    residue_components = component_metrics(residue_by_component, seed=5209)
    pair_components = component_metrics(pair_by_component, seed=5227)
    baseline_path = ROOT / args.baselines
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    matched_baselines = {
        name: values["matched_held_out_fold_4"]
        for name, values in baselines["arms"].items()}
    frozen_encoder_auroc = matched_baselines["frozen_encoder_logistic_probe"]["auroc"]
    gates = {
        "held_out_residue_auroc_at_least_0_80": residue["auroc"] >= 0.80,
        "held_out_pair_auroc_at_least_0_80": pair["auroc"] >= 0.80,
        "held_out_pair_ap_above_prevalence": pair["average_precision"] > pair["positive_fraction"],
        "residue_auroc_above_matched_frozen_encoder_probe": (
            residue["auroc"] > frozen_encoder_auroc),
        "residue_component_macro_ci95_lower_at_least_0_70": (
            residue_components["macro_auroc_component_bootstrap_ci95"][0] >= 0.70),
        "finite_scores": bool(np.isfinite(residue_scores).all() and np.isfinite(pair_scores).all()),
    }
    report = {
        "schema_version": 1,
        "status": "exposed_contact_v2_development_evaluation_complete",
        "classification": "retrospective exposed development; not confirmatory or external",
        "checkpoint": {"path": args.checkpoint.as_posix(), "sha256": sha256(checkpoint_path)},
        "config": {"path": args.config, "sha256": sha256(ROOT / args.config)},
        "dataset_manifest": {
            "path": args.dataset_manifest.as_posix(), "sha256": sha256(manifest_path)},
        "fixed_t": 0.5,
        "held_out_fold": manifest["eval_fold"],
        "held_out_records": len(dataset),
        "residue_contact": residue,
        "pair_contact": pair,
        "component_level": {
            "residue_contact": residue_components,
            "pair_contact": pair_components,
        },
        "matched_same_record_baselines": {
            "artifact": {"path": args.baselines.as_posix(), "sha256": sha256(baseline_path)},
            "held_out_fold_4": matched_baselines,
            "candidate_minus_frozen_encoder_residue_auroc": (
                residue["auroc"] - frozen_encoder_auroc),
        },
        "gates": gates,
        "eligible_for_future_confirmation": all(gates.values()),
        "rows": rows,
        "claim_boundary": "These metrics select a frozen candidate only; they do not confirm performance.",
    }
    output_path = ROOT / args.output
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: report[key] for key in (
        "residue_contact", "pair_contact", "gates", "eligible_for_future_confirmation")}, indent=2))


if __name__ == "__main__":
    main()
