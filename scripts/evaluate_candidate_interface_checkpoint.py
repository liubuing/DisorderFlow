#!/usr/bin/env python
"""Evaluate candidate-interface checkpoints on complete scaffold groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset
from disorderflow.models import get_model
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate
from disorderflow.utils.data import lmdb_records_sha256
from disorderflow.utils.train import recursive_to


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left, right, ranked=False):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if ranked:
        left, right = rankdata(left), rankdata(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "pearson": correlation(prediction, target),
        "spearman": correlation(prediction, target, ranked=True),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
    }


def grouped_metrics(records, prediction_key, target_key, noise_key=None):
    groups = defaultdict(list)
    for record in records:
        groups[record["scaffold_id"]].append(record)
    spearman = []
    prediction_std = []
    target_std = []
    correct = 0
    pairs = 0
    for group in groups.values():
        prediction = np.asarray([row[prediction_key] for row in group])
        target = np.asarray([row[target_key] for row in group])
        prediction_std.append(float(np.std(prediction)))
        target_std.append(float(np.std(target)))
        value = correlation(prediction, target, ranked=True)
        if value is not None:
            spearman.append(value)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                target_delta = target[left] - target[right]
                noise = np.hypot(
                    float(group[left].get(noise_key, 0.0)),
                    float(group[right].get(noise_key, 0.0)),
                ) if noise_key else 0.0
                if abs(target_delta) <= noise:
                    continue
                prediction_delta = prediction[left] - prediction[right]
                correct += int(np.sign(prediction_delta) == np.sign(target_delta))
                pairs += 1
    return {
        "groups": len(groups),
        "mean_group_spearman": (
            float(np.mean(spearman)) if spearman else None),
        "groups_with_defined_spearman": len(spearman),
        "pairwise_ordering_accuracy": correct / pairs if pairs else None,
        "ordered_pairs": pairs,
        "mean_prediction_std": float(np.mean(prediction_std)),
        "mean_target_std": float(np.mean(target_std)),
    }


def pose_stability(records, prediction_key, target_key, noise_key=None):
    constructs = defaultdict(list)
    for record in records:
        constructs[record["construct_id"]].append(record)
    prediction_std = []
    target_std = []
    for rows in constructs.values():
        prediction_std.append(float(np.std(
            [row[prediction_key] for row in rows], ddof=1)))
        if noise_key:
            target_std.append(float(np.mean([row[noise_key] for row in rows])))
        else:
            target_std.append(float(np.std([row[target_key] for row in rows])))
    return {
        "constructs": len(constructs),
        "mean_prediction_std_across_conditions": float(np.mean(prediction_std)),
        "mean_target_std_across_conditions": float(np.mean(target_std)),
    }


def aggregate_conditions(records):
    constructs = defaultdict(list)
    for record in records:
        constructs[record["construct_id"]].append(record)
    output = []
    for construct_id, rows in constructs.items():
        row = {
            "construct_id": construct_id,
            "scaffold_family": rows[0]["scaffold_family"],
            "scaffold_id": rows[0]["scaffold_family"],
        }
        for name in ("plddt", "iptm", "pae"):
            row[f"pred_{name}"] = float(np.mean(
                [value[f"pred_{name}"] for value in rows]))
            row[f"target_{name}"] = float(np.mean(
                [value[f"target_{name}"] for value in rows]))
            row[f"noise_{name}"] = float(np.mean(
                [value[f"noise_{name}"] for value in rows]))
        output.append(row)
    return output


def evaluate_checkpoint(path, loader, device, expected_manifest_sha256=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    configured_manifest_sha256 = checkpoint.get("config", {}).get(
        "lineage", {}).get("dataset_manifest_sha256")
    if (expected_manifest_sha256 is not None
            and configured_manifest_sha256 != expected_manifest_sha256):
        raise ValueError(
            "Checkpoint dataset lineage does not match the evaluation manifest: "
            f"{configured_manifest_sha256}")
    model = get_model(checkpoint["config"].model)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = recursive_to(batch, device)
            _, pair_feat = model.encode(
                batch,
                remove_structure=model.cfg.get('train_structure', True),
                remove_sequence=model.cfg.get('train_sequence', True),
            )
            batch["pair_feat"] = pair_feat
            scores = model.bfn.score_fixed(batch, fixed_t=0.5)
            candidate = batch["generate_flag"].bool() & batch["mask"].bool()
            antigen = batch["mask_antigen"].bool() & batch["mask"].bool()
            pae_antigen = batch.get(
                "pae_supervision_antigen_mask", antigen).bool() & batch["mask"].bool()
            for index in range(batch["aa"].shape[0]):
                candidate_index = candidate[index]
                antigen_index = antigen[index]
                pae_antigen_index = pae_antigen[index]
                normalized = bool(batch["af2_pae_normalized"][index].item())
                target_pae = batch["af2_pae_matrix"][index]
                if not normalized:
                    target_pae = target_pae / 31.0
                records.append({
                    "prediction_id": batch["pdb_id"][index],
                    "construct_id": batch["construct_id"][index],
                    "scaffold_family": batch["scaffold_family"][index],
                    "protocol_id": batch["protocol_id"][index],
                    "af2_seed": int(batch["af2_seed"][index].item()),
                    "scaffold_id": int(batch["scaffold_id"][index].item()),
                    "pred_plddt": float(
                        scores["candidate_plddt_summary"][index]
                        if "candidate_plddt_summary" in scores
                        else scores["plddt"][index][candidate_index].mean()),
                    "target_plddt": float(batch["af2_plddt"][index][candidate_index].mean()),
                    "pred_iptm": float(
                        scores.get("candidate_iptm_summary", scores["iptm"])[index]),
                    "target_iptm": float(batch["af2_iptm"][index]),
                    "pred_pae": float(
                        scores["candidate_pae_summary"][index]
                        if "candidate_pae_summary" in scores
                        else scores["pae"][index][candidate_index][:, pae_antigen_index].mean()),
                    "target_pae": float(
                        target_pae[candidate_index][:, pae_antigen_index].mean()),
                    "noise_plddt": float(batch["af2_candidate_plddt_sem"][index]),
                    "noise_iptm": float(batch["af2_iptm_sem"][index]),
                    "noise_pae": float(
                        batch["af2_interface_pae_normalized_sem"][index]),
                    "replicate_std_plddt": float(
                        batch["af2_candidate_plddt_std"][index]),
                    "replicate_std_iptm": float(batch["af2_iptm_std"][index]),
                    "replicate_std_pae": float(
                        batch["af2_interface_pae_normalized_std"][index]),
                    "sample_weight": float(batch["confidence_sample_weight"][index]),
                })
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    metrics = {}
    aggregated_records = aggregate_conditions(records)
    for name in ("plddt", "iptm", "pae"):
        prediction_key = f"pred_{name}"
        target_key = f"target_{name}"
        metrics[name] = regression_metrics(
            [row[prediction_key] for row in records],
            [row[target_key] for row in records],
        )
        metrics[name]["within_group"] = grouped_metrics(
            records, prediction_key, target_key, f"noise_{name}")
        metrics[name]["condition_stability"] = pose_stability(
            records, prediction_key, target_key, f"replicate_std_{name}")
        metrics[name]["entity_aggregated"] = regression_metrics(
            [row[prediction_key] for row in aggregated_records],
            [row[target_key] for row in aggregated_records],
        )
        metrics[name]["entity_aggregated"]["within_scaffold"] = grouped_metrics(
            aggregated_records, prediction_key, target_key, f"noise_{name}")
    return {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": sha256(path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "weights_kind": checkpoint.get("weights_kind", "unknown"),
        "stored_avg_val_loss": float(checkpoint.get("avg_val_loss", math.nan)),
        "metrics": metrics,
        "records": records,
        "entity_aggregated_records": aggregated_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    parser.add_argument(
        "--checkpoint-name",
        help="Evaluate only this checkpoint filename, for example best.pt")
    parser.add_argument(
        "--dataset",
        default="data/confidence_candidate_interface_multiscaffold_dual_sem_v1/calibration.lmdb")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/confidence_candidate_interface_v1/evaluation.json")
    parser.add_argument(
        "--dataset-manifest",
        default="data/confidence_candidate_interface_multiscaffold_dual_sem_v1/manifest.json")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint_name:
        checkpoints = [checkpoint_dir / args.checkpoint_name]
        if not checkpoints[0].is_file():
            raise FileNotFoundError(checkpoints[0])
    else:
        checkpoints = sorted(
            checkpoint_dir.glob("*.pt"),
            key=lambda path: (
                path.name == "best.pt",
                int(path.stem) if path.stem.isdigit() else 0,
            ),
        )
    manifest_path = ROOT / args.dataset_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / args.dataset
    split_name = dataset_path.stem
    expected_lmdb_sha256 = manifest.get("summary", {}).get(
        split_name, {}).get("lmdb_records_sha256")
    checkpoint_heads = {
        torch.load(path, map_location="cpu", weights_only=False)
        .get("config", {}).get("model", {}).get("confidence_head_kind")
        for path in checkpoints
    }
    if checkpoint_heads & {"candidate_interface_v2", "candidate_interface_v2_2",
                           "candidate_interface_v2_3", "candidate_interface_v3"}:
        if expected_lmdb_sha256 is None:
            raise ValueError("v2 evaluation manifest does not bind the LMDB")
        actual_lmdb_sha256 = lmdb_records_sha256(dataset_path)
        if actual_lmdb_sha256 != expected_lmdb_sha256:
            raise ValueError("Evaluation LMDB does not match its manifest")
    manifest_sha256 = sha256(manifest_path)
    dataset = ConfidenceRegressionDataset({
        "db_path": str(dataset_path),
        "candidate_interface_v1": True,
        "max_residues": 0,
    })
    sampler = CompleteGroupBatchSampler(dataset, 20, shuffle=False)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=PaddingCollate(), num_workers=0)
    evaluations = [
        evaluate_checkpoint(
            path, loader, args.device,
            expected_manifest_sha256=manifest_sha256)
        for path in checkpoints
    ]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "candidate_interface_checkpoint_evaluation_v2",
        "classification": "development_only",
        "dataset_manifest_sha256": manifest_sha256,
        "evaluations": evaluations,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    summaries = [{
        "checkpoint": Path(row["checkpoint"]).name,
        "iteration": row["iteration"],
        "weights_kind": row["weights_kind"],
        "val_loss": row["stored_avg_val_loss"],
        "iptm_mae": row["metrics"]["iptm"]["mae"],
        "iptm_group_spearman": row["metrics"]["iptm"]["within_group"]["mean_group_spearman"],
        "iptm_pair_accuracy": row["metrics"]["iptm"]["within_group"]["pairwise_ordering_accuracy"],
        "iptm_group_std": row["metrics"]["iptm"]["within_group"]["mean_prediction_std"],
    } for row in evaluations]
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
