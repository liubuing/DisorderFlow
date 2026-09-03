#!/usr/bin/env python
"""Evaluate train-only scaffold-grouped mean and ridge baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def composition(aa, mask, classes=20):
    selected = aa[mask].long()
    counts = torch.bincount(selected.clamp(0, classes - 1), minlength=classes)
    return (counts.float() / max(selected.numel(), 1)).numpy()


def record_features(batch):
    candidate = batch["generate_flag"].bool()
    antigen = batch["pae_supervision_antigen_mask"].bool()
    aa = batch["aa"]
    candidate_ca = batch["pos_heavyatom"][candidate, 1]
    antigen_ca = batch["pos_heavyatom"][antigen, 1]
    distances = torch.cdist(candidate_ca, antigen_ca)
    minimum = distances.min(dim=1).values
    geometry = torch.tensor([
        candidate.sum(),
        antigen.sum(),
        minimum.mean(),
        minimum.std(unbiased=False),
        torch.quantile(minimum, 0.25),
        torch.quantile(minimum, 0.50),
        torch.quantile(minimum, 0.75),
        (minimum < 6.0).float().mean(),
        (minimum < 8.0).float().mean(),
        (minimum < 10.0).float().mean(),
    ]).float().numpy()
    return np.concatenate([
        composition(aa, candidate),
        composition(aa, antigen),
        geometry,
    ])


def record_targets(batch):
    candidate = batch["generate_flag"].bool()
    antigen = batch["pae_supervision_antigen_mask"].bool()
    interface = candidate.unsqueeze(1) & antigen.unsqueeze(0)
    pae = batch["af2_pae_matrix"].float()
    if not bool(batch["af2_pae_normalized"]):
        pae = pae / 31.0
    return np.asarray([
        batch["af2_plddt"].float()[candidate].mean().item(),
        batch["af2_iptm"].float().item(),
        pae[interface].mean().item(),
    ])


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(left, right):
    left, right = rankdata(left), rankdata(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metrics(prediction, target, groups):
    group_correlations = []
    for group in sorted(set(groups)):
        mask = groups == group
        value = spearman(prediction[mask], target[mask])
        if value is not None:
            group_correlations.append(value)
    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))
    std_ratio = prediction_std / target_std if target_std > 0 else None
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "prediction_std": prediction_std,
        "target_std": target_std,
        "std_ratio": std_ratio,
        "variance_ratio": std_ratio ** 2 if std_ratio is not None else None,
        "median_scaffold_spearman": (
            float(np.median(group_correlations)) if group_correlations else None),
        "scaffolds_with_defined_spearman": len(group_correlations),
    }


def load_entities(dataset):
    entities = defaultdict(lambda: {"features": [], "targets": [], "scaffold": None})
    for index in range(len(dataset)):
        batch = dataset[index]
        entity = entities[batch["construct_id"]]
        entity["features"].append(record_features(batch))
        entity["targets"].append(record_targets(batch))
        entity["scaffold"] = batch["scaffold_family"]
    ids = sorted(entities)
    features = np.stack([
        np.mean(entities[entity_id]["features"], axis=0) for entity_id in ids])
    targets = np.stack([
        np.mean(entities[entity_id]["targets"], axis=0) for entity_id in ids])
    groups = np.asarray([entities[entity_id]["scaffold"] for entity_id in ids])
    return ids, features, targets, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/confidence_candidate_interface_multiscaffold_dual_sem_v2_final6/train.lmdb")
    parser.add_argument(
        "--dataset-manifest",
        default="data/confidence_candidate_interface_multiscaffold_dual_sem_v2_final6/manifest.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--evaluation-dataset",
        help="Optional separate development dataset evaluated after fitting all train entities")
    parser.add_argument(
        "--output",
        default="results/confidence_candidate_interface_v2_1/grouped_baselines.json")
    args = parser.parse_args()

    dataset = ConfidenceRegressionDataset({
        "db_path": str(ROOT / args.dataset),
        "candidate_interface_v1": True,
        "max_residues": 0,
    })
    try:
        ids, features, targets, groups = load_entities(dataset)
    finally:
        dataset.close()

    splitter = GroupKFold(n_splits=args.folds)
    mean_prediction = np.zeros_like(targets)
    ridge_prediction = np.zeros_like(targets)
    folds = []
    for fold, (train_indices, validation_indices) in enumerate(
            splitter.split(features, targets, groups), start=1):
        mean_prediction[validation_indices] = targets[train_indices].mean(axis=0)
        for output_index in range(targets.shape[1]):
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=args.ridge_alpha))
            model.fit(features[train_indices], targets[train_indices, output_index])
            ridge_prediction[validation_indices, output_index] = model.predict(
                features[validation_indices])
        folds.append({
            "fold": fold,
            "train_scaffolds": sorted(set(groups[train_indices])),
            "validation_scaffolds": sorted(set(groups[validation_indices])),
        })

    names = ("plddt", "iptm", "pae")
    output = {
        "schema_version": "candidate_interface_grouped_baselines_v1",
        "classification": "train_only_development_diagnostic",
        "dataset_manifest_sha256": sha256(ROOT / args.dataset_manifest),
        "entities": len(ids),
        "scaffolds": len(set(groups)),
        "folds": folds,
        "feature_contract": (
            "candidate/antigen composition and length plus candidate-to-antigen "
            "CA minimum-distance summaries, averaged over six AF2 conditions"),
        "ridge_alpha": args.ridge_alpha,
        "metrics": {},
    }
    for model_name, prediction in (
            ("fold_train_mean", mean_prediction), ("ridge", ridge_prediction)):
        output["metrics"][model_name] = {
            name: metrics(prediction[:, index], targets[:, index], groups)
            for index, name in enumerate(names)
        }
    if args.evaluation_dataset:
        evaluation_dataset = ConfidenceRegressionDataset({
            "db_path": str(ROOT / args.evaluation_dataset),
            "candidate_interface_v1": True,
            "max_residues": 0,
        })
        try:
            _, evaluation_features, evaluation_targets, evaluation_groups = (
                load_entities(evaluation_dataset))
        finally:
            evaluation_dataset.close()
        separate_mean = np.broadcast_to(
            targets.mean(axis=0), evaluation_targets.shape)
        separate_ridge = np.zeros_like(evaluation_targets)
        for output_index in range(targets.shape[1]):
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=args.ridge_alpha))
            model.fit(features, targets[:, output_index])
            separate_ridge[:, output_index] = model.predict(evaluation_features)
        output["separate_evaluation"] = {
            "dataset": args.evaluation_dataset,
            "metrics": {},
        }
        for model_name, prediction in (
                ("train_mean", separate_mean), ("ridge", separate_ridge)):
            output["separate_evaluation"]["metrics"][model_name] = {
                name: metrics(
                    prediction[:, index], evaluation_targets[:, index],
                    evaluation_groups)
                for index, name in enumerate(names)
            }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output["metrics"], indent=2))
    if "separate_evaluation" in output:
        print(json.dumps(output["separate_evaluation"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
