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
    # 计算数据集清单的 SHA-256，登记进输出以保证诊断结果可溯源。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def composition(aa, mask, classes=20):
    # 掩码区域的 20 类氨基酸组成频率（bincount 归一化）；空区域返回全零。
    selected = aa[mask].long()
    counts = torch.bincount(selected.clamp(0, classes - 1), minlength=classes)
    return (counts.float() / max(selected.numel(), 1)).numpy()


def record_features(batch):
    # 每条记录的序数特征: 候选区/抗原区氨基酸组成（各 20 维）+ 10 维几何摘要
    # （两侧残基数与候选-抗原 CA 最小距离的均值/标准差/四分位数，以及
    # <6/<8/<10 埃的比例——6 埃内视为直接接触，8/10 埃为宽松阈值）。
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
    # 回归目标三元组: 候选区平均 pLDDT、ipTM、候选-抗原界面平均 PAE。
    # PAE 单位为埃、越低越好；未归一化时除以 31.0（AF2 PAE 的实际分辨率
    # 上限）映射到近 0-1，与 pLDDT/ipTM 的量纲对齐。
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
    # 平均秩（并列取平均）：mergesort 保证稳定；与 scipy.stats.rankdata 一致。
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
    # 评估指标: MAE、预测/目标标准差之比（<1 表示过度收缩）、方差比，
    # 以及"支架内 Spearman 中位数"——只在每个支架组内部计算秩相关再取
    # 中位数，衡量模型对同支架候选的排序能力（跨支架分数不可比时这
    # 才是公平口径）。零方差组（秩相关无定义）被剔除并计数。
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
    # 按 construct_id 聚合实体: 特征与目标在 6 个 AF2 条件（2 模型 x 3 种子）
    # 上取均值，组标签取支架家族——交叉验证按支架分组（GroupKFold），
    # 防止同一支架的实体同时出现在训练/验证两侧而高估泛化。
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
    # 只用训练数据拟合两个朴素基线（5 折 GroupKFold 下的折外预测）:
    #   - fold_train_mean: 训练折的目标均值（无特征基线）;
    #   - ridge: 标准化 + Ridge(alpha=1.0 默认) 的线性回归。
    # 作用是为更复杂模型提供"必须显著超过"的下限参照。
    # --evaluation-dataset 可选: 用全部训练实体重新拟合后在独立开发集上评估。
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
