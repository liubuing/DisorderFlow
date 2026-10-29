#!/usr/bin/env python
"""Audit pairwise H3 geometry after heavy-framework alignment."""

# 本脚本审计 IDP ensemble v3 各成分构象（pose）之间的多样性：
# 将每对已接受 pose 在"重链框架残基"上做 Kabsch 叠合，分别度量
# 框架 CA-RMSD 与 H3 环 CA-RMSD；H3 RMSD 低于配置阈值视为近似重复对。
# 用于回答"候选多样性是否来自真实构象差异而非噪声"，且只做敏感性报告，
# 不做事后删除 pose（见 decision 字段）。输出多样性审计 JSON。

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_idp_ensemble_v3_candidates import featurize_arm  # noqa: E402


# Kabsch 叠合度量：将 moving 点集按质心对齐后用 SVD 求最优旋转
# （行列式为负时翻转最后一列以避免反射），分别返回
# 框架残基与 H3 残基对齐后的 CA-RMSD（单位埃）。
def kabsch_metrics(reference, moving, reference_h3, moving_h3):
    reference_center = reference.mean(axis=0)
    moving_center = moving.mean(axis=0)
    covariance = (moving - moving_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    fitted_framework = (moving - moving_center) @ rotation + reference_center
    fitted_h3 = (moving_h3 - moving_center) @ rotation + reference_center
    framework_rmsd = float(np.sqrt(np.mean(np.sum(
        (fitted_framework - reference) ** 2, axis=1
    ))))
    h3_rmsd = float(np.sqrt(np.mean(np.sum(
        (fitted_h3 - reference_h3) ** 2, axis=1
    ))))
    return framework_rmsd, h3_rmsd


# 审计主流程：输出已存在则拒绝覆盖；读取冻结 pose 与就绪清单，
# 复用候选生成侧的 featurize_arm 获得统一特征（CA 坐标、mask、design 区、
# H3 位置），框架残基定义为"非 H3 且所有 pose 均有效且均属设计区"的位点，
# 对全部 pose 两两做 Kabsch 叠合并统计近重复对，写审计 JSON。
def audit(config_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pose diversity: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    pose_path = ROOT / config["inputs"]["poses"]
    readiness_path = ROOT / config["inputs"]["readiness"]
    poses = json.loads(pose_path.read_text(encoding="ascii"))
    readiness = json.loads(readiness_path.read_text(encoding="ascii"))
    readiness_rows = {row["component_id"]: row for row in readiness["components"]}
    threshold = config["pose_diversity"]["near_duplicate_threshold_angstrom"]
    components = []
    all_h3_rmsd = []
    for component in poses["components"]:
        component_id = component["component_id"]
        pose_rows = [pose for pose in component["poses"] if pose.get("accepted") is True]
        native_h3 = readiness_rows[component_id]["h3_sequence"]
        adapter, provenance = featurize_arm(
            None, pose_rows, "diversity_audit", native_h3, torch.device("cpu")
        )
        tensors = adapter.tensors
        ca = tensors["X"][:, :, 1, :].cpu().numpy()
        mask = tensors["mask"].cpu().numpy() > 0
        design = tensors["chain_M"].cpu().numpy() > 0
        h3 = set(adapter.h3_positions)
        framework_indices = [
            index for index in range(ca.shape[1])
            if index not in h3
            and all(mask[:, index])
            and all(design[:, index])
        ]
        h3_indices = list(adapter.h3_positions)
        pairs = []
        for left, right in itertools.combinations(range(len(pose_rows)), 2):
            framework_rmsd, h3_rmsd = kabsch_metrics(
                ca[left, framework_indices], ca[right, framework_indices],
                ca[left, h3_indices], ca[right, h3_indices],
            )
            all_h3_rmsd.append(h3_rmsd)
            pairs.append({
                "left_pose": provenance["pose_ids"][left],
                "right_pose": provenance["pose_ids"][right],
                "left_source": pose_rows[left]["source"],
                "right_source": pose_rows[right]["source"],
                "framework_ca_rmsd_angstrom": round(framework_rmsd, 6),
                "h3_ca_rmsd_angstrom": round(h3_rmsd, 6),
                "near_duplicate": h3_rmsd < threshold,
            })
        components.append({
            "component_id": component_id,
            "pose_count": len(pose_rows),
            "pair_count": len(pairs),
            "near_duplicate_pair_count": sum(row["near_duplicate"] for row in pairs),
            "minimum_h3_ca_rmsd_angstrom": min(
                row["h3_ca_rmsd_angstrom"] for row in pairs
            ),
            "median_h3_ca_rmsd_angstrom": statistics.median(
                row["h3_ca_rmsd_angstrom"] for row in pairs
            ),
            "maximum_h3_ca_rmsd_angstrom": max(
                row["h3_ca_rmsd_angstrom"] for row in pairs
            ),
            "pairs": pairs,
        })
    payload = {
        "schema_version": 1,
        "status": "pose_diversity_audit_complete",
        "classification": config["classification"],
        "config": str(config_path),
        "near_duplicate_threshold_angstrom": threshold,
        "component_count": len(components),
        "component_with_near_duplicates_count": sum(
            row["near_duplicate_pair_count"] > 0 for row in components
        ),
        "global_median_h3_ca_rmsd_angstrom": statistics.median(all_h3_rmsd),
        "components": components,
        "decision": "report_sensitivity_without_posthoc_pose_removal",
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


# 主入口：读取配置并驱动 audit()，打印近重复成分与全局中位 H3 RMSD。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_v3_computational_robustness.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = audit(config_path, ROOT / config["outputs"]["pose_diversity"])
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "components_with_near_duplicates": result[
            "component_with_near_duplicates_count"
        ],
        "global_median_h3_ca_rmsd_angstrom": result[
            "global_median_h3_ca_rmsd_angstrom"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
