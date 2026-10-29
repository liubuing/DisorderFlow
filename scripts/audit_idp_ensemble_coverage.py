#!/usr/bin/env python
"""Audit whether pose panels extend beyond local native-neighborhood replicas."""

# 目的：审计 IDP 集成姿态面板的覆盖度是否超出"本地近邻构象复制"——即姿态是否
# 具有真实的多来源与多样性。方法：抗体主链叠合后计算抗原界面的 CA-RMSD，
# 近重复对与坐标哈希相同的姿态并入并查集得到"有效姿态数"，再逐项检查配置要求。
# 输入：YAML 要求配置 + 姿态清单 JSON；输出：逐组分 coverage_ready 判定的 JSON。

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ca_coordinates(path, chain_ids):
    model = PDBParser(QUIET=True).get_structure(path.stem, str(path))[0]
    return np.asarray([
        residue["CA"].coord for chain_id in chain_ids for residue in model[chain_id]
        if "CA" in residue
    ], dtype=float)


def kabsch_rmsd(reference, mobile):
    # Kabsch 算法：SVD 求最优旋转后计算 CA-RMSD；检测旋转矩阵行列式为负时
    # 翻转最后一个奇异向量以排除镜像解。少于 3 个原子无法定义刚体叠合，返回 None。
    if reference.shape != mobile.shape or len(reference) < 3:
        return None
    ref = reference - reference.mean(axis=0)
    mov = mobile - mobile.mean(axis=0)
    left, _, right = np.linalg.svd(mov.T @ ref)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    fitted = mov @ rotation
    return float(np.sqrt(np.mean(np.sum((fitted - ref) ** 2, axis=1))))


def antibody_aligned_antigen_rmsd(ref_antibody, mob_antibody, ref_antigen, mob_antigen):
    # 以抗体为参照做刚体叠合，再把变换应用到抗原上，度量"界面姿态"差异：
    # 抗体内部差异被消除，剩余差异反映抗原相对抗体的摆放（对接姿态）而非整体折叠。
    if (ref_antibody.shape != mob_antibody.shape
            or ref_antigen.shape != mob_antigen.shape
            or len(ref_antibody) < 3 or len(ref_antigen) < 1):
        return None
    ref_center = ref_antibody.mean(axis=0)
    mob_center = mob_antibody.mean(axis=0)
    ref = ref_antibody - ref_center
    mob = mob_antibody - mob_center
    left, _, right = np.linalg.svd(mob.T @ ref)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    fitted_antigen = (mob_antigen - mob_center) @ rotation + ref_center
    return float(np.sqrt(np.mean(np.sum(
        (fitted_antigen - ref_antigen) ** 2, axis=1))))


def effective_pose_count(pose_ids, coordinate_hashes, duplicate_pairs):
    # 用并查集把坐标完全相同（哈希一致）与界面近重复的姿态合并，
    # 返回不重复等价类的个数，即"有效姿态数"，防止同一构象被重复计数虚增多样性。
    parent = {pose_id: pose_id for pose_id in pose_ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_hash = {}
    for pose_id, digest in zip(pose_ids, coordinate_hashes, strict=True):
        if digest in by_hash:
            union(pose_id, by_hash[digest])
        else:
            by_hash[digest] = pose_id
    for left, right in duplicate_pairs:
        union(left, right)
    return len({find(pose_id) for pose_id in pose_ids})


def audit(config_path, manifest_path, output_path):
    # 主审计：逐组分计算两两界面 RMSD，标记低于近重复阈值的姿态对，
    # 再对照配置要求（最小原始/有效姿态数、来源类别数、非本地姿态数、中位 RMSD 下限）
    # 生成 coverage_ready 布尔判定。
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    requirements = config["requirements"]
    classes = config["source_classification"]
    components = []
    for component in manifest["components"]:
        poses = [row for row in component["poses"] if row.get("accepted", True)]
        coordinate_hashes = [row.get("coordinate_sha256") or sha256(ROOT / row["path"])
                             for row in poses]
        source_classes = [classes.get(row["source"], "unclassified") for row in poses]
        pair_rmsd = []
        near_duplicates = []
        antibody_coordinates = [
            ca_coordinates(ROOT / row["path"], row["antibody_chains"])
            for row in poses]
        antigen_coordinates = [
            ca_coordinates(ROOT / row["path"], row["antigen_chains"])
            for row in poses]
        for left in range(len(poses)):
            for right in range(left + 1, len(poses)):
                value = antibody_aligned_antigen_rmsd(
                    antibody_coordinates[left], antibody_coordinates[right],
                    antigen_coordinates[left], antigen_coordinates[right])
                if value is not None:
                    pair_rmsd.append(value)
                    if value < float(requirements[
                            "near_duplicate_interface_ca_rmsd_angstrom"]):
                        near_duplicates.append([poses[left]["pose_id"], poses[right]["pose_id"]])
        effective = effective_pose_count(
            [row["pose_id"] for row in poses], coordinate_hashes, near_duplicates)
        median_rmsd = float(np.median(pair_rmsd)) if pair_rmsd else 0.0
        source_set = set(source_classes)
        nonlocal_count = sum(
            value in {"experimental", "nmr_or_deposited_model", "predicted"}
            for value in source_classes)
        checks = {
            "minimum_raw_poses": len(poses) >= int(requirements["minimum_raw_poses"]),
            "minimum_effective_poses": effective >= int(
                requirements["minimum_effective_poses"]),
            "minimum_source_classes": len(source_set) >= int(
                requirements["minimum_source_classes"]),
            "minimum_nonlocal_poses": nonlocal_count >= int(
                requirements["minimum_nonlocal_poses"]),
            "source_class_requirement": bool(
                source_set & set(requirements["required_source_classes_any"])),
            "median_h3_ca_rmsd": median_rmsd >= float(
                requirements["minimum_median_interface_ca_rmsd_angstrom"]),
            "not_local_sampling_only": source_set != {"local_sampling"},
        }
        components.append({
            "component_id": component["component_id"],
            "raw_pose_count": len(poses),
            "effective_pose_count": effective,
            "source_classes": sorted(source_set),
            "median_antibody_aligned_antigen_ca_rmsd_angstrom": median_rmsd,
            "near_duplicate_pairs": near_duplicates,
            "checks": checks,
            "coverage_ready": all(checks.values()),
        })
    payload = {
        "schema_version": 1,
        "status": "idp_ensemble_coverage_audit_complete",
        "classification": config["classification"],
        "config_sha256": sha256(config_path),
        "manifest_sha256": sha256(manifest_path),
        "component_count": len(components),
        "coverage_ready_count": sum(row["coverage_ready"] for row in components),
        "coverage_blocked_count": sum(not row["coverage_ready"] for row in components),
        "components": components,
        "decision": (
            "use_only_coverage_ready_components_for_broad_ensemble_development"),
        "claim_boundary": config["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/benchmarks/idp_ensemble_coverage_v1.yml"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "reviewer_outputs/idp_ensemble_expanded_development_v2/merged_pose_manifest_v2.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    output = ROOT / (args.output or Path(config["output"]))
    result = audit(config_path, ROOT / args.manifest, output)
    print(json.dumps({
        "status": result["status"], "components": result["component_count"],
        "coverage_ready": result["coverage_ready_count"],
        "coverage_blocked": result["coverage_blocked_count"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
