#!/usr/bin/env python
"""Validate and materialize the pre-candidate prospective pilot contract."""

# 本脚本在校验后物化"候选访问之前"的前瞻性先导（pilot）合同：
# 校验配置未被 superseded、成分注册表与 pose 清单哈希匹配、
# 干净成分数与冻结期望一致、候选预算自洽（成分数 × 种子数 × 每臂每种子数），
# 随后写出带双臂（PoE / single_state）、预算与主要终点的候选清单。
# 清单与配置一经写出即拒绝覆盖，保证合同先于任何候选生成被锁定。

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    # 计算文件 SHA-256，用于配置/注册表/pose 清单的哈希校验
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path):
    # 显示为相对项目根的路径；位于根外时回退绝对路径
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# 合同校验与物化主流程：哈希核对 → 排除 sensitivity_only 成分 →
# 预算一致性检查 → 写候选清单（已存在则拒绝覆盖）
def prepare(config_path):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    if str(config.get("status", "")).startswith("superseded"):
        raise ValueError("Refusing to materialize a superseded pilot contract")
    registry_item = config["component_registry"]
    registry_path = ROOT / registry_item["path"]
    if sha256(registry_path) != registry_item["sha256"]:
        raise ValueError("Component registry hash mismatch")
    registry = yaml.safe_load(registry_path.read_text(encoding="ascii"))
    pose_item = config["pose_manifest"]
    pose_path = ROOT / pose_item["path"]
    if sha256(pose_path) != pose_item["sha256"]:
        raise ValueError("Pose manifest hash mismatch")
    pose_manifest = json.loads(pose_path.read_text(encoding="ascii"))
    pose_by_id = {
        row["component_id"]: row for row in pose_manifest["components"]
    }
    components = [
        row for row in registry["components"] if not row.get("sensitivity_only", False)
    ]
    if len(components) != config["component_registry"]["expected_components"]:
        raise ValueError("Clean component count does not match frozen pilot contract")
    if config["matched_budget"]["total_generated_candidates_per_arm"] != (
        len(components)
        * len(config["matched_budget"]["seeds"])
        * config["matched_budget"]["candidates_per_component_arm_seed"]
    ):
        raise ValueError("Frozen candidate budget is inconsistent")
    manifest = {
        "schema_version": 1,
        "status": "contract_validated_before_candidate_access",
        "classification": config["classification"],
        "config": display_path(config_path),
        "config_sha256": sha256(config_path),
        "component_registry": registry_item,
        "components": [
            {
                "component_id": row["component_id"],
                "target": row["target"],
                "epitope_cluster": row["epitope_cluster"],
                "native_h3": pose_by_id[row["component_id"]]["native_h3"],
                "pose_count": len(pose_by_id[row["component_id"]]["pose_paths"]),
                "arms": ["ensemble_product_of_experts", "single_state"],
            }
            for row in components
        ],
        "budget": config["matched_budget"],
        "primary_endpoint": config["primary_endpoint"],
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / config["outputs"]["candidate_manifest"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot manifest: {output}")
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return manifest


def main():
    # 主入口：默认使用 v1 前瞻性先导配置，打印物化后的合同清单
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_prospective_pilot_v1.yml"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT / args.config), indent=2))


if __name__ == "__main__":
    main()
