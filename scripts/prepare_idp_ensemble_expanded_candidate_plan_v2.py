#!/usr/bin/env python
"""Freeze the expanded-IDP development candidate-generation plan."""

# 本脚本在生成任何候选之前冻结扩展 IDP 开发的候选生成计划：
# 校验结构面板与 pose 面板均已就绪且成分集合一致后，固化生成预算——
# 每成分两臂（ensemble / single_state）、每臂 4 条候选、种子
# [12001, 12011, 12021, 12031]、可变区仅重链 H3、替换桶 2/4/6/8
# （每桶 1 条，H3 最大替换数 8）、每成分 1 条天然对照。
# future_confirmation_eligible_count 固定为 0（仅回顾性开发用途）。

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    # 计算文件 SHA-256，记录两个上游面板的冻结指纹
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 计划冻结主流程：输出拒绝覆盖 → 校验 pose 面板状态与成分集合一致 →
# 写出含完整预算/种子/突变空间/声明边界的冻结计划
def prepare(structural_panel, pose_status, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate plan: {output}")
    panel = json.loads(structural_panel.read_text(encoding="ascii"))
    poses = json.loads(pose_status.read_text(encoding="ascii"))
    if poses.get("status") != "pose_panel_ready":
        raise RuntimeError("Expanded pose panel is not ready")
    component_ids = [row["component_id"] for row in panel["components"]]
    if set(component_ids) != set(poses["components"]):
        raise RuntimeError("Structural and pose panels contain different components")
    payload = {
        "schema_version": 1,
        "status": "expanded_candidate_plan_frozen_before_generation",
        "classification": "retrospective_expanded_idp_candidate_development",
        "structural_panel": str(structural_panel),
        "structural_panel_sha256": sha256(structural_panel),
        "pose_status": str(pose_status),
        "pose_status_sha256": sha256(pose_status),
        "components": [
            {
                "component_id": row["component_id"],
                "target": row["target"],
                "antibody_lineage": row["lineage_proxy"],
            }
            for row in panel["components"]
        ],
        "arms": ["ensemble", "single_state"],
        "seeds": [12001, 12011, 12021, 12031],
        "candidates_per_component_arm": 4,
        "native_controls_per_component": 1,
        "mutation_space": {
            "mutable_regions": ["heavy_chain_h3"],
            "substitution_buckets": [2, 4, 6, 8],
            "maximum_h3_substitutions": 8,
            "candidate_count_per_bucket": 1,
        },
        "future_confirmation_eligible_count": 0,
        "claim_boundary": (
            "Candidate-generation plan on exposed retrospective expanded-IDP "
            "development structures only; no final confirmation or binding claim"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    # 主入口：三个路径均有默认值，打印计划摘要与总候选预算
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-panel", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "structural_panel.json"
        ),
    )
    parser.add_argument(
        "--pose-status", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "pose_status.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "candidate_plan.json"
        ),
    )
    args = parser.parse_args()
    result = prepare(
        ROOT / args.structural_panel, ROOT / args.pose_status, ROOT / args.output
    )
    print(json.dumps({
        "status": result["status"],
        "components": len(result["components"]),
        "candidate_budget": (
            len(result["components"]) * 2
            * result["candidates_per_component_arm"]
        ),
        "native_controls": len(result["components"]),
        "future_confirmation_eligible": result[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
