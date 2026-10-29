#!/usr/bin/env python
"""Evaluate fixed-candidate stability after omitting each pose in turn."""

# IDP ensemble v3 的"留一 pose"稳健性分析：候选序列集合固定不变，
# 每次从骨架 pose 面板中移除一个构象，重新训练 prefix adapter 并给
# 同一批候选打分，检验打分排序与关键结论(ensemble vs single-state 的
# 方向)是否对 pose 面板的构成敏感。
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROTEIN_MPNN = ROOT / "ProteinMPNN"
# 将仓库根目录与 ProteinMPNN 源码目录都加入 sys.path，
# 以便导入仓库内脚本和 ProteinMPNN 的模型实现。
for entry in (str(ROOT), str(PROTEIN_MPNN)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts.generate_idp_ensemble_v3_candidates import featurize_arm  # noqa: E402
from scripts.protein_mpnn_prefix_adapter import build_model  # noqa: E402
from scripts.score_idp_ensemble_v3_source_sensitivity import (  # noqa: E402
    score_sequence,
)


def direction(value):
    # 只保留符号：1 / -1 / 0，用于判断差值的定性方向是否稳定。
    return 1 if value > 0 else -1 if value < 0 else 0


def analyze(config_path, source_sensitivity_path, output):
    # 核心分析流程（禁止覆盖既有结果，保证产物不可变、可审计）。
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite leave-one-pose-out: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    candidates = json.loads(
        (ROOT / config["inputs"]["candidates"]).read_text(encoding="ascii")
    )
    poses = json.loads(
        (ROOT / config["inputs"]["poses"]).read_text(encoding="ascii")
    )
    source = json.loads(source_sensitivity_path.read_text(encoding="ascii"))
    # 基线打分来自 source_sensitivity 结果中 mixed(混合来源)面板的记录，
    # 以 (component_id, candidate_id) 为键。
    baseline = {
        (row["component_id"], row["candidate_id"]): row["mean_h3_log_probability"]
        for row in source["records"]
        if row["source_panel"] == "mixed"
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(ROOT / config["inputs"]["checkpoint"], device=device)
    pose_rows = {row["component_id"]: row["poses"] for row in poses["components"]}
    component_results = []
    all_rank_correlations = []
    all_max_deltas = []
    sign_stable_pairs = 0
    total_pairs = 0
    for component_id, component in candidates["components"].items():
        native = component["native_h3"]
        # 把各实验臂(arm)的候选展平为固定候选集合，并补上候选 id 与来源臂。
        fixed = [
            {
                "candidate_id": f"{arm_name}_{row['substitution_bucket']}",
                "origin_arm": arm_name,
                **row,
            }
            for arm_name, arm in component["arms"].items()
            for row in arm["candidates"]
        ]
        full_values = [baseline[(component_id, row["candidate_id"])] for row in fixed]
        omission_results = []
        values_by_omission = {}
        for omitted_index, omitted in enumerate(pose_rows[component_id]):
            # 留一：仅保留除被移除 pose 之外的面板，重新训练 adapter 后打分。
            retained = [
                row for index, row in enumerate(pose_rows[component_id])
                if index != omitted_index
            ]
            adapter, _ = featurize_arm(
                model, retained, "leave_one_pose_out", native, device
            )
            values = {
                row["candidate_id"]: score_sequence(adapter, row["sequence"])
                for row in fixed
            }
            values_by_omission[omitted["pose_id"]] = values
            ordered = [values[row["candidate_id"]] for row in fixed]
            # 稳健性指标 1：候选打分排序与完整 mixed 面板的 Spearman 相关；
            # 指标 2：相对基线的最大绝对分值漂移。
            correlation = float(spearmanr(full_values, ordered).statistic)
            max_delta = max(
                abs(values[row["candidate_id"]] - baseline[(component_id, row["candidate_id"])])
                for row in fixed
            )
            all_rank_correlations.append(correlation)
            all_max_deltas.append(max_delta)
            omission_results.append({
                "omitted_pose_id": omitted["pose_id"],
                "omitted_source": omitted["source"],
                "retained_pose_count": len(retained),
                "spearman_vs_full_mixed": correlation,
                "maximum_absolute_score_delta": max_delta,
                "candidate_scores": values,
            })
        # 结论稳定性：对每个替换桶(substitution bucket)检查
        # ensemble-single 差值方向在所有留一情形下是否与完整面板一致。
        pair_stability = []
        for bucket in component["substitution_buckets"]:
            ensemble_id = f"ensemble_{bucket}"
            single_id = f"single_state_{bucket}"
            full_difference = (
                baseline[(component_id, ensemble_id)]
                - baseline[(component_id, single_id)]
            )
            omission_directions = {
                pose_id: direction(values[ensemble_id] - values[single_id])
                for pose_id, values in values_by_omission.items()
            }
            stable = all(
                value == direction(full_difference)
                for value in omission_directions.values()
            )
            sign_stable_pairs += stable
            total_pairs += 1
            pair_stability.append({
                "substitution_bucket": bucket,
                "full_mixed_difference": full_difference,
                "full_direction": direction(full_difference),
                "omission_directions": omission_directions,
                "direction_stable_across_all_omissions": stable,
            })
        component_results.append({
            "component_id": component_id,
            "pose_count": len(pose_rows[component_id]),
            "minimum_spearman_vs_full_mixed": min(
                row["spearman_vs_full_mixed"] for row in omission_results
            ),
            "maximum_absolute_score_delta": max(
                row["maximum_absolute_score_delta"] for row in omission_results
            ),
            "stable_bucket_pair_count": sum(
                row["direction_stable_across_all_omissions"]
                for row in pair_stability
            ),
            "omissions": omission_results,
            "matched_bucket_stability": pair_stability,
        })
    # 汇总输出。decision 字段明确结论边界：只报告 pose 影响大小，
    # 不支持"事后剔除 pose"式的事后修正声明。
    payload = {
        "schema_version": 1,
        "status": "leave_one_pose_out_complete",
        "classification": config["classification"],
        "config": str(config_path),
        "source_sensitivity": str(source_sensitivity_path),
        "device": str(device),
        "component_count": len(component_results),
        "omission_panel_count": len(all_rank_correlations),
        "matched_bucket_pair_count": total_pairs,
        "direction_stable_pair_count": sign_stable_pairs,
        "minimum_rank_correlation": min(all_rank_correlations),
        "median_rank_correlation": statistics.median(all_rank_correlations),
        "maximum_absolute_score_delta": max(all_max_deltas),
        "median_maximum_absolute_score_delta": statistics.median(all_max_deltas),
        "components": component_results,
        "decision": "report_pose_influence_without_posthoc_pose_removal",
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_v3_computational_robustness.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    # 输入为 source_sensitivity 的既有产物；本脚本产出 leave_one_pose_out 结果。
    source_path = ROOT / config["outputs"]["source_sensitivity"]
    output = ROOT / config["outputs"]["leave_one_pose_out"]
    result = analyze(config_path, source_path, output)
    print(json.dumps({
        "status": result["status"],
        "omission_panels": result["omission_panel_count"],
        "stable_pairs": result["direction_stable_pair_count"],
        "total_pairs": result["matched_bucket_pair_count"],
        "minimum_rank_correlation": result["minimum_rank_correlation"],
        "median_rank_correlation": result["median_rank_correlation"],
        "maximum_absolute_score_delta": result["maximum_absolute_score_delta"],
    }, indent=2))


if __name__ == "__main__":
    main()
