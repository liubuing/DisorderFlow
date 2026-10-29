#!/usr/bin/env python
"""Score matched-generator H3 pools by leave-one-conformer-out selection."""

# 匹配生成器(matched-generation)候选 H3 池的打分基准：
# 对每个"组分 x 方法"的候选池，在多个保留构象(held-out conformer)上做
# leave-one-conformer-out 评估——用其余构象选出的 ensemble 候选，
# 到被留出的构象上打分，与逐构象挑选(single-state)及天然 H3 对比，
# 量化"多构象集成选择"是否带来真实的界面化学打分增益。
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 依赖仓库内的 H3 界面接触提取与序列化学打分模块。
from modules.h3_interface_contacts import extract_h3_interface, score_h3_sequence  # noqa: E402


def sha256(path):
    # 文件 sha256：用于冻结输入校验与 pose 溯源。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def robust_value(values, aggregation):
    # 稳健聚合打分：p25 分位数、最小值、均值的加权和，再减去标准差惩罚。
    # 权重全部来自冻结配置；惩罚项抑制在部分构象上表现不稳定的候选。
    values = np.asarray(values, dtype=np.float64)
    return float(
        float(aggregation["p25_weight"]) * np.percentile(values, 25)
        + float(aggregation["minimum_weight"]) * values.min()
        + float(aggregation["mean_weight"]) * values.mean()
        - float(aggregation["standard_deviation_penalty"]) * values.std()
    )


def bootstrap_mean(values, trials, seed):
    # 对组分级效应做 bootstrap，返回均值的 95% 置信区间 [2.5%, 97.5%] 分位。
    # 无数据时返回 None。
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = generator.choice(array, size=(trials, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def exact_sign_flip_p(values):
    # 精确符号翻转检验：穷举全部 +/-1 符号组合，计算 |均值| 不小于
    # 观测值的组合比例（双侧面精确 p 值）。只适用于小样本(n 个值 -> 2^n 组合)。
    # 1e-12 容差避免浮点误差导致漏计相等情形。
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    estimates = [
        abs(float(np.mean(array * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ]
    return float(np.mean(np.asarray(estimates) >= observed - 1e-12))


def candidate_pools(raw_generation, methods):
    # 从匹配生成结果中汇集候选池：只保留 status=success 且方法在指定集合内
    # 的 slot；按 (component_id, method) 分组并对序列去重（保持首次出现顺序）。
    grouped = defaultdict(list)
    for slot in raw_generation["slots"]:
        if slot["status"] != "success" or slot["method"] not in methods:
            continue
        grouped[(slot["component_id"], slot["method"])].extend(slot["candidates"])
    pools = {}
    for key, rows in grouped.items():
        sequences = []
        seen = set()
        for row in rows:
            sequence = row["sequence"]
            if sequence not in seen:
                seen.add(sequence)
                sequences.append(sequence)
        pools[key] = sequences
    return pools


def pose_rows_by_component(components, abeta_audit, nonabeta_audit):
    # 按组分收集通过审计的 pose 行：
    # - amyloid_beta 组分：匹配 reference_pdb 且 status=pass 的 abeta 审计条目；
    # - 其他组分：还要求 selected_for_panel 为真（非 abeta 面板筛选更严格）。
    # 结果按 conformer 标识排序，保证打分矩阵列顺序确定。
    output = {}
    for component in components:
        if component["target"] == "amyloid_beta":
            rows = [
                row for row in abeta_audit["entries"]
                if row["reference_pdb"] == component["reference_pdb"] and row["status"] == "pass"
            ]
        else:
            rows = [
                row for row in nonabeta_audit["entries"]
                if row["reference_id"] == component["reference_pdb"]
                and row["status"] == "pass"
                and row.get("selected_for_panel")
            ]
        output[component["component_id"]] = sorted(rows, key=lambda row: str(row["conformer"]))
    return output


def score_matrix(component, pose_rows, sequences, scoring):
    # 对单个组分构建"序列 x 构象"的化学打分矩阵：
    # 行为候选序列，列为各 pose 的 H3 界面接触特征打分(chemistry_score)。
    # 同时给出天然 H3 在各 pose 上的参照打分 native，以及 pose 溯源信息。
    interfaces = [
        extract_h3_interface(
            str(ROOT / row["pose_pdb"]),
            component["heavy_chain"],
            "P",
            light_chain=component["light_chain"],
            expected_h3_sequence=component["native_h3"],
            contact_cutoff=float(scoring["contact_cutoff_angstrom"]),
            sidechain_cutoff=float(scoring["sidechain_cutoff_angstrom"]),
        )
        for row in pose_rows
    ]
    matrix = np.asarray([
        [score_h3_sequence(sequence, interface)["chemistry_score"] for interface in interfaces]
        for sequence in sequences
    ], dtype=np.float64)
    native = np.asarray([
        score_h3_sequence(component["native_h3"], interface)["chemistry_score"]
        for interface in interfaces
    ], dtype=np.float64)
    return matrix, native, [
        {
            "conformer": row["conformer"],
            "pose_pdb": row["pose_pdb"],
            "pose_sha256": sha256(ROOT / row["pose_pdb"]),
            "h3_geometry_contacts": len(interface["contacts"]),
            "h3_sidechain_contacts": sum(
                contact.min_sidechain_distance is not None for contact in interface["contacts"]
            ),
        }
        for row, interface in zip(pose_rows, interfaces, strict=True)
    ]


def leave_one_out(component_id, method, sequences, matrix, native, aggregation):
    # 留一构象评估：对每一列(holdout)，
    # 1) 仅用其余列按 robust_value 选出 ensemble 候选；
    # 2) single-state 基线为其余各列逐列 argmax 候选在 holdout 上的得分中位数；
    # 3) 与 holdout 上的天然 H3 打分对比。
    # 平局时用序列字符串作次级键，保证选择结果确定性。
    rows = []
    for holdout in range(matrix.shape[1]):
        training = [index for index in range(matrix.shape[1]) if index != holdout]
        ensemble_index = max(
            range(len(sequences)),
            key=lambda index: (robust_value(matrix[index, training], aggregation), sequences[index]),
        )
        single_indices = [int(np.argmax(matrix[:, conformer])) for conformer in training]
        ensemble_score = float(matrix[ensemble_index, holdout])
        single_scores = [float(matrix[index, holdout]) for index in single_indices]
        single_score = float(np.median(single_scores))
        native_score = float(native[holdout])
        rows.append({
            "component_id": component_id,
            "method": method,
            "heldout_conformer_index": holdout,
            "candidate_pool_size": len(sequences),
            "ensemble_candidate": sequences[ensemble_index],
            "ensemble_score": ensemble_score,
            "single_state_score": single_score,
            "native_score": native_score,
            "ensemble_minus_single_state": ensemble_score - single_score,
            "ensemble_minus_native": ensemble_score - native_score,
        })
    return rows


def summarize(rows, methods, components, statistics, gates):
    # 两级汇总：
    # - 组件级：每个 (component, method) 的平均集成增益（相对 single-state 与 native）；
    # - 方法级：跨组分聚合，附 bootstrap 95% CI、精确符号翻转 p 值，
    #   并按冻结的开发门槛(gates)判定各生成器是否通过。
    by_component_method = []
    effects = {}
    for method in methods:
        for component in components:
            selected = [
                row for row in rows
                if row["method"] == method and row["component_id"] == component
            ]
            if not selected:
                continue
            delta = float(np.mean([row["ensemble_minus_single_state"] for row in selected]))
            native_delta = float(np.mean([row["ensemble_minus_native"] for row in selected]))
            effects[(component, method)] = delta
            by_component_method.append({
                "component_id": component,
                "method": method,
                "heldout_conformers": len(selected),
                "mean_ensemble_minus_single_state": delta,
                "mean_ensemble_minus_native": native_delta,
            })
    by_method = {}
    for method_index, method in enumerate(methods):
        values = [effects[(component, method)] for component in components if (component, method) in effects]
        # 每个 method 使用 bootstrap_seed + method_index，保证各方法的随机流独立可复现。
        ci = bootstrap_mean(
            values,
            int(statistics["bootstrap_trials"]),
            int(statistics["bootstrap_seed"]) + method_index,
        )
        # 三道开发门槛：有效组分数下限、正增益组分比例下限、CI 下界大于 0。
        gate_results = {
            "minimum_valid_components": len(values) >= int(gates["minimum_valid_components_per_generator"]),
            "minimum_positive_component_fraction": (
                sum(value > 0 for value in values) / len(values) if values else 0.0
            ) >= float(gates["minimum_positive_component_fraction"]),
            "bootstrap_ci95_lower_above_zero": bool(ci and ci[0] > 0),
        }
        by_method[method] = {
            "valid_components": len(values),
            "mean_ensemble_minus_single_state": float(np.mean(values)) if values else None,
            "median_ensemble_minus_single_state": float(np.median(values)) if values else None,
            "positive_component_fraction": (
                sum(value > 0 for value in values) / len(values) if values else 0.0
            ),
            "component_bootstrap_ci95": ci,
            "exact_sign_flip_p": exact_sign_flip_p(values),
            "gate_results": gate_results,
            "passed": all(gate_results.values()),
        }
    return by_component_method, by_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_matched_scoring_dev_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # 与其他打分脚本一致：所有冻结输入先做 sha256 校验，任何不匹配立即失败。
    frozen = {
        "component_manifest": config["component_manifest"],
        "raw_generation": config["raw_generation"],
        "abeta_pose_audit": config["pose_audits"]["abeta"],
        "nonabeta_pose_audit": config["pose_audits"]["nonabeta"],
    }
    paths = {name: ROOT / item["path"] for name, item in frozen.items()}
    for name, path in paths.items():
        if sha256(path) != frozen[name]["sha256"]:
            raise ValueError(f"Frozen hash mismatch: {path}")

    manifest = json.loads(paths["component_manifest"].read_text(encoding="utf-8"))
    raw = json.loads(paths["raw_generation"].read_text(encoding="utf-8"))
    # 匹配生成必须零失败完成，否则不同方法的候选池不可比。
    if raw["status"] != "complete" or raw["failures"]:
        raise ValueError("Matched generation must be complete with zero failures")
    components = manifest["components"]
    methods = config["candidate_pool"]["methods"]
    pools = candidate_pools(raw, methods)
    poses = pose_rows_by_component(
        components,
        json.loads(paths["abeta_pose_audit"].read_text(encoding="utf-8")),
        json.loads(paths["nonabeta_pose_audit"].read_text(encoding="utf-8")),
    )

    rows = []
    matrices = []
    invalid_component_methods = []
    expected_conformers = int(config["scoring"]["conformers_per_component"])
    for component in components:
        # 每个组分的保留 pose 数必须与配置一致，保证留一评估口径统一。
        component_id = component["component_id"]
        if len(poses[component_id]) != expected_conformers:
            raise ValueError(f"{component_id} does not have {expected_conformers} held-out poses")
        for method in methods:
            sequences = pools.get((component_id, method), [])
            if not sequences:
                raise ValueError(f"Missing candidate pool: {component_id} {method}")
            matrix, native, pose_provenance = score_matrix(
                component, poses[component_id], sequences, config["scoring"]
            )
            # 质量门：若某 pose 的 H3 侧链接触数低于阈值，视为低质量构象，
            # 该 (component, method) 整体跳过并记录原因，不进入统计。
            minimum_sidechain_contacts = int(
                config["scoring"]["minimum_sidechain_contacts_per_conformer"]
            )
            invalid_poses = [
                row for row in pose_provenance
                if row["h3_sidechain_contacts"] < minimum_sidechain_contacts
            ]
            if invalid_poses:
                invalid_component_methods.append({
                    "component_id": component_id,
                    "method": method,
                    "reason": "insufficient_h3_sidechain_contacts",
                    "invalid_conformers": [row["conformer"] for row in invalid_poses],
                })
                continue
            rows.extend(leave_one_out(
                component_id,
                method,
                sequences,
                matrix,
                native,
                config["scoring"]["ensemble_aggregation"],
            ))
            # 保留完整打分矩阵与 pose 溯源，便于事后复核与重聚合。
            matrices.append({
                "component_id": component_id,
                "method": method,
                "sequences": sequences,
                "scores": matrix.tolist(),
                "native_scores": native.tolist(),
                "poses": pose_provenance,
            })

    component_ids = [component["component_id"] for component in components]
    component_summary, method_summary = summarize(
        rows, methods, component_ids, config["statistics"], config["development_gates"]
    )
    output = {
        "schema_version": 1,
        "status": (
            "all_generator_development_gates_passed"
            if all(row["passed"] for row in method_summary.values())
            else "development_gate_failed"
        ),
        "classification": config["classification"],
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "inputs": {name: sha256(path) for name, path in paths.items()},
        },
        "method_summary": method_summary,
        "component_method_summary": component_summary,
        "invalid_component_methods": invalid_component_methods,
        "leave_one_conformer_out": rows,
        "score_matrices": matrices,
        "claim_boundary": config["claim_boundary"],
    }
    out_path = ROOT / (args.out or config["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": output["status"], "method_summary": method_summary}, indent=2))


if __name__ == "__main__":
    main()
