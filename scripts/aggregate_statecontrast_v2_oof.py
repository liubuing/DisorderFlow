#!/usr/bin/env python
"""Aggregate five held-out-fold evaluations and apply the frozen gate once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    # 计算文件内容的 SHA-256 摘要，用于把每个 fold 评估结果锚定到本聚合报告，
    # 保证 provenance 可复核（文件被改动后哈希即失配）。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path):
    # 把绝对路径转成相对仓库根目录的 POSIX 风格路径，便于在报告里展示；
    # 若路径不在仓库内（ValueError）则原样返回字符串形式。
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def bootstrap_mean(values, trials=10000, seed=7201):
    # 对组分效应均值做自助法（bootstrap）重采样，返回 95% 置信区间 [q2.5, q97.5]。
    # trials=10000 为常规重采样次数，足以稳定估计分位数；seed=7201 为冻结协议
    # 预先指定的随机种子，保证结果可精确复现。重采样为有放回抽取（replace=True），
    # 与原样本等长的均值分布。
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = generator.choice(
        values, size=(int(trials), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(contract_path, gate_path, evaluations, output_path):
    """聚合五个 held-out fold 的评估结果并执行冻结判定门（gate）。

    参数:
        contract_path: 交叉验证契约 JSON，声明全部 fold 编号（缺一不可）。
        gate_path: 冻结的 YAML 判定门配置，含各阈值与 claim_boundary。
        evaluations: {fold 号: 评估 JSON 路径}，每个 fold 恰好对应一份结果，
            且该结果必须是在验证角色（split=="val"）上产生的。
        output_path: 聚合报告输出路径。

    关键约束（防挑选偏差）:
        - 任一 component 不得出现在多个 fold 中（否则说明数据泄漏/重复计数）；
        - 每个 fold 的评估文件与 checkpoint 均记录 SHA-256 作为 provenance；
        - 阈值含义: positive_fraction 要求多数组分效应为正（ensemble 优于
          single-state，效应越大越好）；source_fraction 要求各 source 面板方向
          一致；teacher_fraction 要求独立评分器也倾向 ensemble（非自我评价）；
          target_noninferior >= 0.5 表示至少半数配对达到目标态非劣；
          hard_failure 份额受上限约束（可开发性硬失败越少越好）。
    全部门通过则 status 置为 passed，决策为"冻结现有方法并用于一个全新
    未触碰的 IDP 面板"；否则不得接触未触碰面板。
    """
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    gate = yaml.safe_load(gate_path.read_text(encoding="ascii"))
    expected_folds = {int(row["fold"]) for row in contract["folds"]}
    if set(evaluations) != expected_folds:
        raise ValueError("Every frozen fold requires exactly one evaluation")
    payloads = {}
    for fold, path in evaluations.items():
        result = json.loads(path.read_text(encoding="ascii"))
        if result.get("split") != "val":
            raise ValueError(f"Fold {fold} was not evaluated on validation role")
        payloads[fold] = result

    component_effects = {}
    pairs = []
    provenance = []
    for fold in sorted(payloads):
        result = payloads[fold]
        overlap = set(component_effects) & set(result["component_effects"])
        if overlap:
            raise ValueError(f"Components occur in multiple held-out folds: {overlap}")
        component_effects.update(result["component_effects"])
        pairs.extend(result["matched_pairs"])
        provenance.append({
            "fold": fold,
            "path": display_path(evaluations[fold]),
            "sha256": sha256(evaluations[fold]),
            "checkpoint_sha256": result["checkpoint_sha256"],
        })
    values = list(component_effects.values())
    ci = bootstrap_mean(values)
    targets = {row["target"] for row in pairs}
    thresholds = gate["gates"]
    positive_fraction = sum(value > 0 for value in values) / max(1, len(values))
    source_fraction = sum(row["source_direction_consistent"] for row in pairs) / max(1, len(pairs))
    teacher_rows = [row for row in pairs if row["independent_teacher_difference"] is not None]
    teacher_fraction = sum(
        row["independent_teacher_difference"] > 0 for row in teacher_rows
    ) / max(1, len(teacher_rows))
    target_noninferior = sum(row["ensemble_target_noninferior"] for row in pairs) / max(1, len(pairs))
    hard_failure = sum(row["ensemble_hard_developability_failure"] for row in pairs) / max(1, len(pairs))
    required = gate["required_panels"]["expanded_replication"]
    checks = {
        "all_folds_present": True,
        "minimum_components": len(component_effects) >= int(required["minimum_components"]),
        "minimum_targets": len(targets) >= int(required["minimum_targets"]),
        "minimum_positive_component_fraction": positive_fraction >= float(
            thresholds["minimum_positive_component_fraction"]),
        "bootstrap_ci95_lower_above_zero": ci[0] > 0,
        "source_panel_direction_consistency": source_fraction >= float(
            thresholds["minimum_source_panel_direction_consistency"]),
        "independent_scorer_favored_fraction": teacher_fraction >= float(
            thresholds["minimum_independent_scorer_favored_fraction"]),
        "absolute_target_state_noninferiority": target_noninferior >= 0.5,
        "developability_hard_failure_fraction": hard_failure <= float(
            thresholds["maximum_developability_hard_failure_fraction"]),
    }
    output = {
        "schema_version": 1,
        "status": (
            "statecontrast_v2_oof_gate_passed" if all(checks.values())
            else "statecontrast_v2_oof_gate_failed"),
        "classification": gate["classification"],
        "contract_sha256": sha256(contract_path),
        "gate_sha256": sha256(gate_path),
        "fold_provenance": provenance,
        "component_count": len(component_effects),
        "target_count": len(targets),
        "matched_pair_count": len(pairs),
        "mean_component_effect": float(np.mean(values)),
        "positive_component_fraction": positive_fraction,
        "component_bootstrap_ci95": ci,
        "source_direction_consistent_fraction": source_fraction,
        "independent_teacher_favored_fraction": teacher_fraction,
        "target_noninferior_fraction": target_noninferior,
        "developability_hard_failure_fraction": hard_failure,
        "checks": checks,
        "component_effects": component_effects,
        "decision": (
            "freeze_unchanged_method_for_one_new_untouched_idp_panel"
            if all(checks.values()) else
            "do_not_access_untouched_panel; continue_exposed_development"),
        "claim_boundary": gate["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


def main():
    # 命令行入口：以 FOLD=path 形式逐个传入各 fold 的评估文件（可重复 --evaluation），
    # contract/gate 默认指向仓库内冻结配置，输出路径必填。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/contract.json"))
    parser.add_argument("--gate", type=Path, default=Path(
        "configs/benchmarks/statecontrast_v2_expanded_replication_gate.yml"))
    parser.add_argument("--evaluation", action="append", required=True,
                        help="FOLD=path/to/evaluation.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluations = {}
    for value in args.evaluation:
        fold, path = value.split("=", 1)
        evaluations[int(fold)] = ROOT / path
    result = aggregate(
        ROOT / args.contract, ROOT / args.gate, evaluations, ROOT / args.output)
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "targets": result["target_count"],
        "ci95": result["component_bootstrap_ci95"],
        "checks": result["checks"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
