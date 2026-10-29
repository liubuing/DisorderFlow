#!/usr/bin/env python
"""Prepare and sequence-audit the frozen nonviral exploratory v3 subset."""

# 本脚本构建并审计 IDP ensemble v3 的"非病毒"探索性冻结子集：
# 从候选文件中按配置筛选目标成分（排除病毒等超范围成分并记录排除原因），
# 核对成分/靶标/候选数量与冻结期望值，随后对每条候选序列做 QC
# （氨基酸合法性、最长疏水连续段、电荷代理、糖基化/脱酰胺/异构化/氧化 motif）。
# 湿实验测量被刻意跳过（仅计算分支），原 v3 修正门槛保持阻断状态。
# 输出：子集冻结 JSON、序列 QC 分析 JSON、实验状态 JSON；已存在则拒绝覆盖。

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# 残基分类常量：疏水、阳性电荷、阴性电荷（用于电荷代理与疏水段统计）
HYDROPHOBIC = set("AILMFWVY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")


def sha256(path):
    # 计算文件 SHA-256 摘要，用于冻结输入的来源校验
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 计算序列中最长连续疏水残基段长度（用于聚集倾向的粗略代理）
def longest_hydrophobic_run(sequence):
    longest = current = 0
    for amino_acid in sequence:
        current = current + 1 if amino_acid in HYDROPHOBIC else 0
        longest = max(longest, current)
    return longest


# 计算单条候选序列的 QC 指标：
# - 电荷代理 = 阳性(K/R/H)残基数 - 阴性(D/E)残基数；
# - N-连接糖基化 motif 用正则 N[^P][ST] 匹配（Asn 后接 Pro 不计）；
# - 脱酰胺/异构化/氧化易感性按合同配置的 motif/残基清单标记；
# - qc_pass 要求全部硬性检查通过且不含糖基化 motif。
def sequence_metrics(sequence, contract):
    allowed = set(contract["allowed_amino_acids"])
    charge = sum(aa in POSITIVE for aa in sequence) - sum(
        aa in NEGATIVE for aa in sequence
    )
    glycosylation = [
        match.group(0)
        for match in re.finditer(r"N[^P][ST]", sequence)
    ]
    deamidation = [motif for motif in contract["flag_deamidation_motifs"] if motif in sequence]
    isomerization = [motif for motif in contract["flag_isomerization_motifs"] if motif in sequence]
    oxidation = sorted({aa for aa in contract["flag_oxidation_residues"] if aa in sequence})
    run = longest_hydrophobic_run(sequence)
    checks = {
        "valid_amino_acids": set(sequence) <= allowed,
        "hydrophobic_run_ok": run <= contract["maximum_hydrophobic_run"],
        "charge_proxy_ok": abs(charge) <= contract["maximum_absolute_charge_proxy"],
    }
    return {
        "sequence": sequence,
        "length": len(sequence),
        "charge_proxy": charge,
        "hydrophobic_fraction": round(
            sum(aa in HYDROPHOBIC for aa in sequence) / len(sequence), 6
        ),
        "longest_hydrophobic_run": run,
        "n_linked_glycosylation_motifs": glycosylation,
        "deamidation_motifs": deamidation,
        "isomerization_motifs": isomerization,
        "oxidation_residues": oxidation,
        "checks": checks,
        "qc_pass": all(checks.values()) and not glycosylation,
    }


# 构建冻结子集与序列 QC 分析：
# 1) 先检查三个输出文件均不存在，避免覆盖既有冻结产物；
# 2) 按 included_targets 过滤候选成分，其余成分连同原因写入排除清单；
# 3) 核对成分/靶标/候选总数与配置中的冻结期望值，不一致即抛错；
# 4) 对子集中全部候选逐臂生成 QC 记录；
# 5) 实验状态固定为"刻意跳过测量"，并说明对原 v3 修正门槛的影响。
def analyze(config_path, candidates_path, subset_path, analysis_path, status_path):
    for path in (subset_path, analysis_path, status_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite nonviral artifact: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    candidates = json.loads(candidates_path.read_text(encoding="ascii"))
    included_targets = set(config["source"]["included_targets"])
    included, excluded = {}, []
    for component_id, component in candidates["components"].items():
        if component["target"] in included_targets:
            included[component_id] = component
        else:
            excluded.append({
                "component_id": component_id,
                "target": component["target"],
                "reason": "viral_or_outside_frozen_nonviral_targets",
            })
    expected = config["selection"]
    candidate_count = sum(
        len(arm["candidates"])
        for component in included.values()
        for arm in component["arms"].values()
    )
    targets = {component["target"] for component in included.values()}
    if (
        len(included) != expected["expected_components"]
        or len(targets) != expected["expected_targets"]
        or candidate_count != expected["expected_candidates"]
    ):
        raise RuntimeError("Nonviral subset does not match the frozen counts")
    subset = {
        "schema_version": 1,
        "status": "nonviral_subset_frozen",
        "classification": config["classification"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_candidates": str(candidates_path),
        "source_candidates_sha256": sha256(candidates_path),
        "component_count": len(included),
        "target_count": len(targets),
        "candidate_count": candidate_count,
        "native_control_count": len(included),
        "components": included,
        "excluded_components": excluded,
        "claim_boundary": config["claim_boundary"],
    }
    records = []
    for component_id, component in included.items():
        for arm_name, arm in component["arms"].items():
            for candidate in arm["candidates"]:
                records.append({
                    "component_id": component_id,
                    "target": component["target"],
                    "arm": arm_name,
                    "substitution_bucket": candidate["substitution_bucket"],
                    **sequence_metrics(candidate["sequence"], config["sequence_qc"]),
                })
    analysis = {
        "schema_version": 1,
        "status": "nonviral_sequence_qc_complete",
        "classification": config["classification"],
        "subset": str(subset_path),
        "record_count": len(records),
        "qc_pass_count": sum(row["qc_pass"] for row in records),
        "flagged_count": sum(not row["qc_pass"] for row in records),
        "records": records,
        "claim_boundary": config["claim_boundary"],
    }
    experiment = {
        "schema_version": 1,
        "status": "experimental_measurements_intentionally_skipped",
        "measurement_gate_passed": False,
        "reason": "user_requested_computational_nonviral_branch",
        "effect_on_correction_gate": "original_v3_remains_4_of_5_and_blocked",
        "claim_boundary": config["claim_boundary"],
    }
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(json.dumps(subset, indent=2) + "\n", encoding="ascii")
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="ascii")
    status_path.write_text(json.dumps(experiment, indent=2) + "\n", encoding="ascii")
    return subset, analysis, experiment


# 主入口：读取配置，按配置中的输入/输出路径驱动 analyze()，打印摘要。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_v3_nonviral_exploratory.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    outputs = config["outputs"]
    subset, analysis, experiment = analyze(
        config_path,
        ROOT / config["source"]["candidates"],
        ROOT / outputs["subset"],
        ROOT / outputs["analysis"],
        ROOT / outputs["experiment_status"],
    )
    print(json.dumps({
        "subset_status": subset["status"],
        "components": subset["component_count"],
        "targets": subset["target_count"],
        "candidates": subset["candidate_count"],
        "qc_pass": analysis["qc_pass_count"],
        "flagged": analysis["flagged_count"],
        "experiment_status": experiment["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
