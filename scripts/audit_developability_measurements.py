#!/usr/bin/env python
"""Create templates or audit measured candidate developability records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sequence_sha256(sequence):
    # 对氨基酸序列取 SHA-256；截前 12 位十六进制作为候选 ID 后缀以避免明文泄漏。
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def candidate_template(candidate_path):
    # 从候选集生成待填写的发展性（developability）测量模板。
    # 参数: candidate_path — 候选集 JSON，含 components 列表（每项有 component_id 与序列）。
    # 返回: 模板字典——每条 (component_id, sequence) 去重后一行，测量字段全部留空，
    #   由实验人员填写实测值（表达量 mg/L、SEC 单体比例、聚集比例、
    #   多特异性、Tm 摄氏度等）。
    # 注意: 计算代理值不被接受，模板 note 明确要求只填实测数据。
    source = json.loads(candidate_path.read_text(encoding="ascii"))
    rows = []
    seen = set()
    for component in source["components"]:
        sequence = component["candidate_sequence"]
        key = (component["component_id"], sequence)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "candidate_id": component["component_id"] + "|" + sequence_sha256(sequence)[:12],
            "component_id": component["component_id"],
            "sequence_sha256": sequence_sha256(sequence),
            "assay_batch": None,
            "replicates": None,
            "expression_mg_per_l": None,
            "sec_monomer_fraction": None,
            "aggregate_fraction": None,
            "polyspecificity_normalized": None,
            "melting_temperature_c": None,
        })
    return {
        "schema_version": 1,
        "status": "developability_measurement_template",
        "records": rows,
        "note": "Populate measured values; computational proxies are not accepted",
    }


def audit(config_path, measurements_path, output_path):
    # 按冻结配置审计实测发展性数据是否达标。
    # 参数:
    #   config_path: 冻结门槛配置（YAML），给出各测量的最小/最大阈值。
    #   measurements_path: 实测记录 JSON（通常由模板填写而来）。
    #   output_path: 审计结果输出 JSON 路径。
    # 返回: payload 字典——逐条记录的完整性检查、阈值检查与总体 pass/blocked 状态。
    # 检查项方向约定：表达量与 SEC 单体比例需 >= 下限；聚集比例与多特异性
    # 需 <= 上限；阈值全部来自冻结配置，不在代码中硬编码。
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    measurements = json.loads(measurements_path.read_text(encoding="ascii"))
    thresholds = config["required_measurements"]
    required_fields = (
        "replicates", "expression_mg_per_l", "sec_monomer_fraction",
        "aggregate_fraction", "polyspecificity_normalized",
    )
    records = []
    for row in measurements["records"]:
        missing = [field for field in required_fields if row.get(field) is None]
        checks = {
            "measurements_complete": not missing,
            "minimum_replicates": (
                row.get("replicates") is not None
                and int(row["replicates"]) >= int(thresholds["minimum_replicates"])),
            "expression": (
                row.get("expression_mg_per_l") is not None
                and float(row["expression_mg_per_l"])
                >= float(thresholds["minimum_expression_mg_per_l"])),
            "sec_monomer": (
                row.get("sec_monomer_fraction") is not None
                and float(row["sec_monomer_fraction"])
                >= float(thresholds["minimum_sec_monomer_fraction"])),
            "aggregation": (
                row.get("aggregate_fraction") is not None
                and float(row["aggregate_fraction"])
                <= float(thresholds["maximum_aggregate_fraction"])),
            "polyspecificity": (
                row.get("polyspecificity_normalized") is not None
                and float(row["polyspecificity_normalized"])
                <= float(thresholds["maximum_polyspecificity_normalized"])),
        }
        records.append({
            **row, "missing_measurements": missing,
            "checks": checks, "developability_pass": all(checks.values()),
        })
    payload = {
        "schema_version": 1,
        "status": (
            "developability_measurements_complete"
            if records and all(row["checks"]["measurements_complete"] for row in records)
            else "developability_measurements_blocked"),
        "classification": config["classification"],
        "candidate_count": len(records),
        "complete_count": sum(row["checks"]["measurements_complete"] for row in records),
        "pass_count": sum(row["developability_pass"] for row in records),
        "records": records,
        "decision": "use_measured_qc_only_never_replace_with_proxy",
        "claim_boundary": config["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    # 命令行入口：--template-from 生成测量模板，或 --measurements 执行审计（二选一）。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/benchmarks/developability_measurement_gate_v1.yml"))
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--template-from", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.measurements) == bool(args.template_from):
        parser.error("Choose exactly one of --measurements or --template-from")
    if args.template_from:
        output = ROOT / (args.output or Path(
            "reviewer_outputs/developability_measurement_gate_v1/template.json"))
        payload = candidate_template(ROOT / args.template_from)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    else:
        config_path = ROOT / args.config
        config = yaml.safe_load(config_path.read_text(encoding="ascii"))
        output = ROOT / (args.output or Path(config["output"]))
        payload = audit(config_path, ROOT / args.measurements, output)
    print(json.dumps({
        "status": payload["status"], "records": len(payload["records"]),
        "output": str(output.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
