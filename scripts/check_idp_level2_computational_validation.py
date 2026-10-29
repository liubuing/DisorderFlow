#!/usr/bin/env python
"""Fail-closed adjudication of DisorderFlow level-2 computational evidence.

This module does not calculate a new biological score. It verifies immutable
evidence artifacts and evaluates preregistered engineering/scientific gates.
Its purpose is to prevent a collection of individually favorable diagnostics
from being reported as level-2 success when any required axis still fails.
"""
# 本脚本对 DisorderFlow level-2 计算证据做"失败即关闭"的裁定：
# 只校验不可变证据产物的哈希并评估 YAML 中预注册的工程/科学门槛，
# 不计算任何新的生物学分数。门槛支持四类规则：字段等于期望值、
# 分数不低于/不高于阈值、最低/最高占比（分母为空按 0 计，绝不空过）、
# 以及"某臂占比须同时不低于阈值与对照臂"的对比规则。
# 已有失败证据优先于缺失确认：补采新队列不能抹去已观察到的方法失败。

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    # 计算文件 SHA-256，用于证据产物的冻结哈希校验
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested(payload, field):
    """Read a dotted JSON field such as ``method_summary.proteinmpnn.passed``."""
    # 按点号路径逐层取值；任一层缺失会抛 KeyError（失败即关闭）
    value = payload
    for part in field.split("."):
        value = value[part]
    return value


def evaluate_gate(rule, evidence):
    """Evaluate one declarative YAML gate and retain the observed value."""
    # 单条门槛求值：按规则键分派（expected / minimum_fraction /
    # maximum_fraction / minimum / arm_fraction_at_least），
    # 未识别的规则类型直接报错，绝不静默通过
    payload = evidence[rule["source"]]
    observed = None
    if "expected" in rule:
        observed = nested(payload, rule["field"])
        passed = observed == rule["expected"]
    elif "minimum_fraction" in rule or "maximum_fraction" in rule:
        # Empty denominators fail closed. They never become vacuous passes.
        numerator = float(nested(payload, rule["numerator"]))
        denominator = float(nested(payload, rule["denominator"]))
        observed = numerator / denominator if denominator else 0.0
        passed = (
            observed >= float(rule["minimum_fraction"])
            if "minimum_fraction" in rule
            else observed <= float(rule["maximum_fraction"])
        )
    elif "minimum" in rule:
        observed = float(nested(payload, rule["field"]))
        passed = observed >= float(rule["minimum"])
    elif "arm_fraction_at_least" in rule:
        settings = rule["arm_fraction_at_least"]
        summaries = payload["arm_summary"]
        arm = summaries[settings["arm"]]
        comparator = summaries[settings["comparator_arm"]]
        observed = float(arm[settings["numerator"]]) / float(
            arm[settings["denominator"]]
        )
        comparator_fraction = float(comparator[settings["numerator"]]) / float(
            comparator[settings["denominator"]]
        )
        passed = observed >= float(settings["minimum"]) and observed >= comparator_fraction
        return {
            "passed": passed,
            "observed": observed,
            "comparator_observed": comparator_fraction,
            "category": rule["category"],
        }
    else:
        raise ValueError(f"Unsupported gate rule: {rule}")
    return {"passed": passed, "observed": observed, "category": rule["category"]}


# 生成 Markdown 裁定报告：门槛明细表 + 失败/缺失证据清单 + 决策与声明边界
def write_report(path, payload):
    lines = [
        "# DisorderFlow Level-2 Computational Validation", "",
        f"Status: `{payload['status']}`", "",
        f"Passed gates: {payload['passed_gate_count']}/{payload['required_gate_count']}", "",
        "| Gate | Category | Passed | Observed |", "|---|---|---:|---:|",
    ]
    for name, result in payload["gates"].items():
        lines.append(
            f"| {name} | {result['category']} | {result['passed']} | "
            f"{result.get('observed')} |"
        )
    lines.extend([
        "", "## Failed Existing Evidence", "",
        *[f"- {name}" for name in payload["failed_existing_evidence"]],
        "", "## Missing Evidence", "",
        *[f"- {name}" for name in payload["missing_evidence"]],
        "", "## Decision", "", payload["decision"], "",
        f"Claim boundary: {payload['claim_boundary']}", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def check(config_path, output_path=None):
    """Verify evidence hashes, evaluate all gates, and write the decision."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    evidence = {}
    provenance = {}
    # Hash verification happens before JSON parsing so a changed result cannot
    # be evaluated under thresholds frozen for a different artifact.
    for name, item in config["evidence"].items():
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing level-2 evidence: {path}")
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Level-2 evidence hash mismatch: {path}")
        evidence[name] = json.loads(path.read_text(encoding="ascii"))
        provenance[name] = {"path": item["path"], "sha256": digest}

    gates = {
        name: evaluate_gate(rule, evidence)
        for name, rule in config["gates"].items()
    }
    failed_existing = [
        name for name, result in gates.items()
        if result["category"] == "existing_evidence" and not result["passed"]
    ]
    missing = [
        name for name, result in gates.items()
        if result["category"] == "missing_evidence" and not result["passed"]
    ]
    # Existing negative evidence takes precedence over missing confirmation.
    # Acquiring a new cohort cannot erase a method failure already observed.
    if failed_existing:
        status = config["decision"]["failed_status"]
    elif missing:
        status = config["decision"]["blocked_status"]
    else:
        status = config["decision"]["passed_status"]
    payload = {
        "schema_version": 1,
        "status": status,
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "passed": status == config["decision"]["passed_status"],
        "passed_gate_count": sum(result["passed"] for result in gates.values()),
        "required_gate_count": len(gates),
        "failed_existing_evidence": failed_existing,
        "missing_evidence": missing,
        "gates": gates,
        "provenance": provenance,
        "decision": (
            "do_not_claim_level2; improve_method_before_new_confirmation"
            if failed_existing else
            "acquire_untouched_idp_cohort_without_changing_frozen_method"
            if missing else
            "level2_computational_claim_permitted_within_claim_boundary"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    output_path = Path(output_path or ROOT / config["outputs"]["status"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    report_path = ROOT / config["outputs"]["report"]
    if output_path == ROOT / config["outputs"]["status"]:
        write_report(report_path, payload)
    return payload


def main():
    # 主入口：默认使用 idp_level2_computational_validation_v1 配置；
    # --output 仅覆盖状态 JSON 路径，输出到配置默认位置时才同时写 Markdown 报告
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_level2_computational_validation_v1.yml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(ROOT / args.config, ROOT / args.output if args.output else None)
    print(json.dumps({
        "status": result["status"],
        "passed_gates": result["passed_gate_count"],
        "required_gates": result["required_gate_count"],
        "failed_existing_evidence": result["failed_existing_evidence"],
        "missing_evidence": result["missing_evidence"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
