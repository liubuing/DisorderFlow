#!/usr/bin/env python
"""Apply the predeclared 12-component ECLS gate without endpoint retuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_idp_ensemble_matched_candidates import bootstrap_mean, exact_sign_flip_p  # noqa: E402,I001

def sha256(path):
    # 计算文件 SHA-256，用于核对冻结输入与配置中预登记的哈希是否一致，
    # 防止在不知情中改用了被篡改或更新过的分析结果。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component_effects(analysis, method):
    # 从某方法的分析结果中抽取各 component 的效应量：
    # mean_ensemble_minus_single_state，即 ensemble 相对 single-state 的均值差，
    # 数值越大表示多构象集成相对单态越优。
    rows = analysis["method_summary"][method]["components"]
    return {
        row["component_id"]: float(row["mean_ensemble_minus_single_state"])
        for row in rows
    }


def summarize(effects, config, seed_offset=0):
    # 汇总一组组分效应并按预登记门限（predeclared_gates）逐项判定是否通过。
    # bootstrap 采用配置中的 trials/seed；seed_offset 按方法序号偏移随机种子，
    # 使不同方法/敏感性分析的重采样彼此独立又不失可复现性。
    # seed_offset=100 起始段保留给敏感性分析（剔除合成组分后重算）。
    # 注意阈值本身不做任何事后再调（endpoint retuning 禁止），全部来自冻结配置。
    values = list(effects.values())
    statistics = config["statistics"]
    gates = config["predeclared_gates"]
    ci = bootstrap_mean(
        values,
        int(statistics["bootstrap_trials"]),
        int(statistics["bootstrap_seed"]) + seed_offset,
    )
    positive = sum(value > 0 for value in values)
    zero = sum(value == 0 for value in values)
    negative = sum(value < 0 for value in values)
    gate_results = {
        "minimum_valid_components": len(values) >= int(gates["minimum_valid_components"]),
        "minimum_strictly_positive_components": positive
        >= int(gates["minimum_strictly_positive_components"]),
        "bootstrap_ci95_lower_above_zero": bool(ci and ci[0] > 0),
    }
    return {
        "valid_components": len(values),
        "mean_ensemble_minus_single_state": float(np.mean(values)),
        "median_ensemble_minus_single_state": float(np.median(values)),
        "strictly_positive_components": positive,
        "zero_components": zero,
        "negative_components": negative,
        "positive_component_fraction": positive / len(values),
        "component_bootstrap_ci95": ci,
        "exact_sign_flip_p": exact_sign_flip_p(values),
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "components": [
            {"component_id": component_id, "mean_ensemble_minus_single_state": value}
            for component_id, value in effects.items()
        ],
    }


def main():
    # 入口：读取冻结配置（含 development 与 expansion 两份分析的路径+SHA-256），
    # 先校验输入哈希，再按配置中的方法列表逐个汇总 development/expansion 合并效应。
    # 两个面板的组分不得重叠（防止同一 component 被重复计入而放大显著性）。
    # 敏感性分析剔除配置指定的合成组分，并将门限同步收紧为 11/8（与 12 组分
    # 主门的等比例约定），种子偏移 100+方法序号以独立复现。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_ecls_combined_12_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = {}
    analyses = {}
    for name in ("development_analysis", "expansion_analysis"):
        path = ROOT / config[name]["path"]
        digest = sha256(path)
        if digest != config[name]["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {path}")
        inputs[name] = {"path": config[name]["path"], "sha256": digest}
        analyses[name] = json.loads(path.read_text(encoding="utf-8"))

    method_summary = {}
    sensitivity_summary = {}
    excluded = config["sensitivity"]["excluded_component"]
    for method_index, method in enumerate(config["methods"]):
        development = component_effects(analyses["development_analysis"], method)
        expansion = component_effects(analyses["expansion_analysis"], method)
        overlap = set(development) & set(expansion)
        if overlap:
            raise ValueError(f"Component overlap between panels: {sorted(overlap)}")
        combined = {**development, **expansion}
        method_summary[method] = summarize(combined, config, method_index)
        sensitivity = {key: value for key, value in combined.items() if key != excluded}
        sensitivity_config = {
            **config,
            "predeclared_gates": {
                **config["predeclared_gates"],
                "minimum_valid_components": 11,
                "minimum_strictly_positive_components": 8,
            },
        }
        sensitivity_summary[method] = summarize(
            sensitivity, sensitivity_config, 100 + method_index
        )

    primary = config["primary_method"]
    output = {
        "schema_version": 1,
        "status": "predeclared_primary_gate_passed"
        if method_summary[primary]["passed"] else "predeclared_primary_gate_failed",
        "classification": config["classification"],
        "primary_method": primary,
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "inputs": inputs,
        },
        "method_summary": method_summary,
        "sensitivity_excluding_synthetic_component": {
            "excluded_component": excluded,
            "methods": sensitivity_summary,
        },
        "claim_boundary": config["claim_boundary"],
    }
    output_path = ROOT / (args.out or config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": output["status"],
        "primary": method_summary[primary],
        "sensitivity": sensitivity_summary[primary],
    }, indent=2))


if __name__ == "__main__":
    main()
