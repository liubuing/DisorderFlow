#!/usr/bin/env python
"""Select a target-balanced, lineage-deduplicated expanded development panel."""

# 扩充版开发集筛选：从已冻结的 registry 中按"目标(target)平衡 + 谱系去重"
# 的规则挑选组分。每个 (target, lineage_proxy) 组合只保留分辨率最高
# （并列时按 PDB id / 重链 id 字典序）的一条记录，再按每个 target 的
# 组分上限截取，最终冻结为扩充开发面板。该面板仅用于开发迭代，
# 不参与未来确认实验（future_confirmation_eligible_count 恒为 0）。
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    # 文件 sha256：用于校验 registry 在冻结后未被改动。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolution_value(value):
    # 将分辨率解析为可比较的浮点数；缺失/非法/非正数一律记为无穷大，
    # 使这些记录在排序中被自然排到有效记录之后。
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed > 0 else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def select(config_path, output):
    if output.exists():
        # 面板一经冻结即不可覆盖，保证实验可追溯。
        raise FileExistsError(f"Refusing to overwrite expanded panel: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    registry_path = ROOT / config["input"]["registry"]
    # 校验 registry 哈希：选择规则冻结后输入源不能悄悄改变。
    if sha256(registry_path) != config["input"]["registry_sha256"]:
        raise RuntimeError("Expanded registry hash changed after selection freeze")
    registry = json.loads(registry_path.read_text(encoding="ascii"))
    # 谱系去重：同一 (target, lineage_proxy) 只留一条，
    # 排序键为 (分辨率, pdb_id, heavy_chain)，分辨率最优者优先。
    best_by_lineage = {}
    for row in registry["records"]:
        key = (row["target"], row["lineage_proxy"])
        candidate_key = (
            resolution_value(row["resolution"]),
            row["pdb_id"],
            row["heavy_chain"],
        )
        if key not in best_by_lineage or candidate_key < best_by_lineage[key][0]:
            best_by_lineage[key] = (candidate_key, row)
    # 目标平衡：按 target 分组、按排序键升序截取前 maximum 个组分。
    grouped = defaultdict(list)
    for (_target, _lineage), (sort_key, row) in best_by_lineage.items():
        grouped[row["target"]].append((sort_key, row))
    maximum = config["selection"]["maximum_components_per_target"]
    selected = []
    target_summary = {}
    for target in sorted(grouped):
        rows = [row for _, row in sorted(grouped[target])[:maximum]]
        selected.extend(rows)
        target_summary[target] = {
            "available_lineages": len(grouped[target]),
            "selected_components": len(rows),
        }
    target_count = len(target_summary)
    # 目标数下限：低于阈值说明 registry 覆盖不足，扩充面板不成立。
    if target_count < config["selection"]["minimum_targets"]:
        raise RuntimeError("Expanded registry has too few eligible targets")
    payload = {
        "schema_version": 1,
        "status": "balanced_expanded_development_panel_frozen",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "component_count": len(selected),
        "target_count": target_count,
        "lineage_proxy_count": len({row["lineage_proxy"] for row in selected}),
        "target_summary": target_summary,
        "components": selected,
        "future_confirmation_eligible_count": 0,
        "decision": "materialize_coordinates_for_expanded_development_only",
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/benchmarks/idp_ensemble_expanded_development_v2_selection.yml"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = select(config_path, ROOT / config["output"])
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "targets": result["target_count"],
        "lineage_proxies": result["lineage_proxy_count"],
        "target_summary": result["target_summary"],
        "future_confirmation_eligible": result[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
