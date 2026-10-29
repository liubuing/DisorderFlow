#!/usr/bin/env python
"""Verify that all records for a scientific unit stay in one CV fold."""

# 本脚本审计 statecontrast v2 训练清单的交叉验证隔离性：
# 按"科学单元"（抗体谱系簇 × 全局序列簇）、teacher 组与候选组三个层级
# 检查同一单元/组的全部记录是否只出现在一个 CV 折内；任一跨折即为泄漏。
# 同时核对观测折集合与期望折数完全一致。隔离不通过则禁止训练
# （decision: train_only_when_isolation_passes），输出隔离审计 JSON。

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# 审计主流程：逐条记录登记科学单元/teacher 组/候选组出现过的折号，
# 某键对应的折号集合大小大于 1 即视为泄漏；再核对折号集合与期望一致。
def audit(manifest_path, output_path):
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    unit_folds = defaultdict(set)
    teacher_folds = defaultdict(set)
    candidate_folds = defaultdict(set)
    fold_components = defaultdict(set)
    fold_records = defaultdict(int)
    for row in manifest["records"]:
        fold = int(row["fold_id"])
        scientific_unit = (
            row["antibody_lineage_cluster"], row["global_sequence_cluster"])
        unit_folds[scientific_unit].add(fold)
        teacher_folds[row["teacher_group_id"]].add(fold)
        candidate_folds[row["group_id"]].add(fold)
        fold_components[fold].add(row["component_id"])
        fold_records[fold] += 1
    leaks = {
        "scientific_units": [str(key) for key, folds in unit_folds.items() if len(folds) != 1],
        "teacher_groups": [key for key, folds in teacher_folds.items() if len(folds) != 1],
        "candidate_groups": [key for key, folds in candidate_folds.items() if len(folds) != 1],
    }
    expected = set(range(int(manifest["cross_validation_folds"])))
    observed = set(fold_records)
    payload = {
        "schema_version": 1,
        "status": (
            "statecontrast_v2_cv_isolation_passed"
            if observed == expected and not any(leaks.values())
            else "statecontrast_v2_cv_isolation_failed"),
        "expected_folds": sorted(expected),
        "observed_folds": sorted(observed),
        "fold_summary": {
            str(fold): {
                "records": fold_records[fold],
                "components": len(fold_components[fold]),
                "component_ids": sorted(fold_components[fold]),
            }
            for fold in sorted(observed)
        },
        "leaks": leaks,
        "decision": "train_only_when_isolation_passes",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


# 主入口：解析 --manifest 与 --output 路径并驱动审计，打印折摘要与泄漏清单。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_expanded_states_v1/training_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_cross_validation_v2/isolation_audit.json"))
    args = parser.parse_args()
    result = audit(ROOT / args.manifest, ROOT / args.output)
    print(json.dumps({
        "status": result["status"],
        "fold_summary": result["fold_summary"],
        "leaks": result["leaks"],
    }, indent=2))


if __name__ == "__main__":
    main()
