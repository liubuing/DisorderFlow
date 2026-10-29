#!/usr/bin/env python
"""Apply final abstention rules to exposed development robustness results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def audit(config_path, source_path, loo_path, output):
    # 对已曝光的开发数据应用最终弃权（abstention）规则。
    # 参数:
    #   config_path: 冻结确认配置（YAML），此处仅用于登记来源，不再搜索阈值。
    #   source_path: 来源面板敏感性分析结果（matched_bucket_contrasts）。
    #   loo_path: leave-one-pose-out 稳定性结果（逐组件的 matched_bucket_stability）。
    #   output: 输出 JSON 路径；已存在则拒绝写入（fail-closed）。
    # 返回: payload 字典——逐对 (component_id, substitution_bucket) 的方向一致性
    #   与最终是否被接纳。
    # 规则：只有当来源面板方向一致 且 leave-one-pose-out 在所有省略下方向
    # 都稳定时才接纳；否则弃权并记录原因。该脚本是对已曝光开发数据的
    # 规则审计，不得外推为 IDP、结合或确认性结论（见 claim_boundary）。
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite abstention audit: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    source = json.loads(source_path.read_text(encoding="ascii"))
    loo = json.loads(loo_path.read_text(encoding="ascii"))
    source_rows = {
        (row["component_id"], row["substitution_bucket"]): row
        for row in source["matched_bucket_contrasts"]
    }
    loo_rows = {
        (component["component_id"], row["substitution_bucket"]): row
        for component in loo["components"]
        for row in component["matched_bucket_stability"]
    }
    records = []
    for key in sorted(source_rows):
        source_stable = source_rows[key]["source_panel_sign_consistent"]
        loo_stable = loo_rows[key]["direction_stable_across_all_omissions"]
        records.append({
            "component_id": key[0],
            "substitution_bucket": key[1],
            "source_direction_consistent": source_stable,
            "leave_one_pose_out_direction_consistent": loo_stable,
            "admitted_by_final_abstention": source_stable and loo_stable,
            "abstention_reasons": [
                reason for condition, reason in (
                    (not source_stable, "source_panel_direction_inconsistent"),
                    (not loo_stable, "leave_one_pose_out_direction_inconsistent"),
                ) if condition
            ],
        })
    payload = {
        "schema_version": 1,
        "status": "final_abstention_development_audit_complete",
        "classification": "exposed_non_idp_development_rule_audit",
        "config": str(config_path.relative_to(ROOT)),
        "matched_pair_count": len(records),
        "admitted_pair_count": sum(
            row["admitted_by_final_abstention"] for row in records
        ),
        "abstained_pair_count": sum(
            not row["admitted_by_final_abstention"] for row in records
        ),
        "records": records,
        "decision": "freeze_rules_without_further_threshold_search",
        "claim_boundary": (
            "Rule audit on exposed non-IDP development data only; no IDP, "
            "binding, or confirmatory claim"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    # 命令行入口：默认路径全部指向冻结的 v3 稳健性产物，结果落盘并打印摘要。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_final_confirmation_v1.yml"),
    )
    parser.add_argument(
        "--source", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_correction_v3_robustness/"
            "source_sensitivity.json"
        ),
    )
    parser.add_argument(
        "--loo", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_correction_v3_robustness/"
            "leave_one_pose_out.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_final_confirmation_v1/"
            "abstention_development_audit.json"
        ),
    )
    args = parser.parse_args()
    result = audit(
        ROOT / args.config, ROOT / args.source, ROOT / args.loo, ROOT / args.output
    )
    print(json.dumps({
        "status": result["status"],
        "matched_pairs": result["matched_pair_count"],
        "admitted": result["admitted_pair_count"],
        "abstained": result["abstained_pair_count"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
