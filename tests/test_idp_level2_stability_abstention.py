"""IDP Level2 稳定性弃权诊断测试：验证双轴稳定性门槛与诊断-only 边界。

守护 scripts.analyze_idp_level2_stability_abstention.analyze：
资格必须同时要求来源面板稳定与留一 pose 稳定（两轴缺一不可）、
全部 48 对尝试都被评估、决策固定为 diagnostic_only（不得改动 level2 状态）——
防止单轴稳定被误判为合格或诊断结论越权。
"""
from pathlib import Path

from scripts.analyze_idp_level2_stability_abstention import analyze

ROOT = Path(__file__).resolve().parents[1]


# 双轴与 (AND) 语义：eligible 必须等于两个稳定性检查的逻辑与，
# 决策必须保持 diagnostic_only——防止稳定性门槛放宽或结论越权改状态。
def test_two_axis_abstention_requires_both_stability_checks(tmp_path):
    result = analyze(
        ROOT / "configs/benchmarks/idp_level2_stability_abstention_diagnostic_v1.yml",
        tmp_path / "analysis.json",
    )
    assert result["attempted_pairs"] == 48
    assert all(
        row["eligible"] == (
            row["source_panel_stable"] and row["leave_one_pose_out_stable"]
        )
        for row in result["records"]
    )
    assert result["decision"] == "diagnostic_only_do_not_change_level2_status"
