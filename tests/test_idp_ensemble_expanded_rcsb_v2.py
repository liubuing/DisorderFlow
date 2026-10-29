"""IDP 集成扩展 RCSB v2 计划测试：验证采集节奏闸门与查询规模下限。

守护 scripts.run_idp_ensemble_expanded_rcsb_v2.build_plan：
未到节奏窗口时计划必须处于 cadence_gate_closed 且不可执行（防止过度采集）、
RCSB 查询谱系（两个查询分支的节点数）必须不低于冻结下限——
防止查询被静默缩窄导致覆盖缺口。
"""
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_idp_ensemble_expanded_rcsb_v2 import build_plan

ROOT = Path(__file__).resolve().parents[1]


# 节奏闸门与查询完整性：闸门关闭时 eligible_to_execute 必须为 False，
# 两个查询分支的节点数须达到冻结下限——防止绕过节奏或缩窄查询。
def test_expanded_rcsb_plan_is_cadence_guarded():
    plan = build_plan(
        ROOT / "configs/benchmarks/idp_ensemble_expanded_development_v2.yml",
        now=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert plan["status"] == "cadence_gate_closed"
    assert plan["eligible_to_execute"] is False
    assert len(plan["query_spec"]["query"]["nodes"][0]["nodes"]) >= 6
    assert len(plan["query_spec"]["query"]["nodes"][1]["nodes"]) >= 8
