"""IDP 集成最终确认 v1 检查测试：验证三重冻结检查的 fail-closed 语义。

守护 scripts.check_idp_ensemble_final_confirmation_v1.check：
管线契约与 checkpoint 冻结、未触碰 IDP 队列就绪三项检查必须独立报告；
队列缺失时整体状态必须为 final_confirmation_blocked——
防止在数据未就绪时误发"最终确认通过"。
"""
from pathlib import Path

from scripts.check_idp_ensemble_final_confirmation_v1 import check

ROOT = Path(__file__).resolve().parents[1]


# fail-closed：缺新队列时即使前两项冻结检查通过，整体也必须 blocked，
# 且三项检查逐项可见（不合并隐藏）——防止半就绪状态被误判为可确认。
def test_final_confirmation_is_fail_closed_without_new_idp_cohort(tmp_path):
    result = check(
        ROOT / "configs/benchmarks/idp_ensemble_final_confirmation_v1.yml",
        tmp_path / "readiness.json",
    )
    assert result["status"] == "final_confirmation_blocked"
    assert result["checks"]["pipeline_contract_frozen"] is True
    assert result["checks"]["checkpoint_frozen"] is True
    assert result["checks"]["untouched_idp_cohort_ready"] is False
