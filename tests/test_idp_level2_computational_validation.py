"""IdP Level-2 计算验证的契约测试：守护比例门槛在阈值处 fail-closed（0.75 < 0.8 必须失败），
以及当前证据集下 Level-2 不得宣称通过且升级声明保持缺失——防止验证等级被静默放宽。"""
from pathlib import Path

from scripts.check_idp_level2_computational_validation import check, evaluate_gate

ROOT = Path(__file__).resolve().parents[1]


# 边界严格：3/4=0.75 低于最小比例 0.8 时必须判失败——"恰好差一点"也不放行。
def test_fraction_gate_is_fail_closed_at_threshold():
    evidence = {"result": {"good": 3, "total": 4}}
    rule = {
        "source": "result", "numerator": "good", "denominator": "total",
        "minimum_fraction": 0.8, "category": "existing_evidence",
    }
    result = evaluate_gate(rule, evidence)
    assert result["observed"] == 0.75
    assert result["passed"] is False


# 等级诚实：四项既有证据门全部失败、未触动队列证据缺失时，状态必须为 level2_failed_existing_evidence
# 且各失败项逐一列名——防止验证结论被夸大或失败原因被笼统化。
def test_current_level2_evidence_fails_without_claim_upgrade(tmp_path):
    result = check(
        ROOT / "configs/benchmarks/idp_level2_computational_validation_v1.yml",
        tmp_path / "status.json",
    )
    assert result["status"] == "level2_failed_existing_evidence"
    assert result["passed"] is False
    assert "idp_extension_replication" in result["failed_existing_evidence"]
    assert "independent_bfn_scoring" in result["failed_existing_evidence"]
    assert "expanded_idp_independent_support" in result["failed_existing_evidence"]
    assert "expanded_idp_source_stability" in result["failed_existing_evidence"]
    assert "untouched_idp_cohort" in result["missing_evidence"]
