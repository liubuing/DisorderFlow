"""可开发性测量审计的契约测试：守护 fail-closed 原则——测量缺失时审计必须阻断候选而非放行，
只有重复数与产量/SEC/聚体/多特异性全部达标才可判 complete。"""
import json

from scripts.audit_developability_measurements import audit


# 缺失即阻断：任何一项测量为 None 时状态必须为 blocked 且通过数为 0，
# 防止数据不全的候选带着"未验证"的空白进入下游宣称。
def test_missing_measurements_fail_closed(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        "classification: test\n"
        "required_measurements:\n"
        "  minimum_replicates: 2\n"
        "  minimum_expression_mg_per_l: 10\n"
        "  minimum_sec_monomer_fraction: 0.9\n"
        "  maximum_aggregate_fraction: 0.1\n"
        "  maximum_polyspecificity_normalized: 0.35\n"
        "claim_boundary: test only\n")
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"records": [{
        "candidate_id": "x", "replicates": None,
        "expression_mg_per_l": None, "sec_monomer_fraction": None,
        "aggregate_fraction": None, "polyspecificity_normalized": None,
    }]}))
    result = audit(config, measurements, tmp_path / "out.json")
    assert result["status"] == "developability_measurements_blocked"
    assert result["pass_count"] == 0


# 放行条件：全部指标达标的完整记录才判 complete 且通过数加一，保证审计门槛双向都有效。
def test_complete_measurement_record_can_pass(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        "classification: test\n"
        "required_measurements:\n"
        "  minimum_replicates: 2\n"
        "  minimum_expression_mg_per_l: 10\n"
        "  minimum_sec_monomer_fraction: 0.9\n"
        "  maximum_aggregate_fraction: 0.1\n"
        "  maximum_polyspecificity_normalized: 0.35\n"
        "claim_boundary: test only\n")
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"records": [{
        "candidate_id": "x", "replicates": 2,
        "expression_mg_per_l": 20, "sec_monomer_fraction": 0.95,
        "aggregate_fraction": 0.05, "polyspecificity_normalized": 0.2,
    }]}))
    result = audit(config, measurements, tmp_path / "out.json")
    assert result["status"] == "developability_measurements_complete"
    assert result["pass_count"] == 1
