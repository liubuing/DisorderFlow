"""IdP 系综 ECLS 联合分析主门槛的契约测试：守护"至少 8 个严格为正成分"的预声明门槛，
不足 8 个时整体必须判失败——防止少数成分的正效应冒充全局证据。"""
from scripts.analyze_idp_ensemble_ecls_combined import summarize


# 门槛计数：7 个正效应 + 5 个零效应时严格正成分只有 7 个（<8），门槛必须失败且整体 passed=False——
# 防止门槛常量被放松或把零效应计入支持证据。
def test_combined_gate_requires_eight_strictly_positive_components():
    config = {
        "statistics": {"bootstrap_trials": 1000, "bootstrap_seed": 1},
        "predeclared_gates": {
            "minimum_valid_components": 12,
            "minimum_strictly_positive_components": 8,
            "require_component_bootstrap_ci95_lower_above_zero": True,
        },
    }
    effects = {f"c{index}": 0.1 for index in range(7)}
    effects.update({f"z{index}": 0.0 for index in range(5)})
    result = summarize(effects, config)
    assert result["strictly_positive_components"] == 7
    assert result["gate_results"]["minimum_strictly_positive_components"] is False
    assert result["passed"] is False
