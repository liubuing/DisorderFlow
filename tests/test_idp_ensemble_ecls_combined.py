from scripts.analyze_idp_ensemble_ecls_combined import summarize


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
