import numpy as np

from scripts.analyze_idp_ensemble_endpoint_v3 import (
    complex_robust_cost,
    median_provenance,
    select_ensemble,
)

CONFIG = {
    "selection": {
        "complex_constraint": "candidate_robust_complex_nll_not_above_native_robust_complex_nll",
        "constraint_tolerance": 1e-12,
        "gain_aggregation": {
            "p25_weight": 0.45,
            "minimum_weight": 0.25,
            "mean_weight": 0.20,
            "standard_deviation_penalty": 0.10,
        },
        "complex_cost_aggregation": {
            "p75_weight": 0.45,
            "maximum_weight": 0.25,
            "mean_weight": 0.20,
            "standard_deviation_penalty": 0.10,
        },
    }
}


def test_complex_cost_is_negative_gain_robust_of_negative_nll():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    cost = complex_robust_cost(values, CONFIG["selection"]["complex_cost_aggregation"])
    assert cost > np.mean(values) * 0.8


def test_native_constraint_excludes_high_gain_poor_complex_candidate():
    matrices = {
        "sequences": ["POOR", "SAFE"],
        "native_index": 1,
        "gain": np.asarray([[10.0] * 4, [1.0] * 4]),
        "complex": np.asarray([[5.0] * 4, [1.0] * 4]),
    }
    winner, feasible, _, _ = select_ensemble(matrices, [0, 1, 2, 3], CONFIG)
    assert feasible == [1]
    assert winner == 1


def test_top_fraction_constraint_keeps_only_best_complex_subset():
    config = {
        "selection": {
            **CONFIG["selection"],
            "complex_constraint": "top_fraction_by_robust_complex_nll",
            "complex_top_fraction": 0.25,
            "minimum_feasible_candidates": 2,
        }
    }
    matrices = {
        "sequences": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "native_index": 7,
        "gain": np.asarray([[float(index)] * 4 for index in range(8)]),
        "complex": np.asarray([[float(index)] * 4 for index in range(8)]),
    }
    winner, feasible, _, threshold = select_ensemble(
        matrices, [0, 1, 2, 3], config
    )
    assert feasible == [0, 1]
    assert threshold == complex_robust_cost(
        matrices["complex"][1], config["selection"]["complex_cost_aggregation"]
    )
    assert winner == 1


def test_median_provenance_supports_two_single_state_observations():
    records = [
        {"gain": 3.0, "training_conformer": 1, "candidate": "B"},
        {"gain": 1.0, "training_conformer": 0, "candidate": "A"},
    ]
    median, contributors = median_provenance(records, "gain")
    assert median == 2.0
    assert [row["gain"] for row in contributors] == [1.0, 3.0]
