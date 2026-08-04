from scripts.analyze_h3_generator_calibration import (
    aggregate_units,
    benjamini_hochberg,
    candidate_nnr,
    choose_mpnn_coefficient,
    exact_sign_permutation_p,
)


def _row(unit="E1"):
    return {
        "unit": unit,
        "native_scores": {"complex_nll": 3.0, "apo_nll": 3.0},
        "candidates": [
            {"complex_nll": 1.0, "apo_nll": 1.0},
            {"complex_nll": 2.0, "apo_nll": 1.0},
        ],
    }


def test_apo_weight_can_reverse_generator_self_likelihood_bias():
    row = _row()
    assert candidate_nnr(row, 0.0) == 0.0
    assert candidate_nnr(row, 2.0) == 1.0


def test_aggregation_averages_seeds_within_unit():
    assert aggregate_units([_row(), _row()], 2.0) == {"E1": 1.0}


def test_calibration_uses_smallest_passing_coefficient():
    settings = {
        "lambda_grid_start": 0,
        "lambda_grid_stop": 3,
        "lambda_grid_step": 1,
        "minimum_mean_nnr": 0.95,
        "minimum_top1_fraction": 0.90,
    }
    selected, _ = choose_mpnn_coefficient([_row()], settings)
    assert selected == 2.0


def test_exact_sign_permutation_detects_consistent_effect():
    assert exact_sign_permutation_p([1.0, 1.0, 1.0]) == 0.25


def test_benjamini_hochberg_is_monotone_in_rank():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.04, 0.04]
