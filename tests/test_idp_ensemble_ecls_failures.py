import numpy as np
import pytest

from scripts.analyze_idp_ensemble_ecls_failures import (
    classify_effect,
    diversity_metrics,
    winner_index,
)

THRESHOLDS = {
    "mathematical_effect_epsilon": 1e-9,
    "practical_neutral_absolute_effect": 1e-4,
}


def test_effect_classification_separates_identity_equivalence_and_near_tie():
    assert classify_effect(0.0, 5, 5, THRESHOLDS) == "selector_identity"
    assert classify_effect(0.0, 1, 5, THRESHOLDS) == "score_equivalence"
    assert classify_effect(5e-6, 0, 5, THRESHOLDS) == "numerical_near_tie"
    assert classify_effect(-0.01, 0, 5, THRESHOLDS) == "negative_transfer"


def test_winner_tie_break_matches_lexicographically_greatest_sequence():
    matrix = np.ones((2, 5))
    aggregation = {
        "p25_weight": 0.45,
        "minimum_weight": 0.25,
        "mean_weight": 0.20,
        "standard_deviation_penalty": 0.10,
    }
    assert winner_index(matrix, list(range(5)), ["AAAA", "CCCC"], aggregation) == 1


def test_diversity_reports_raw_duplicates_and_hamming_distance():
    result = diversity_metrics(["AAAA", "AAAC"], ["AAAA", "AAAA", "AAAC"])
    assert result["raw_candidates"] == 3
    assert result["unique_candidates"] == 2
    assert result["duplicate_fraction"] == pytest.approx(1 / 3)
    assert result["pairwise_hamming"]["minimum"] == 1
    assert result["near_clonal"] is True
