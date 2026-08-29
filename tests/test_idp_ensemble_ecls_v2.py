import numpy as np

from scripts.analyze_idp_ensemble_ecls_v2 import jackknife_lower_envelope

AGGREGATION = {
    "p25_weight": 0.45,
    "minimum_weight": 0.25,
    "mean_weight": 0.20,
    "standard_deviation_penalty": 0.10,
}


def test_jackknife_lower_envelope_penalizes_single_conformer_dependence():
    matrix = np.asarray([
        [10.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    winner, scores, inner = jackknife_lower_envelope(
        matrix, [0, 1, 2, 3], ["SPIKE", "ROBUST"], AGGREGATION
    )
    assert winner == 1
    assert scores[0] < scores[1]
    assert len(inner[0]) == 4


def test_jackknife_tie_break_is_lexicographically_greatest():
    matrix = np.ones((2, 4))
    winner, _, _ = jackknife_lower_envelope(
        matrix, [0, 1, 2, 3], ["AAAA", "CCCC"], AGGREGATION
    )
    assert winner == 1
