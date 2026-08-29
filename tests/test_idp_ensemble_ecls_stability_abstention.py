import numpy as np

from scripts.analyze_idp_ensemble_ecls_stability_abstention import consensus_selection

AGGREGATION = {
    "p25_weight": 0.45,
    "minimum_weight": 0.25,
    "mean_weight": 0.20,
    "standard_deviation_penalty": 0.10,
}


def test_consensus_selects_candidate_stable_to_every_inner_deletion():
    matrix = np.asarray([
        [2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    winner, rows = consensus_selection(
        matrix, [0, 1, 2, 3], ["STABLE", "OTHER"], AGGREGATION
    )
    assert winner == 0
    assert len(rows) == 4


def test_consensus_abstains_when_inner_winners_disagree():
    matrix = np.asarray([
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
    ])
    winner, _ = consensus_selection(
        matrix, [0, 1, 2, 3], ["LEFT", "RIGHT"], AGGREGATION
    )
    assert winner is None
