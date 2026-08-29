import numpy as np

from scripts.score_idp_ensemble_expanded_independent_v2 import minimum_distance


def test_minimum_distance_returns_closest_atom_pair():
    left = {"CA": np.array([0.0, 0.0, 0.0])}
    right = {
        "CA": np.array([5.0, 0.0, 0.0]),
        "CB": np.array([2.0, 0.0, 0.0]),
    }
    distance, pair = minimum_distance(left, right)
    assert distance == 2.0
    assert pair == ("CA", "CB")
