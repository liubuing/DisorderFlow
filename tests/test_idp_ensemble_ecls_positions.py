import numpy as np

from scripts.analyze_idp_ensemble_ecls_positions import (
    candidate_state_profiles,
    contribution_mechanism,
    middle_comparator_records,
)


def test_position_profiles_sum_to_apo_minus_complex_mean_nll():
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    apo = np.full((2, len(alphabet)), -2.0)
    complex_state = np.full((2, len(alphabet)), -2.0)
    complex_state[0, alphabet.index("A")] = -1.0
    profile = candidate_state_profiles(apo, complex_state, "AC", [0, 1], alphabet)
    assert profile["gain"].sum() == 0.5
    assert np.allclose(profile["gain"], [0.5, 0.0])


def test_middle_comparator_retains_duplicate_observations():
    records = [
        {"training_conformer": 0, "candidate_index": 1, "heldout_score": 0.0},
        {"training_conformer": 1, "candidate_index": 2, "heldout_score": 1.0},
        {"training_conformer": 2, "candidate_index": 2, "heldout_score": 1.0},
        {"training_conformer": 3, "candidate_index": 3, "heldout_score": 2.0},
    ]
    selected, ordered, nonunique, alternatives = middle_comparator_records(records)
    assert [row["training_conformer"] for row in selected] == [1, 2]
    assert len(ordered) == 4
    assert nonunique is False
    assert alternatives == []


def test_negative_contribution_mechanism_separates_state_causes():
    assert contribution_mechanism(-0.08, -0.07, -0.01) == "apo_baseline_dominance"
    assert contribution_mechanism(0.02, 0.03, -0.01) == "complex_support_deficit"
