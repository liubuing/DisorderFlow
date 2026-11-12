import pytest

from ecls_scorer.scoring import (
    MPNN_ALPHABET,
    advantage_summary,
    composition_shuffles,
    ecls_delta,
    percentile_lower_is_better,
    sequence_nll,
)


def test_sequence_nll_uses_only_requested_positions():
    import numpy as np

    log_probs = np.full((5, len(MPNN_ALPHABET)), -10.0)
    log_probs[1, MPNN_ALPHABET.index("A")] = -1.0
    log_probs[3, MPNN_ALPHABET.index("C")] = -3.0
    assert sequence_nll(log_probs, "AC", [1, 3]) == 2.0


def test_sequence_nll_rejects_index_length_mismatch():
    import numpy as np

    log_probs = np.full((4, len(MPNN_ALPHABET)), -1.0)
    with pytest.raises(ValueError):
        sequence_nll(log_probs, "AC", [1])
    with pytest.raises(ValueError):
        sequence_nll(log_probs, "", [1])


def test_ecls_delta_is_complex_minus_apo():
    assert ecls_delta(5.0, 3.0) == 2.0
    assert ecls_delta(3.0, 5.0) == -2.0


def test_composition_shuffles_is_deterministic_and_unique():
    first = composition_shuffles("ACDEFGHKLM", 50, seed=2309)
    second = composition_shuffles("ACDEFGHKLM", 50, seed=2309)
    assert first == second
    assert len(first) == len(set(first))
    assert "ACDEFGHKLM" not in first
    assert len(first) == 50


def test_composition_shuffles_preserves_composition():
    sequence = "ACDEFGHKLM"
    for shuffle in composition_shuffles(sequence, 10, seed=1):
        assert sorted(shuffle) == sorted(sequence)


def test_composition_shuffles_capped_by_multiset_permutations():
    assert composition_shuffles("AAAA", 10, seed=1) == []
    assert len(composition_shuffles("AABB", 10, seed=1)) == 5  # 4!/(2!2!) - 1


def test_percentile_lower_is_better():
    assert percentile_lower_is_better(1.0, [0.5, 1.5, 2.0, 3.0]) == 0.75


def test_advantage_summary_arithmetic():
    summary = advantage_summary(2.0, [3.0, 5.0])
    assert summary["shuffle_count"] == 2
    assert summary["shuffle_mean_ecls"] == 4.0
    assert summary["ecls_advantage"] == 2.0
    assert summary["native_ecls_percentile"] == 1.0


def test_advantage_summary_requires_controls():
    with pytest.raises(ValueError):
        advantage_summary(2.0, [])
