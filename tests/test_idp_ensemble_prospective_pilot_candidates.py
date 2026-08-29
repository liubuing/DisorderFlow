import numpy as np

from scripts.generate_idp_ensemble_prospective_pilot_candidates import (
    sample_h3,
)


def test_sample_h3_enforces_mutation_budget():
    probabilities = np.zeros((4, 21), dtype=float)
    probabilities[:, :2] = 0.5
    candidate = sample_h3("AAAA", probabilities, np.random.default_rng(1), 2)
    assert sum(left != right for left, right in zip("AAAA", candidate, strict=True)) == 2


def test_sample_h3_rejects_invalid_substitution_count():
    probabilities = np.zeros((4, 21), dtype=float)
    probabilities[:, :2] = 0.5
    with np.testing.assert_raises(ValueError):
        sample_h3("AAAA", probabilities, np.random.default_rng(1), 5)
