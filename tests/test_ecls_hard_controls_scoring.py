import numpy as np
import pytest
import torch

from scripts.score_ecls_hard_controls import ecls_scores, rigid_transform


def test_rigid_transform_preserves_distances_and_changes_coordinates():
    x = torch.tensor([[1., 2., 3.], [-2., 0., 5.], [4., 3., 1.]])
    transformed = rigid_transform(x)
    assert not torch.equal(x, transformed)
    torch.testing.assert_close(torch.cdist(x, x), torch.cdist(transformed, transformed))


def test_ecls_uses_local_positions_and_correct_difference_direction():
    complex_logp = np.full((4, 21), -2.)
    apo_logp = np.full((4, 21), -3.)
    apo_logp[0] = -100
    np.testing.assert_allclose(ecls_scores(complex_logp, apo_logp, [1, 2], ["AC", "CA"]), [1, 1])


def test_length_mismatch_cannot_be_silently_truncated():
    with pytest.raises(ValueError):
        ecls_scores(np.zeros((3, 21)), np.zeros((3, 21)), [1, 2], ["A"])
