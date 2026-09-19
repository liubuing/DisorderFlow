import numpy as np
import pytest

from scripts.benchmark_ecls_pae_joint import joint_scores, screening


def test_screening_perfect_reverse_and_ties():
    target = np.arange(10)
    assert screening(target, target, .2) == 1
    assert screening(-target, target, .2) == 0
    assert screening(np.zeros(10), target, .2) == pytest.approx(.2)


def test_target_boundary_ties_are_fractional():
    assert screening([0, 1, 2, 3, 4], [0, 0, 1, 2, 3], .2) == .5


def test_joint_direction_and_scale_invariance():
    ecls = np.array([3., 1., 2.])
    pae = np.array([1., 3., 2.])
    np.testing.assert_equal(joint_scores(ecls, pae), [1, 3, 2])
    np.testing.assert_equal(joint_scores(ecls * 100 + 7, pae / 31), [1, 3, 2])


def test_invalid_screening_inputs_fail():
    with pytest.raises(ValueError):
        screening([float('nan')], [0], .2)
    with pytest.raises(ValueError):
        screening([0], [0, 1], .2)
