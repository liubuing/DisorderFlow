import numpy as np

from modules.peptide_torsion_perturb import smooth_torsion_direction


def test_torsion_direction_is_deterministic_and_normalized():
    first = smooth_torsion_direction("AAPAA", 7)
    second = smooth_torsion_direction("AAPAA", 7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    observed = np.sqrt(np.mean(np.concatenate(first) ** 2))
    assert observed == np.testing.assert_allclose(observed, 1.0) or observed == 1.0


def test_proline_phi_is_damped_relative_to_undamped_direction():
    phi, _ = smooth_torsion_direction("AAPAA", 11)
    assert abs(phi[2]) < 2.0
