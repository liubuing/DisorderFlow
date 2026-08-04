from scripts.analyze_h3_t1_ensemble import exact_sign_flip_p


def test_exact_sign_flip_for_three_consistent_effects():
    assert exact_sign_flip_p([1.0, 1.0, 1.0]) == 0.25
