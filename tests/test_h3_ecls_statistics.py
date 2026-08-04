from scripts.analyze_h3_ecls_statistics import leave_one_out_range, sign_flip_p


def test_exact_sign_flip_for_three_consistent_units():
    result = sign_flip_p([1.0, 1.0, 1.0])
    assert result["method"] == "exact_two_sided_sign_flip"
    assert result["p"] == 0.25


def test_leave_one_out_range():
    assert leave_one_out_range([1.0, 2.0, 3.0]) == [1.5, 2.5]
