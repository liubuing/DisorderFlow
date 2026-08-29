from scripts.score_idp_ensemble_v3_leave_one_pose_out import direction


def test_direction_handles_positive_negative_and_zero():
    assert direction(0.5) == 1
    assert direction(-0.5) == -1
    assert direction(0.0) == 0
