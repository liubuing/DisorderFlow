def test_final_abstention_requires_both_stability_axes():
    combinations = [
        (source, leave_one_out, source and leave_one_out)
        for source in (False, True)
        for leave_one_out in (False, True)
    ]
    assert combinations == [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ]
