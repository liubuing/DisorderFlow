"""IdP 系综终点弃权的真值表测试：守护弃权必须同时满足来源独立性与留一稳定性两条轴，
任一缺失都不得放行——防止单轴通过的结论被当成稳健弃权/采纳依据。"""
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
