"""IdP 系综 v3 留一姿态打分的契约测试：守护指标方向函数的三值语义（越低越好为 -1、
越高越好为 +1、零增益为 0），防止打分排序把方向写反。"""
from scripts.score_idp_ensemble_v3_leave_one_pose_out import direction


# 方向锚点：正值 +1（越高越好）、负值 -1（PAE 等越低越好）、零为 0——方向反转是最危险的科学回归。
def test_direction_handles_positive_negative_and_zero():
    assert direction(0.5) == 1
    assert direction(-0.5) == -1
    assert direction(0.0) == 0
