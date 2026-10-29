"""构象系综姿态迁移（Kabsch）的契约测试：守护刚体变换拟合的精确恢复（RMSD 趋零）
以及对点数不足输入的显式拒绝——保证跨构象坐标对齐的数值正确性。"""
import numpy as np

from ensemble_pose_transfer import kabsch_transform


# Kabsch 拟合必须精确恢复已知旋转+平移（RMSD < 1e-10），防止 SVD 符号/行列式错误导致的镜像解。
def test_kabsch_transform_recovers_rigid_transform():
    moving = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = moving @ rotation + np.array([4.0, -2.0, 7.0])
    fitted_rotation, translation, rmsd = kabsch_transform(moving, target)
    assert rmsd < 1e-10
    assert np.allclose(moving @ fitted_rotation + translation, target)


# 输入守护：少于 3 个点无法确定唯一刚体变换，必须报 ValueError 而非返回退化解。
def test_kabsch_transform_rejects_too_few_points():
    try:
        kabsch_transform(np.zeros((2, 3)), np.zeros((2, 3)))
    except ValueError as error:
        assert "three" in str(error)
    else:
        raise AssertionError("Expected a ValueError")
