"""ECLS 硬对照打分测试：验证刚体变换、局部位置差分方向与长度不匹配防护。

守护 scripts.score_ecls_hard_controls 的 ECLS（复合物-Apo 对数概率差）打分：
刚体变换只改变坐标不改变距离矩阵、打分只用指定局部位置且方向为复合物-Apo、
logp 与序列长度不一致必须显式报错（禁止静默截断）。
"""
import numpy as np
import pytest
import torch

from scripts.score_ecls_hard_controls import ecls_scores, rigid_transform


# 刚体变换守护：坐标必须改变但成对距离矩阵保持不变——
# 防止变换实现引入缩放/剪切而破坏结构等价性。
def test_rigid_transform_preserves_distances_and_changes_coordinates():
    x = torch.tensor([[1., 2., 3.], [-2., 0., 5.], [4., 3., 1.]])
    transformed = rigid_transform(x)
    assert not torch.equal(x, transformed)
    torch.testing.assert_close(torch.cdist(x, x), torch.cdist(transformed, transformed))


# 差分方向与局部性：只用给定局部位置的复合物-Apo logp 差（复合物更高→得分 1），
# 非局部位置（被置为 -100 的第 0 行）不得参与——防止全局差或方向反转。
def test_ecls_uses_local_positions_and_correct_difference_direction():
    complex_logp = np.full((4, 21), -2.)
    apo_logp = np.full((4, 21), -3.)
    apo_logp[0] = -100
    np.testing.assert_allclose(ecls_scores(complex_logp, apo_logp, [1, 2], ["AC", "CA"]), [1, 1])


# 长度防护：logp 矩阵与序列长度不一致必须抛 ValueError——
# 防止静默截断把错位残基的 logp 当作有效证据。
def test_length_mismatch_cannot_be_silently_truncated():
    with pytest.raises(ValueError):
        ecls_scores(np.zeros((3, 21)), np.zeros((3, 21)), [1, 2], ["A"])
