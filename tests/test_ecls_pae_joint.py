"""ECLS 与 PAE 联合打分测试：验证筛选指标的方向、平局分数化与尺度不变性。

守护 scripts.benchmark_ecls_pae_joint 的联合评估：
screening 对完美/反向/全平局预测分别给 1/0/随机期望、
边界平局给分数化 credit、joint_scores 只按秩合并两指标且对单调变换不变、
NaN 与长度不一致必须报错。
"""
import numpy as np
import pytest

from scripts.benchmark_ecls_pae_joint import joint_scores, screening


# 方向与平局基线：正向完美=1、反向=0、全平局=阈值比例（随机期望）——
# 防止平局处理偏向虚高分数。
def test_screening_perfect_reverse_and_ties():
    target = np.arange(10)
    assert screening(target, target, .2) == 1
    assert screening(-target, target, .2) == 0
    assert screening(np.zeros(10), target, .2) == pytest.approx(.2)


# 边界平局分数化：与阈值恰好并列的目标必须得 0.5 而非 0 或 1——
# 防止 top-k 边界比较符（<=/<）写反导致口径漂移。
def test_target_boundary_ties_are_fractional():
    assert screening([0, 1, 2, 3, 4], [0, 0, 1, 2, 3], .2) == .5


# 联合打分方向与尺度不变性：ECLS 高者优先、PAE 低者优先（科研方向：PAE 越低越好），
# 且对仿射缩放/平移保持秩不变——防止联合公式隐式依赖绝对数值尺度。
def test_joint_direction_and_scale_invariance():
    ecls = np.array([3., 1., 2.])
    pae = np.array([1., 3., 2.])
    np.testing.assert_equal(joint_scores(ecls, pae), [1, 3, 2])
    np.testing.assert_equal(joint_scores(ecls * 100 + 7, pae / 31), [1, 3, 2])


# 非法输入防护：NaN 分数或长度不一致必须抛 ValueError——防止脏数据静默进入筛选统计。
def test_invalid_screening_inputs_fail():
    with pytest.raises(ValueError):
        screening([float('nan')], [0], .2)
    with pytest.raises(ValueError):
        screening([0], [0, 1], .2)
