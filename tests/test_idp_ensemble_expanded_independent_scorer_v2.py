"""IDP 集成扩展独立打分器 v2 测试：验证原子间最小距离的计算正确性。

守护 scripts.score_idp_ensemble_expanded_independent_v2.minimum_distance：
两组原子字典间的最小距离必须取全局最近原子对并返回其原子名——
防止接口距离取均值/错误原子导致接触判断失真。
"""
import numpy as np

from scripts.score_idp_ensemble_expanded_independent_v2 import minimum_distance


# 最小距离语义：应取 CB-CA 对（距离 2.0）而非更远的同名原子对——
# 防止遍历顺序或键匹配错误导致距离取错原子。
def test_minimum_distance_returns_closest_atom_pair():
    left = {"CA": np.array([0.0, 0.0, 0.0])}
    right = {
        "CA": np.array([5.0, 0.0, 0.0]),
        "CB": np.array([2.0, 0.0, 0.0]),
    }
    distance, pair = minimum_distance(left, right)
    assert distance == 2.0
    assert pair == ("CA", "CB")
