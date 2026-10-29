"""IDP 集成 ECLS v2 jackknife 下包络测试：验证单构象依赖惩罚与确定性平局。

守护 scripts.analyze_idp_ensemble_ecls_v2.jackknife_lower_envelope：
jackknife 逐构象留一的 lower-envelope 聚合必须惩罚只靠单一构象得分的候选
（ROBUST 胜出）、打分全相同时按字典序最大序列裁决——防止平局裁决不确定。
"""
import numpy as np

from scripts.analyze_idp_ensemble_ecls_v2 import jackknife_lower_envelope

# 聚合权重 fixture：p25/最小值/均值/标准差惩罚的加权配置。
AGGREGATION = {
    "p25_weight": 0.45,
    "minimum_weight": 0.25,
    "mean_weight": 0.20,
    "standard_deviation_penalty": 0.10,
}


# 单构象依赖惩罚：只在一个构象上得分高的候选（SPIKE）必须输给均衡候选（ROBUST），
# 且留一内层评分完整记录——防止集成胜者实际依赖单一构象（过拟合构象）。
def test_jackknife_lower_envelope_penalizes_single_conformer_dependence():
    matrix = np.asarray([
        [10.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    winner, scores, inner = jackknife_lower_envelope(
        matrix, [0, 1, 2, 3], ["SPIKE", "ROBUST"], AGGREGATION
    )
    assert winner == 1
    assert scores[0] < scores[1]
    assert len(inner[0]) == 4


# 确定性平局：全同打分时胜者必须是字典序最大序列（CCCC）——防止平局结果不可复现。
def test_jackknife_tie_break_is_lexicographically_greatest():
    matrix = np.ones((2, 4))
    winner, _, _ = jackknife_lower_envelope(
        matrix, [0, 1, 2, 3], ["AAAA", "CCCC"], AGGREGATION
    )
    assert winner == 1
