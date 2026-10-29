"""IdP 系综 ECLS 稳定性弃权分析的契约测试：守护共识选择对每次内部留一删除都稳定才可判胜，
以及内部胜者意见分裂时必须弃权（返回 None）——防止单一划分的偶然赢家冒充稳健结论。"""
import numpy as np

from scripts.analyze_idp_ensemble_ecls_stability_abstention import consensus_selection

# 聚合权重配置：p25/最小值/均值加权并扣标准差惩罚，用于驱动共识打分。
AGGREGATION = {
    "p25_weight": 0.45,
    "minimum_weight": 0.25,
    "mean_weight": 0.20,
    "standard_deviation_penalty": 0.10,
}


# 稳定赢家：每次内部删除划分下 STABLE 都优于 OTHER，共识必须选它且逐删除记录完整（4 行）。
def test_consensus_selects_candidate_stable_to_every_inner_deletion():
    matrix = np.asarray([
        [2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    winner, rows = consensus_selection(
        matrix, [0, 1, 2, 3], ["STABLE", "OTHER"], AGGREGATION
    )
    assert winner == 0
    assert len(rows) == 4


# 弃权守护：两个候选各赢一半内部划分（意见完全分裂）时必须弃权而非随机二选一，
# 防止不稳定的排序结论被强行输出。
def test_consensus_abstains_when_inner_winners_disagree():
    matrix = np.asarray([
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
    ])
    winner, _ = consensus_selection(
        matrix, [0, 1, 2, 3], ["LEFT", "RIGHT"], AGGREGATION
    )
    assert winner is None
