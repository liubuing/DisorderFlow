"""IdP 系综 ECLS 位置级分析的契约测试：守护逐位置增益分解与 apo−complex 平均 NLL 的一致、
中间比较者记录保留重复观测，以及负贡献机制（apo 基线主导 vs complex 支持不足）的归因分离。"""
import numpy as np

from scripts.analyze_idp_ensemble_ecls_positions import (
    candidate_state_profiles,
    contribution_mechanism,
    middle_comparator_records,
)


# 位置分解守恒：增益 profile 逐位置之和必须等于全局 apo−complex 平均 NLL（0.5），
# 且只在实测改变的位置非零——防止位置级分解与端到端指标脱节。
def test_position_profiles_sum_to_apo_minus_complex_mean_nll():
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    apo = np.full((2, len(alphabet)), -2.0)
    complex_state = np.full((2, len(alphabet)), -2.0)
    complex_state[0, alphabet.index("A")] = -1.0
    profile = candidate_state_profiles(apo, complex_state, "AC", [0, 1], alphabet)
    assert profile["gain"].sum() == 0.5
    assert np.allclose(profile["gain"], [0.5, 0.0])


# 中位比较者：候选 2 的两条重复观测都要保留在排序记录中，防止去重导致证据计数与账目不符。
def test_middle_comparator_retains_duplicate_observations():
    records = [
        {"training_conformer": 0, "candidate_index": 1, "heldout_score": 0.0},
        {"training_conformer": 1, "candidate_index": 2, "heldout_score": 1.0},
        {"training_conformer": 2, "candidate_index": 2, "heldout_score": 1.0},
        {"training_conformer": 3, "candidate_index": 3, "heldout_score": 2.0},
    ]
    selected, ordered, nonunique, alternatives = middle_comparator_records(records)
    assert [row["training_conformer"] for row in selected] == [1, 2]
    assert len(ordered) == 4
    assert nonunique is False
    assert alternatives == []


# 机制归因：apo 项主导负贡献判 apo_baseline_dominance，complex 项不足判 complex_support_deficit，
# 防止两种失败模式被混为一谈而误判改进方向。
def test_negative_contribution_mechanism_separates_state_causes():
    assert contribution_mechanism(-0.08, -0.07, -0.01) == "apo_baseline_dominance"
    assert contribution_mechanism(0.02, 0.03, -0.01) == "complex_support_deficit"
