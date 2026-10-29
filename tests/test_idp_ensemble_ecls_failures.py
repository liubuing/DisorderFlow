"""IDP 集成 ECLS 失败归因测试：验证效应分类、平局裁决与多样性度量。

守护 scripts.analyze_idp_ensemble_ecls_failures 的失败模式归因：
效应量必须区分为选择器恒等/打分等价/数值近平局/负迁移四类（防近平局被当真实效应）、
打分完全相同时胜者按字典序最大序列裁决（确定性平局处理）、
多样性同时报告原始重复率与最小 Hamming 距离并给出近克隆判断。
"""
import numpy as np
import pytest

from scripts.analyze_idp_ensemble_ecls_failures import (
    classify_effect,
    diversity_metrics,
    winner_index,
)

# 阈值 fixture：数学效应下限与实用中性绝对效应量的判定阈值。
THRESHOLDS = {
    "mathematical_effect_epsilon": 1e-9,
    "practical_neutral_absolute_effect": 1e-4,
}


# 效应分类：零效应+等选择器=selector_identity、零效应+不同选择器=score_equivalence、
# 微小正差=numerical_near_tie、显著负差=negative_transfer——防止近平局被误读为真实优势。
def test_effect_classification_separates_identity_equivalence_and_near_tie():
    assert classify_effect(0.0, 5, 5, THRESHOLDS) == "selector_identity"
    assert classify_effect(0.0, 1, 5, THRESHOLDS) == "score_equivalence"
    assert classify_effect(5e-6, 0, 5, THRESHOLDS) == "numerical_near_tie"
    assert classify_effect(-0.01, 0, 5, THRESHOLDS) == "negative_transfer"


# 确定性平局：打分矩阵完全相同时必须按字典序最大序列选胜者（CCCC）——
# 防止平局裁决依赖不稳定的遍历顺序。
def test_winner_tie_break_matches_lexicographically_greatest_sequence():
    matrix = np.ones((2, 5))
    aggregation = {
        "p25_weight": 0.45,
        "minimum_weight": 0.25,
        "mean_weight": 0.20,
        "standard_deviation_penalty": 0.10,
    }
    assert winner_index(matrix, list(range(5)), ["AAAA", "CCCC"], aggregation) == 1


# 多样性度量：原始/唯一候选数、重复率、最小成对 Hamming 与近克隆标志必须一致——
# 防止去重前后的口径混淆掩盖近克隆失败模式。
def test_diversity_reports_raw_duplicates_and_hamming_distance():
    result = diversity_metrics(["AAAA", "AAAC"], ["AAAA", "AAAA", "AAAC"])
    assert result["raw_candidates"] == 3
    assert result["unique_candidates"] == 2
    assert result["duplicate_fraction"] == pytest.approx(1 / 3)
    assert result["pairwise_hamming"]["minimum"] == 1
    assert result["near_clonal"] is True
