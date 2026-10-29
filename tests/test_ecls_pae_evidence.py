"""ECLS PAE 证据分析的契约测试：守护 top-k 成员资格的平局均分预算、平局预测只得机会水平信用、
教师信号取 H3 局部且方向正确的界面 PAE、歧义 H3 拒绝，以及近天然对照的组成守恒与两位置约束。"""
from collections import Counter

import numpy as np
import pytest

from scripts.analyze_ecls_pae_evidence import concordance, membership, teacher_value
from scripts.prepare_ecls_hard_controls import near_native_controls


# 平局处理：并列第 2 的三个成员按并列名次共享预算（各 2/3），总预算严格守恒——
# 防止平局被任意取舍导致证据计数不可复现。
def test_top_membership_ties_preserve_budget():
    weights = membership([0, 1, 1, 1, 2], 3)
    np.testing.assert_allclose(weights, [1, 2/3, 2/3, 2/3, 0])
    assert weights.sum() == pytest.approx(3)


# 一致性信用：完全并列的预测只能得 0.5（机会水平）而非 1.0，正确逆序得 0——
# 防止恒定/平局预测冒充判别能力。
def test_concordance_tied_predictions_get_chance_credit():
    assert concordance([1, 1, 1], [0, 1, 2]) == {"pairs": 3, "accuracy": .5}
    assert concordance([2, 1, 0], [0, 1, 2])["accuracy"] == 0


# 教师符号与范围：teacher_value 只取重链-抗原界面且限于 H3 区的 PAE（方向正确，得 0.5 而非被反向
# 或全局 PAE 混淆），防止教师信号被错误方向的配对或全链平均稀释。
def test_teacher_is_directional_local_h3_not_reverse_or_global():
    entity = dict(heavy_sequence="ACD", light_sequence="EF", antigen_sequence="GH", h3_sequence="CD")
    matrix = np.zeros((7, 7))
    matrix[1:3, 5:] = 15.5
    matrix[5:, 1:3] = 31
    assert teacher_value(matrix, entity) == .5


# 歧义拒绝：H3 序列在重链中出现多次时教师必须报错，防止局部化索引指向错误的区段。
def test_teacher_rejects_ambiguous_h3():
    with pytest.raises(ValueError):
        teacher_value(np.zeros((6, 6)), dict(heavy_sequence="AAA", light_sequence="C", antigen_sequence="DE", h3_sequence="AA"))


# 近天然对照：同种子可复现、数量达请求值、氨基酸组成严格守恒（Counter 相等）、
# 且每条只改恰好两个位置——保证"近天然"扰动幅度可控且不引入组成偏差。
def test_near_native_controls_preserve_composition_and_only_two_positions():
    seq = "ACDEFGHIKLMN"
    controls = near_native_controls(seq, 20, 7)
    assert controls == near_native_controls(seq, 20, 7)
    assert len({r['h3_sequence'] for r in controls}) == 20
    for row in controls:
        assert Counter(row['h3_sequence']) == Counter(seq)
        assert sum(a != b for a, b in zip(seq, row['h3_sequence'])) == 2


# 边界情形：全同聚体（homopolymer）无法构造不同的两位置交换对照，必须返回空列表而非近似凑数。
def test_homopolymer_has_no_distinct_swap_control():
    assert near_native_controls("AAAA", 20, 1) == []
