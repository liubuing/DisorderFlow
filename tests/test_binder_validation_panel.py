"""结合体验证面板测试：验证 CDR 组成打乱对照与界面 PAE 聚合的双向不变量。

守护结合体验证面板管线：
composition_scramble 只允许打乱 CDR 区位置（保持全局氨基酸组成与骨架不变）、
interface_pae 必须聚合跨链两个方向的 PAE——防止单向或全矩阵均值导致打分口径错误。
"""
from scripts.build_binder_validation_panel import CDR_SLICES, PANEL, composition_scramble
from scripts.parse_binder_validation_panel import interface_pae


# 组成打乱对照只改变 CDR 区内的残基（多重集不变且确实发生变化），
# CDR 之外逐位保持原样——防止阴性对照破坏骨架或引入组成偏差。
def test_composition_scramble_only_changes_cdr_positions():
    native = PANEL["3STB_native"]
    scrambled = composition_scramble(native, 4201)
    positions = {index for start, end in CDR_SLICES for index in range(start, end)}
    assert sorted(native[index] for index in positions) == sorted(scrambled[index] for index in positions)
    assert any(native[index] != scrambled[index] for index in positions)
    assert all(native[index] == scrambled[index] for index in range(len(native)) if index not in positions)


# 界面 PAE 必须同时取两条链间双向（row 与 column）的非对角均值，
# 防止只看单向或误含对角元素导致打分系统性偏差。
def test_interface_pae_uses_both_cross_chain_directions():
    score = {"pae": [[0, 2, 4], [6, 0, 8], [10, 12, 0]]}
    assert interface_pae(score, [2, 1]) == 8.5
