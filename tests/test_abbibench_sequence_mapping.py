"""AbBiBench 序列映射的契约测试：守护 H3-only 坐标兼容分类、滑窗映射的歧义拒绝、
非规范/长度不符序列的拦截，以及结合亲和力数值解析对负值保留、缺失值排除的规则。"""
from scripts.audit_abbibench_sequence_mapping import classify, window_mapping
from scripts.crosswalk_abbibench_aayl_endpoints import finite_value


# 分类规则：仅 H3 区差异且轻链/框架完全一致才判为坐标兼容；轻链或 H3 外差异必须被拒绝，
# 防止结构对比在非同源序列上给出虚假结论。
def test_h3_only_requires_unchanged_light_and_framework():
    assert classify('ACDA', 'LLLL', 'ACDE', 'LLLL', [3]) == 'H3_only_coordinate_compatible'
    assert classify('ACDA', 'LLLA', 'ACDE', 'LLLL', [3]) == 'light_chain_differs'
    assert classify('CCDA', 'LLLL', 'ACDE', 'LLLL', [3]) == 'heavy_changes_outside_H3'
    assert classify('ACDE', 'LLLL', 'ACDE', 'LLLL', [3]) == 'identical_to_coordinate_sequence'


# 窗口映射必须拒绝歧义（目标多次出现）与越界查询（查询长于观测坐标），唯一匹配时给出正确偏移，
# 防止模糊对齐静默进入下游 H3 集合。
def test_mapping_rejects_ambiguous_and_unobserved_sequence():
    assert window_mapping('ACDEACDE', 'ACDE')['status'] == 'ambiguous_window'
    assert window_mapping('ACD', 'ACDE')['status'] == 'query_longer_than_observed_coordinates'
    assert window_mapping('YYACDEGG', 'ACDE')['offset'] == 2


# 质量门槛：含非规范残基或长度不符的序列不得进入 H3 数据集，防止脏数据污染基准。
def test_invalid_or_length_changed_sequences_do_not_enter_h3_set():
    assert classify('ACDX', 'LLLL', 'ACDE', 'LLLL', [3]) == 'noncanonical'
    assert classify('ACD', 'LLLL', 'ACDE', 'LLLL', [3]) == 'length_mismatch'


# 数值解析契约：负的 log 亲和力是合法科学数据必须保留；空串/NaN/inf 视为缺失并排除，
# 防止把缺失误当零或把负亲和力错误丢弃。
def test_original_log_affinity_keeps_negative_values_but_excludes_missing():
    assert finite_value('-0.4') == -.4
    assert finite_value('0') == 0
    assert finite_value('') is None
    assert finite_value('nan') is None
    assert finite_value('inf') is None
