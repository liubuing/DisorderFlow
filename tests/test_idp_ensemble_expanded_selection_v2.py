"""IDP 集成扩展选择 v2 测试：验证分辨率键的排序语义。

守护 scripts.select_idp_ensemble_expanded_development_v2.resolution_value：
数值分辨率正常解析、缺失值（"NA"/None）映射为 +inf 以排在最后——
防止缺失分辨率的组件在排序中被误当作最优。
"""
from scripts.select_idp_ensemble_expanded_development_v2 import resolution_value


# 缺失值排序语义：NA/None 必须大于任何实数分辨率（排最后）——
# 防止解析失败时抛异常或把缺失当作 0（最高优先）。
def test_resolution_value_sorts_missing_last():
    assert resolution_value("1.5") == 1.5
    assert resolution_value("NA") == float("inf")
    assert resolution_value(None) == float("inf")
