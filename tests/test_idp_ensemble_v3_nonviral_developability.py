"""IDP 集成 v3 非病毒可开发性代理测试：验证代理指标的值域边界。

守护 modules.state_specificity_scorer 的可开发性代理：
sequence_complexity 与 immunogenicity_proxy 都必须落在 [0,1]——
防止代理实现回归导致越界分数破坏下游阈值过滤。
"""
from modules.state_specificity_scorer import immunogenicity_proxy, sequence_complexity


# 值域守护：两个代理分数对典型抗体序列都必须有界——防止归一化回归导致越界。
def test_developability_proxies_are_bounded():
    assert 0.0 <= sequence_complexity("ARDYGGFDY") <= 1.0
    assert 0.0 <= immunogenicity_proxy("ARDYGGFDY") <= 1.0
