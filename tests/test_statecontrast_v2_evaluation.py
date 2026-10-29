"""StateContrast v2 checkpoint 评估测试：验证可开发性硬失败的判定。

守护 scripts.evaluate_statecontrast_v2_checkpoint.hard_developability_failure：
N-糖基化 motif（N-V/T）与连续疏水 run 必须判硬失败、
含 DG/SG 的正常抗体序列不算失败——防止硬过滤漏检或误杀。
"""
from scripts.evaluate_statecontrast_v2_checkpoint import (
    hard_developability_failure,
)


# 硬失败判定：糖基化 motif 与疏水 run 必须拦截，含序模体（DGSG）的正常序列通过——
# 防止可开发性过滤器回归（漏检风险序列或误杀合格序列）。
def test_developability_hard_failure_detects_motif_and_hydrophobic_run():
    assert hard_developability_failure("ANVT") is True
    assert hard_developability_failure("AFFFFFA") is True
    assert hard_developability_failure("ARDGSGY") is False
