"""IdP 系综扩展打分器输入 v2 审计的契约测试：守护序列一致性计算对完全相同序列给 1.0、
对含替换序列严格小于 1.0——防止重复/近重复样本被当作独立输入。"""
from scripts.audit_idp_ensemble_expanded_scorer_inputs_v2 import (
    sequence_identity,
)


# 一致性锚点：相同序列为 1.0、单点替换后严格下降——防止实现恒返 1.0 使重复检测失效。
def test_sequence_identity_is_exact_for_equal_sequences():
    assert sequence_identity("ARDY", "ARDY") == 1.0
    assert sequence_identity("ARDY", "ARAY") < 1.0
