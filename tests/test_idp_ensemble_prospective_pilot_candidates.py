"""IDP 集成前瞻先导候选生成测试：验证 H3 采样严格遵循突变预算。

守护 scripts.generate_idp_ensemble_prospective_pilot_candidates.sample_h3：
采样候选与亲本的替换数必须恰好等于预算、非法替换数必须报错——
防止突变预算失控（过度突变破坏保守性约束）。
"""
import numpy as np

from scripts.generate_idp_ensemble_prospective_pilot_candidates import (
    sample_h3,
)


# 突变预算：预算为 2 时候选必须恰好有 2 处替换（不多不少）——防止采样器按概率多改或少改。
def test_sample_h3_enforces_mutation_budget():
    probabilities = np.zeros((4, 21), dtype=float)
    probabilities[:, :2] = 0.5
    candidate = sample_h3("AAAA", probabilities, np.random.default_rng(1), 2)
    assert sum(left != right for left, right in zip("AAAA", candidate, strict=True)) == 2


# 非法预算：替换数超过序列长度必须抛 ValueError——防止边界静默通过产生无效候选。
def test_sample_h3_rejects_invalid_substitution_count():
    probabilities = np.zeros((4, 21), dtype=float)
    probabilities[:, :2] = 0.5
    with np.testing.assert_raises(ValueError):
        sample_h3("AAAA", probabilities, np.random.default_rng(1), 5)
