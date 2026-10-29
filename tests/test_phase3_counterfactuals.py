"""Phase3 反事实条件化测试：验证 mismatched 臂保持事实分布的组成不变性。

守护 scripts.evaluate_v5_1_phase3.condition_batch 的反事实条件化：
mismatched 条件下非表位残基的 disorder profile 必须保持与事实值相同的多重集
（只做置换不引入新值）——防止反事实臂意外改变边缘分布而混淆对照。
"""
import numpy as np
import torch

from scripts.evaluate_v5_1_phase3 import condition_batch


# 反事实隔离：错配条件只置换 profile 值的顺序（排序后相等）但不保持原序——
# 防止条件化实现改值（分布漂移）或恒等不动（失去反事实意义）。
def test_mismatched_arm_preserves_factual_distribution():
    batch = {
        'epitope_disorder_profile': torch.tensor([[0.0, 0.1, 0.4, 0.9]]),
        'fragment_type': torch.tensor([[1, 3, 3, 3]]),
        'epitope_disorder': torch.tensor([[[0.0]]]),
    }
    factual = batch['epitope_disorder_profile'][0, 1:].clone()
    condition_batch(batch, 'mismatched', np.array([0.8, 0.1, 0.7, 0.2]))
    mismatch = batch['epitope_disorder_profile'][0, 1:]
    assert torch.allclose(factual.sort().values, mismatch.sort().values)
    assert not torch.allclose(factual, mismatch)
