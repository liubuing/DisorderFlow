"""多构象自回归采样器的契约测试：守护多 pose 对数概率的等权聚合、采样时每个 pose
按当前前缀独立重算、单一批式 pose 适配器的兼容性，以及空 pose 列表的拒绝。"""
import numpy as np

from scripts.autoregressive_multiconformer_sampler import (
    aggregate_pose_log_probs,
    sample_autoregressive,
)


# 测试替身：记录每次调用（前缀, 位置），并在指定位置给出占优 log 概率，
# 用于断言采样器确实按前缀逐位置查询每个 pose。
class FakePose:
    def __init__(self, preferred):
        self.preferred = preferred
        self.calls = []

    def next_log_probs(self, prefix, position):
        self.calls.append((prefix, position))
        values = np.full(3, -4.0)
        values[self.preferred[position]] = -0.1
        return values


# 聚合契约：默认各 pose 等权，log 概率取 logsumexp 后减 log(pose 数)——两个互补 pose 聚合应得 [-1, -1]，
# 防止聚合隐式偏向某个 pose 或误用算术平均。
def test_aggregate_pose_log_probs_is_equal_weight_by_default():
    result = aggregate_pose_log_probs([[0.0, -2.0], [-2.0, 0.0]])
    np.testing.assert_allclose(result, [-1.0, -1.0])


# 自回归正确性：每个 pose 在位置 0 与位置 1 都被重新查询，且位置 1 使用刚采出的真实前缀，
# 防止实现缓存过期 logits 或把同一前缀喂给所有位置。
def test_sampler_recomputes_each_pose_for_each_prefix():
    models = [FakePose([0, 1]), FakePose([1, 0])]
    sequence = sample_autoregressive(
        "AA", models, np.random.default_rng(3), 2, "ABC"
    )
    assert len(sequence) == 2
    assert models[0].calls == [("", 0), (sequence[0], 1)]
    assert models[1].calls == [("", 0), (sequence[0], 1)]


# 适配器兼容：单个暴露 pose_count 且返回批式 logits 的 pose 模型应被正确当作 3 个 pose 使用，
# 防止批式接口接入后采样逻辑崩坏。
def test_sampler_accepts_single_batched_pose_adapter():
    class Batched:
        pose_count = 3

        def next_log_probs(self, prefix, position):
            values = np.full(4, -3.0)
            values[0] = -0.2
            return values

    sequence = sample_autoregressive(
        "AA", [Batched()], np.random.default_rng(1), 1, "ABCD"
    )
    assert len(sequence) == 2


# 边界守护：空 pose 列表必须显式报错而非静默返回空序列或除零。
def test_sampler_rejects_empty_pose_models():
    import pytest

    with pytest.raises(ValueError, match="at least one pose model"):
        sample_autoregressive("AA", [], np.random.default_rng(1), 1, "ABCD")
