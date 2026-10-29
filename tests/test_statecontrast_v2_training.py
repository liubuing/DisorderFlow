"""StateContrast v2 训练目标测试：验证状态对比、pose 权重约束、教师一致性与采样器。

守护 disorderflow StateContrastV2 训练栈：
状态对比损失必须奖励 target 高于 apo/off-target（方向性 + 梯度连通）、
pose 权重受来源质量下限与单 pose 权重上限约束且支持 bfloat16 AMP、
分组版逐状态组归一化、masked_mean 对空抗原返回零、
序列状态分奖励低候选 NLL、教师一致性忽略平局并惩罚倒序、
组合目标可同时反传状态与 pose 头、来源均衡采样器保持组完整且逐 epoch 变序。
"""
import torch

from disorderflow.modules.statecontrast_v2 import (
    APO_STATE,
    OFF_TARGET_STATE,
    TARGET_STATE,
    grouped_source_constrained_pose_weights,
    grouped_state_contrast_loss,
    grouped_teacher_consistency_loss,
    source_constrained_pose_weights,
)
from disorderflow.utils.source_balanced_sampler import (
    SourceBalancedCompleteGroupBatchSampler,
)
from disorderflow.models.statecontrast_v2 import StateContrastV2Objective
from disorderflow.models.statecontrast_v2 import masked_mean
from disorderflow.models.statecontrast_v2 import AntibodyBFNStateContrastV2


# 状态对比方向：target > apo > off-target 的排序损失更低且梯度可反传——
# 防止状态排序方向写反这一致命训练回归。
def test_state_contrast_loss_rewards_target_above_negative_states():
    groups = torch.zeros(3, dtype=torch.long)
    states = torch.tensor([TARGET_STATE, APO_STATE, OFF_TARGET_STATE])
    weights = torch.ones(3)
    good, good_metrics = grouped_state_contrast_loss(
        torch.tensor([2.0, 0.0, -1.0], requires_grad=True),
        groups, states, weights,
    )
    bad_scores = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    bad, _ = grouped_state_contrast_loss(bad_scores, groups, states, weights)
    assert good < bad
    assert good_metrics["mean_target_gap"] > 0
    bad.backward()
    assert bad_scores.grad is not None
    assert bad_scores.grad.abs().sum() > 0


# pose 权重约束：归一化和为 1、来源质量 >= minimum_source_mass、
# 单 pose 权重 <= maximum_pose_weight 且保持可微——防止权重塌缩到单一来源/单一 pose。
def test_pose_weights_balance_sources_and_remain_differentiable():
    logits = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    sources = torch.tensor([0, 0, 1])
    weights, kl = source_constrained_pose_weights(
        logits, sources, minimum_source_mass=0.25, maximum_pose_weight=0.60,
    )
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[sources == 1].sum() >= 0.25
    assert weights.max() <= 0.60 + 1e-6
    (weights[0] + kl).backward()
    assert logits.grad is not None


# AMP 兼容：bfloat16 logits 必须先升为 float32 参与约束且梯度回传不断——
# 防止混合精度训练下 dtype 错误或静默断梯度。
def test_pose_weights_accept_bfloat16_amp_logits():
    logits = torch.tensor([1.0, 0.0], dtype=torch.bfloat16, requires_grad=True)
    weights, kl = source_constrained_pose_weights(
        logits, torch.tensor([0, 1]), maximum_pose_weight=0.9)
    assert weights.dtype == torch.float32
    (weights[0] + kl).backward()
    assert logits.grad is not None


# 分组归一化：每个状态组的 pose 权重各自和为 1（组间互不挪用质量）——
# 防止某状态组抢占其他组的权重质量。
def test_grouped_pose_weights_normalize_each_state_group():
    weights, _ = grouped_source_constrained_pose_weights(
        torch.tensor([1.0, 0.0, 2.0, -1.0]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 0]),
        minimum_source_mass=0.1,
        maximum_pose_weight=0.9,
    )
    assert torch.isclose(weights[:2].sum(), torch.tensor(1.0))
    assert torch.isclose(weights[2:].sum(), torch.tensor(1.0))


# 分组版同样必须支持 bfloat16 输入并输出 float32——保证分组路径 AMP 安全。
def test_grouped_pose_weights_accept_bfloat16_logits():
    weights, _ = grouped_source_constrained_pose_weights(
        torch.tensor([1.0, 0.0], dtype=torch.bfloat16, requires_grad=True),
        torch.tensor([0, 0]), torch.tensor([0, 1]),
        maximum_pose_weight=0.9)
    assert weights.dtype == torch.float32


# 空掩码边界：抗原全被 mask 时 masked_mean 必须返回 0 而非 NaN——
# 防止 apo 状态（无抗原）样本污染损失为 NaN。
def test_masked_mean_returns_zero_for_apo_empty_antigen():
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pooled = masked_mean(values, torch.zeros((1, 2), dtype=torch.bool))
    assert torch.equal(pooled, torch.zeros((1, 2)))


# 序列状态分方向：候选残基 NLL 更低的 logits 必须得到更高序列分——
# 防止 logit→分数方向写反（把更差的序列当更好）。
def test_sequence_state_score_rewards_lower_candidate_nll():
    batch = {
        "aa": torch.tensor([[0, 1], [0, 1]]),
        "generate_flag": torch.ones((2, 2), dtype=torch.bool),
    }
    good = torch.full((2, 2, 22), -5.0)
    good[:, 0, 0] = 5.0
    good[:, 1, 1] = 5.0
    bad = torch.zeros((2, 2, 22))
    assert torch.all(
        AntibodyBFNStateContrastV2._sequence_scores(good, batch)
        > AntibodyBFNStateContrastV2._sequence_scores(bad, batch))


# 教师一致性：与教师排序一致的损失更低；教师平局不得产生梯度影响——
# 防止一致性项把平局噪声当监督或方向写反。
def test_teacher_consistency_ignores_ties_and_penalizes_inversion():
    groups = torch.zeros(3, dtype=torch.long)
    teacher = torch.tensor([2.0, 1.0, 1.0])
    correct = grouped_teacher_consistency_loss(
        torch.tensor([2.0, 0.0, -1.0]), teacher, groups)
    inverted = grouped_teacher_consistency_loss(
        torch.tensor([-1.0, 1.0, 2.0]), teacher, groups)
    assert correct < inverted


# 组合目标接线：state/pose/一致性三项损失的加权和必须能同时反传到
# 状态分与 pose logits，且 pose 权重按样本数归一（和=3）——防止损失项未接线或权重归一错误。
def test_combined_objective_backpropagates_state_and_pose_heads():
    objective = StateContrastV2Objective({
        "minimum_source_mass": 0.1,
        "maximum_pose_weight": 0.9,
    })
    state_scores = torch.tensor([1.0, 0.0, -0.5], requires_grad=True)
    pose_logits = torch.tensor([0.2, 0.1, -0.1], requires_grad=True)
    losses, weights = objective(state_scores, pose_logits, {
        "state_group_id": torch.tensor([0, 0, 0]),
        "state_type": torch.tensor([TARGET_STATE, APO_STATE, OFF_TARGET_STATE]),
        "pose_source_id": torch.tensor([0, 1, 1]),
        "state_sample_weight": torch.ones(3),
        "independent_teacher_score": torch.tensor([1.0, 0.2, -0.2]),
        "independent_teacher_valid": torch.ones(3, dtype=torch.bool),
    })
    total = losses["state_contrast_v2"] + 0.05 * losses["pose_weight_kl"]
    total = total + 0.2 * losses["independent_consistency"]
    total.backward()
    assert state_scores.grad is not None and state_scores.grad.abs().sum() > 0
    assert pose_logits.grad is not None
    assert torch.isclose(weights.sum(), torch.tensor(3.0))


# fixture：带 4 个组与来源标签的最小数据集桩。
class _Dataset:
    group_indices = ((0, 1), (2, 3), (4, 5), (6, 7))
    group_source_ids = ("a", "a", "b", "c")


# 来源均衡采样器：每批要么完整包含某组要么完全不含（组不跨批）、
# 5 个 epoch 的批顺序不完全相同（逐 epoch 变序）——防止组被拆散或每轮顺序僵化。
def test_source_balanced_sampler_keeps_groups_complete_and_changes_epoch_order():
    sampler = SourceBalancedCompleteGroupBatchSampler(
        _Dataset(), max_batch_records=4, seed=7)
    epoch_batches = []
    for _epoch in range(5):
        epoch_batches.append(list(sampler))
    groups = [set(group) for group in _Dataset.group_indices]
    for batch in [row for epoch in epoch_batches for row in epoch]:
        for group in groups:
            assert group <= set(batch) or group.isdisjoint(batch)
    assert len({str(batches) for batches in epoch_batches}) > 1
