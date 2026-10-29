"""组内对比边际损失测试：验证组内排序方向、平局隔离、最小秩差与元数据封装。

守护 disorderflow.modules.bfn.contrastive_losses.grouped_contrastive_margin：
同组内高分必须对应高 contrastive_rank（方向反了损失必增）、
单样本组返回保持梯度链的零损失、平局与秩差不足的对不产生梯度、
StateContrastMetadataDataset 的元数据经 PaddingCollate 后仍完整进入 batch。
"""
from pathlib import Path

import torch

from disorderflow.datasets.statecontrast_metadata import StateContrastMetadataDataset
from disorderflow.modules.bfn.contrastive_losses import grouped_contrastive_margin
from disorderflow.utils.data import PaddingCollate


# 排序方向：组内按 rank 正确排序的分数损失必须低于倒序——防止对比损失方向写反。
def test_grouped_margin_rewards_correct_within_group_ordering():
    group_ids = torch.tensor([0, 0, 0, 1, 1])
    ranks = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    weights = torch.tensor([1.0, 0.2, 0.5, 1.0, 0.3])

    correct = grouped_contrastive_margin(
        torch.tensor([2.0, 0.0, -1.0, 1.0, -0.5]), group_ids, ranks, weights)
    inverted = grouped_contrastive_margin(
        torch.tensor([-1.0, 0.0, 2.0, -0.5, 1.0]), group_ids, ranks, weights)

    assert correct < inverted


# 可微性与单样本边界：正常组损失可反传；单样本组返回 0 但梯度链不断——
# 防止空对/单元素组导致 ValueError 中断训练或静默断梯度。
def test_grouped_margin_is_differentiable_and_singletons_return_connected_zero():
    scores = torch.tensor([0.2, -0.1], requires_grad=True)
    loss = grouped_contrastive_margin(
        scores, torch.tensor([4, 4]), torch.tensor([1.0, 0.0]), torch.ones(2))
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0

    singleton = torch.tensor([0.3], requires_grad=True)
    zero = grouped_contrastive_margin(
        singleton, torch.tensor([1]), torch.tensor([1.0]), torch.tensor([1.0]))
    zero.backward()
    assert zero.item() == 0.0
    assert singleton.grad.item() == 0.0


# 多级状态与平局：支持三个有序状态级（>2 级）、倒序惩罚更大、
# 完全平局的组零损失且零梯度——防止平局噪声注入虚假训练信号。
def test_grouped_margin_supports_three_ordered_state_levels_and_ignores_ties():
    group_ids = torch.zeros(4, dtype=torch.long)
    ranks = torch.tensor([1.0, 0.5, 0.0, 0.5])
    weights = torch.ones(4)
    ordered = grouped_contrastive_margin(
        torch.tensor([2.0, 1.0, 0.0, 1.0]), group_ids, ranks, weights)
    inverted = grouped_contrastive_margin(
        torch.tensor([0.0, 1.0, 2.0, -4.0]), group_ids, ranks, weights)
    assert ordered < inverted

    tied_scores = torch.tensor([0.2, -9.0], requires_grad=True)
    tied = grouped_contrastive_margin(
        tied_scores, torch.zeros(2), torch.ones(2), torch.ones(2))
    tied.backward()
    assert tied.item() == 0.0
    assert tied_scores.grad.abs().sum().item() == 0.0


# 最小秩差：rank 差小于 min_rank_gap 的对必须被忽略（零损失）——
# 防止测量噪声级的秩差驱动无意义的梯度更新。
def test_grouped_margin_respects_minimum_rank_gap():
    scores = torch.tensor([0.0, 2.0], requires_grad=True)
    loss = grouped_contrastive_margin(
        scores, torch.zeros(2), torch.tensor([0.51, 0.5]), torch.ones(2),
        min_rank_gap=0.1)
    loss.backward()
    assert loss.item() == 0.0


# 元数据封装：group_id/rank/weight 必须经 PaddingCollate 完整进入 batch 且 scaffold 归零正确——
# 防止 collate 丢弃对比监督字段导致损失静默失效。
def test_metadata_wrapper_survives_padding_collate():
    base = [
        {"aa": torch.tensor([1, 2]), "generate_flag": torch.tensor([True, False])},
        {"aa": torch.tensor([1, 2, 3]), "generate_flag": torch.tensor([True, True, False])},
    ]
    records = [
        {"group_id": "same", "targets": {"contrastive_rank": 1, "training_weight": 1.0}},
        {"group_id": "same", "targets": {"contrastive_rank": 0, "training_weight": 0.2}},
    ]
    wrapped = StateContrastMetadataDataset(base, records)
    batch = PaddingCollate(eight=False)([wrapped[0], wrapped[1]])

    assert batch["contrastive_group_id"].tolist() == [0, 0]
    assert batch["contrastive_rank"].tolist() == [1.0, 0.0]
    assert batch["contrastive_weight"].shape == (2,)
    assert wrapped._scaffold_ids == [0, 0]


# 源码级守护：接触标签必须使用真实的 CB 重原子索引（index 4）且对比损失已接入 core——
# 防止实现回退到 CA 或丢失损失接线这一类难以用数值测试发现的回归。
def test_core_uses_true_cb_atom_index_for_contact_labels():
    source = Path("disorderflow/modules/bfn/core.py").read_text(encoding="utf-8")
    assert "pos_heavyatom'][:, :, 4]" in source
    assert "mask_heavyatom'][:, :, 4]" in source
    assert "losses['grouped_contrastive'] = grouped_contrastive_margin(" in source
