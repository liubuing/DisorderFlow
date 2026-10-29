"""蛋白内无序度排序损失测试：验证排序方向、掩码/平局/非有限目标隔离与蛋白间等权。

守护 disorderflow.modules.bfn.core_losses.within_protein_disorder_ranking：
预测顺序正确时损失必须低于反向、padding/平局/NaN 目标不产生梯度、
各蛋白独立排序后等权平均、无有效对时返回可反传的零损失。
"""
import pytest

torch = pytest.importorskip('torch')

from disorderflow.modules.bfn.core_losses import within_protein_disorder_ranking


# 排序方向：与目标同序的预测损失必须严格低于反向预测——
# 防止配对符号写反（损失奖励错误顺序）这一致命回归。
def test_correct_residue_order_has_lower_loss_than_reverse_order():
    target = torch.tensor([[0.0, 0.3, 0.7, 1.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    correct = within_protein_disorder_ranking(target * 5, target, mask)
    reverse = within_protein_disorder_ranking(-target * 5, target, mask)
    assert correct < reverse


# 无效目标隔离：padding 位、目标差小于 min_target_gap 的平局对、NaN 目标
# 都不得贡献梯度（对应位置 grad 为 0），有效位置仍有梯度——防止脏目标污染训练。
def test_padding_ties_and_nonfinite_targets_do_not_contribute():
    prediction = torch.tensor([[0.0, 9.0, 1.0, -20.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 1.0, float('nan')]])
    mask = torch.tensor([[True, False, True, True]])
    loss = within_protein_disorder_ranking(
        prediction, target, mask, min_target_gap=0.05)
    loss.backward()
    assert prediction.grad[0, 1] == 0
    assert prediction.grad[0, 3] == 0
    assert prediction.grad[0, :3].abs().sum() > 0


# 蛋白间独立且等权：批内联合损失必须等于各蛋白损失的平均值——
# 防止跨蛋白配对（跨序列比较无意义）或长序列主导梯度。
def test_proteins_are_ranked_independently_and_weighted_equally():
    target = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.3, 0.6, 1.0],
    ])
    prediction = torch.tensor([
        [1.0, 0.0, 100.0, -100.0],
        [0.0, 0.3, 0.6, 1.0],
    ])
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, True],
    ])
    combined = within_protein_disorder_ranking(prediction, target, mask)
    first = within_protein_disorder_ranking(
        prediction[:1], target[:1], mask[:1])
    second = within_protein_disorder_ranking(
        prediction[1:], target[1:], mask[1:])
    assert torch.allclose(combined, (first + second) / 2)


# 无有效对边界：损失为 0 但计算图必须保留（grad 非 None）——
# 防止提前 return 断掉梯度链导致该 batch 静默不更新。
def test_no_valid_pairs_returns_differentiable_zero():
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5]])
    loss = within_protein_disorder_ranking(prediction, target, target > -1)
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
