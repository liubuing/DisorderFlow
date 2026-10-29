"""表位无序度错配负样本管线的契约测试：守护负 profile 的位置保持与供体插值、
错配 profile 不得泄漏事实分布、位置路由对 profile 顺序敏感、直连位置头零初始化，
以及序列交叉熵正确排除禁用类别。"""
import numpy as np
import torch

from disorderflow.datasets.disorder_augmented import DisorderAugmentedDataset
from disorderflow.modules.bfn.core import sequence_cross_entropy_20
from disorderflow.modules.bfn.receiver import AntibodyBFN_Receiver


# 负样本构造：shuffled 版本必须逐位置保留原 profile 的多重集（片段内逆序），
# mismatched 版本取自供体蛋白——两种负例都不改变片段位置布局。
def test_negative_profiles_preserve_positions_and_interpolate_donor():
    wrapper = object.__new__(DisorderAugmentedDataset)
    wrapper._ids = ['a', 'b']
    wrapper._lookup = {'a': np.array([0.1]), 'b': np.array([0.7, 0.9])}
    wrapper._base = [None, None]
    data = {
        'epitope_disorder_profile': torch.zeros(4),
        'fragment_type': torch.tensor([1, 1, 3, 3]),
    }
    data['epitope_disorder_profile'][2:] = torch.tensor([0.2, 0.9])
    output = wrapper._add_negative_profiles(data, 0)
    assert torch.allclose(output['epitope_disorder_shuffled_profile'], torch.tensor([0.0, 0.0, 0.9, 0.2]))
    assert torch.allclose(output['epitope_disorder_mismatched_profile'], torch.tensor([0.0, 0.0, 0.2, 0.9]))


# 泄漏防护：错配 profile 与事实 profile 是彼此的重排（排序后相等但原序不同）——
# 保证负例只破坏"位置-无序度"对应关系而不引入新的边际分布信息。
def test_mismatched_profile_cannot_leak_factual_distribution():
    wrapper = object.__new__(DisorderAugmentedDataset)
    wrapper._ids = ['a', 'b']
    wrapper._lookup = {
        'a': np.array([0.1, 0.4, 0.9]),
        'b': np.array([0.9, 0.2, 0.7, 0.1]),
    }
    wrapper._base = [None, None]
    data = {
        'epitope_disorder_profile': torch.tensor([0.0, 0.1, 0.4, 0.9]),
        'fragment_type': torch.tensor([1, 3, 3, 3]),
    }
    output = wrapper._add_negative_profiles(data, 0)
    factual = output['epitope_disorder_profile'][1:]
    mismatch = output['epitope_disorder_mismatched_profile'][1:]
    assert torch.allclose(factual.sort().values, mismatch.sort().values)
    assert not torch.allclose(factual, mismatch)


# 顺序敏感性：调换 profile 中两位置的无序度值必须改变位置路由上下文——
# 防止路由实现忽略 profile 的位置对齐（等效于集合化）而学不到错配信号。
def test_position_routing_responds_to_profile_order():
    torch.manual_seed(3)
    receiver = AntibodyBFN_Receiver(
        res_feat_dim=8, pair_feat_dim=4, num_layers=1,
        encoder_opt={'pair_routing': True}, num_classes=22, head_dropout=0.0)
    pair = torch.randn(1, 4, 4, 4)
    mask = torch.tensor([[False, False, True, True]])
    first = receiver._position_disorder_context(pair, torch.tensor([[0.0, 0.0, 0.1, 0.9]]), mask)
    second = receiver._position_disorder_context(pair, torch.tensor([[0.0, 0.0, 0.9, 0.1]]), mask)
    assert not torch.allclose(first, second)


# 零初始化契约：直连位置路由头初始输出恒为零，保证训练从"纯 BFN 主干"平滑起步、不引入随机偏置。
def test_direct_position_head_is_zero_at_initialization():
    receiver = AntibodyBFN_Receiver(
        res_feat_dim=8, pair_feat_dim=4, num_layers=1,
        encoder_opt={'pair_routing': True, 'direct_position_routing': True},
        num_classes=22, head_dropout=0.0)
    context = torch.randn(1, 3, 8)
    assert torch.count_nonzero(receiver.disorder_position_head(context)) == 0


# 类别屏蔽：把第 20/21 类 logits 压到 -1e4 后，平滑交叉熵必须精确等于 log(20)——
# 禁用类别不得参与归一化，防止 22 类 softmax 泄漏到 20 类氨基酸空间。
def test_sequence_ce_excludes_disabled_classes():
    logits = torch.zeros(1, 2, 22)
    logits[..., 20:] = -1e4
    targets = torch.tensor([[0, 1]])
    loss = sequence_cross_entropy_20(logits, targets, label_smoothing=0.05)
    assert torch.allclose(loss, torch.full_like(loss, np.log(20.0)), atol=1e-5)
