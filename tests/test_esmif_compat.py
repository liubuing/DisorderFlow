"""ESMIF 兼容层测试：验证 torch_scatter 缺失时的 scatter/scatter_add 纯 PyTorch 回退。

守护 modules.esmif_compat 的依赖降级不变量：
scatter_add 必须与按组求和逐位一致、scatter 的 mean 归约必须与分组均值一致——
防止回退实现与 torch_scatter 语义不一致而静默改变模型数值。
"""
import torch

from modules.esmif_compat import scatter, scatter_add


# scatter_add 回退必须与按索引分组的求和完全一致——防止梯度聚合错位。
def test_scatter_add_matches_grouped_sum():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])
    observed = scatter_add(source, index, dim=0, dim_size=2)
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
    assert torch.equal(observed, expected)


# scatter mean 归约必须与分组均值完全一致——防止池化统计错误传播到下游打分。
def test_scatter_mean_matches_grouped_mean():
    source = torch.tensor([1.0, 3.0, 5.0])
    index = torch.tensor([0, 1, 0])
    assert torch.equal(scatter(source, index, dim=0, reduce="mean"), torch.tensor([3.0, 3.0]))
