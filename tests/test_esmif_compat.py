import torch

from modules.esmif_compat import scatter, scatter_add


def test_scatter_add_matches_grouped_sum():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])
    observed = scatter_add(source, index, dim=0, dim_size=2)
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
    assert torch.equal(observed, expected)


def test_scatter_mean_matches_grouped_mean():
    source = torch.tensor([1.0, 3.0, 5.0])
    index = torch.tensor([0, 1, 0])
    assert torch.equal(scatter(source, index, dim=0, reduce="mean"), torch.tensor([3.0, 3.0]))
