import torch

from disorderflow.modules.bfn.core_losses import grouped_margin_ranking


def test_grouped_margin_ignores_differences_below_replicate_noise():
    prediction = torch.tensor([0.9, 0.1], requires_grad=True)
    target = torch.tensor([0.51, 0.50])
    group = torch.tensor([0, 0])
    noise = torch.tensor([0.02, 0.02])
    loss = grouped_margin_ranking(
        prediction, target, group, target_noise=noise)
    assert loss.item() == 0.0


def test_grouped_margin_weights_reliable_pairs():
    prediction = torch.tensor([0.1, 0.9], requires_grad=True)
    target = torch.tensor([0.8, 0.2])
    group = torch.tensor([0, 0])
    noise = torch.tensor([0.02, 0.02])
    weights = torch.tensor([1.0, 0.5])
    loss = grouped_margin_ranking(
        prediction, target, group, margin=0.05,
        target_noise=noise, sample_weight=weights)
    assert loss.item() > 0.0
    loss.backward()
    assert prediction.grad is not None


def test_grouped_margin_uses_quadrature_pair_uncertainty():
    prediction = torch.tensor([0.9, 0.1], requires_grad=True)
    target = torch.tensor([0.53, 0.50])
    group = torch.tensor([0, 0])
    sem = torch.tensor([0.025, 0.025])
    loss = grouped_margin_ranking(
        prediction, target, group, target_noise=sem)
    assert loss.item() == 0.0
