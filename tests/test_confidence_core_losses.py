import torch

from disorderflow.modules.bfn.core_losses import (
    candidate_antigen_pair_mask,
    grouped_margin_ranking,
    grouped_pair_difference,
)


def test_candidate_antigen_pair_mask_is_forward_only_by_default():
    candidate = torch.tensor([[True, False, False]])
    antigen = torch.tensor([[False, False, True]])
    mask = candidate_antigen_pair_mask(candidate, antigen)
    assert mask[0, 0, 2]
    assert not mask[0, 2, 0]
    assert mask.sum().item() == 1


def test_candidate_antigen_pair_mask_can_reproduce_v1_bidirectional_loss():
    candidate = torch.tensor([[True, False, False]])
    antigen = torch.tensor([[False, False, True]])
    mask = candidate_antigen_pair_mask(candidate, antigen, bidirectional=True)
    assert mask[0, 0, 2]
    assert mask[0, 2, 0]
    assert mask.sum().item() == 2


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


def test_grouped_pair_difference_breaks_exact_prediction_collapse():
    prediction = torch.zeros(3, requires_grad=True)
    target = torch.tensor([0.2, 0.5, 0.8])
    group = torch.tensor([0, 0, 0])
    sem = torch.tensor([0.01, 0.01, 0.01])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem)
    loss.backward()
    assert loss.item() > 0
    assert prediction.grad.abs().sum() > 0


def test_grouped_pair_difference_ignores_pairs_within_joint_sem():
    prediction = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.tensor([0.50, 0.51])
    group = torch.tensor([0, 0])
    sem = torch.tensor([0.02, 0.02])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem)
    assert loss.item() == 0.0


def test_normalized_grouped_pair_difference_breaks_prediction_collapse():
    prediction = torch.zeros(3, requires_grad=True)
    target = torch.tensor([0.20, 0.21, 0.23])
    group = torch.tensor([0, 0, 0])
    sem = torch.tensor([0.001, 0.001, 0.001])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert prediction.grad.abs().sum() > 0


def test_normalized_grouped_pair_difference_is_scale_invariant():
    prediction = torch.tensor([0.10, 0.15, 0.22])
    target = torch.tensor([0.20, 0.24, 0.31])
    group = torch.tensor([0, 0, 0])
    sem = torch.tensor([0.002, 0.003, 0.002])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    scaled_loss = grouped_pair_difference(
        prediction * 10, target * 10, group, target_noise=sem * 10,
        normalize_by_target_rms=True)
    assert torch.allclose(loss, scaled_loss)


def test_normalized_grouped_pair_difference_is_zero_at_exact_differences():
    target = torch.tensor([0.20, 0.24, 0.31])
    prediction = target.clone().requires_grad_(True)
    group = torch.tensor([0, 0, 0])
    loss = grouped_pair_difference(
        prediction, target, group, normalize_by_target_rms=True)
    assert loss.item() == 0.0


def test_normalized_grouped_pair_difference_handles_tiny_eligible_rms():
    prediction = torch.zeros(2, requires_grad=True)
    target = torch.tensor([0.0, 1e-7])
    group = torch.tensor([0, 0])
    sem = torch.zeros(2)
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_normalized_grouped_pair_difference_stays_connected_without_evidence():
    prediction = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.tensor([0.50, 0.51])
    group = torch.tensor([0, 0])
    sem = torch.tensor([0.02, 0.02])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert prediction.grad.abs().sum().item() == 0.0
