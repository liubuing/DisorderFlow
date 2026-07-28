import pytest

torch = pytest.importorskip('torch')

from disorderflow.modules.bfn.core_losses import within_protein_disorder_ranking


def test_correct_residue_order_has_lower_loss_than_reverse_order():
    target = torch.tensor([[0.0, 0.3, 0.7, 1.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    correct = within_protein_disorder_ranking(target * 5, target, mask)
    reverse = within_protein_disorder_ranking(-target * 5, target, mask)
    assert correct < reverse


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


def test_no_valid_pairs_returns_differentiable_zero():
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5]])
    loss = within_protein_disorder_ranking(prediction, target, target > -1)
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
