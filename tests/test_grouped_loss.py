"""Tests for the V14 grouped confidence losses (disorderflow/modules/bfn/core.py).

Validates the helper functions that implement:
  - within-group pair masking
  - grouped std (gradient-bearing variance signal)
  - grouped margin ranking against real AF2 ipTM labels

These run on CPU with tiny tensors; no model/checkpoint required.
"""
import pytest

torch = pytest.importorskip("torch")

from disorderflow.modules.bfn.core import (
    _grouped_margin_ranking,
    _grouped_std,
    _within_group_pair_mask,
)


def test_within_group_pair_mask():
    sid = torch.tensor([0, 0, 0, 1, 1])
    mask = _within_group_pair_mask(sid, 5, device="cpu")
    assert mask.shape == (5, 5)
    # Same-group pairs are True; cross-group are False.
    assert mask[0, 1] and mask[0, 2] and mask[3, 4]
    assert not mask[0, 3] and not mask[2, 4]
    # Diagonal is True (same id).
    assert mask[1, 1]


def test_within_group_mask_none_when_no_scaffold():
    assert _within_group_pair_mask(None, 4, device="cpu") is None


def test_grouped_std_uses_within_group_only():
    """Two groups, each internally spread → mean of within-group stds.

    torch.std uses Bessel correction (N-1): for a 2-element group [a,b],
    std = |a-b|/sqrt(2). So [0.1,0.9]->0.4*sqrt2≈0.5657 and [0.2,0.8]->0.4243.
    """
    values = torch.tensor([0.1, 0.9, 0.2, 0.8])  # group0 spread, group1 spread
    sid = torch.tensor([0, 0, 1, 1])
    std = _grouped_std(values, sid)
    expected = (0.4 * (2 ** 0.5) + 0.3 * (2 ** 0.5)) / 2  # ≈0.4950
    assert abs(std.item() - expected) < 1e-4


def test_grouped_std_ignores_cross_group():
    """If each group is internally constant, within-group std is 0 even though
    the whole-batch std is large."""
    values = torch.tensor([0.0, 0.0, 1.0, 1.0])
    sid = torch.tensor([0, 0, 1, 1])
    assert abs(_grouped_std(values, sid).item()) < 1e-6
    # But whole-batch (no scaffold) std is large.
    assert _grouped_std(values, None).item() > 0.5


def test_grouped_margin_ranking_zero_when_correctly_ordered():
    """If pred order matches af2 order within groups, loss is ~0."""
    pred = torch.tensor([0.9, 0.5, 0.8, 0.3])
    af2 = torch.tensor([0.9, 0.4, 0.8, 0.3])  # same ordering
    sid = torch.tensor([0, 0, 1, 1])
    loss = _grouped_margin_ranking(pred, af2, sid, margin=0.05)
    assert loss.item() < 0.02  # only the margin residual remains


def test_grouped_margin_ranking_penalises_wrong_order():
    """If a higher-af2 design gets lower pred, loss is large."""
    pred = torch.tensor([0.3, 0.9])   # pred inverted vs af2
    af2 = torch.tensor([0.9, 0.3])
    sid = torch.tensor([0, 0])
    loss = _grouped_margin_ranking(pred, af2, sid, margin=0.05)
    assert loss.item() > 0.5  # large penalty


def test_grouped_margin_ranking_isolated_no_cross_group():
    """Designs in different groups are never compared."""
    pred = torch.tensor([0.9, 0.1, 0.9, 0.1])  # inverted WITHIN each group
    af2 = torch.tensor([0.1, 0.9, 0.1, 0.9])   # but each group has both orders
    sid = torch.tensor([0, 0, 1, 1])
    loss = _grouped_margin_ranking(pred, af2, sid, margin=0.05)
    # Both groups are mis-ordered → loss large.
    assert loss.item() > 0.5
    # And it equals a single-group version (groups identical) within tolerance.
    loss_single = _grouped_margin_ranking(
        torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9]),
        torch.tensor([0, 0]), margin=0.05)
    assert abs(loss.item() - loss_single.item()) < 1e-4


def test_grouped_std_is_differentiable():
    """The variance signal must carry gradient (the V14 bug fix)."""
    pred = torch.tensor([0.3, 0.6, 0.4], requires_grad=True)
    sid = torch.tensor([0, 0, 0])
    std = _grouped_std(pred, sid)
    std.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0
