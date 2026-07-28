from pathlib import Path

import torch

from disorderflow.datasets.statecontrast_metadata import StateContrastMetadataDataset
from disorderflow.modules.bfn.contrastive_losses import grouped_contrastive_margin
from disorderflow.utils.data import PaddingCollate


def test_grouped_margin_rewards_correct_within_group_ordering():
    group_ids = torch.tensor([0, 0, 0, 1, 1])
    ranks = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    weights = torch.tensor([1.0, 0.2, 0.5, 1.0, 0.3])

    correct = grouped_contrastive_margin(
        torch.tensor([2.0, 0.0, -1.0, 1.0, -0.5]), group_ids, ranks, weights)
    inverted = grouped_contrastive_margin(
        torch.tensor([-1.0, 0.0, 2.0, -0.5, 1.0]), group_ids, ranks, weights)

    assert correct < inverted


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


def test_grouped_margin_respects_minimum_rank_gap():
    scores = torch.tensor([0.0, 2.0], requires_grad=True)
    loss = grouped_contrastive_margin(
        scores, torch.zeros(2), torch.tensor([0.51, 0.5]), torch.ones(2),
        min_rank_gap=0.1)
    loss.backward()
    assert loss.item() == 0.0


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


def test_core_uses_true_cb_atom_index_for_contact_labels():
    source = Path("disorderflow/modules/bfn/core.py").read_text(encoding="utf-8")
    assert "pos_heavyatom'][:, :, 4]" in source
    assert "mask_heavyatom'][:, :, 4]" in source
    assert "losses['grouped_contrastive'] = grouped_contrastive_margin(" in source
