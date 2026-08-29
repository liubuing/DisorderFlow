import torch

from disorderflow.modules.statecontrast_v2 import (
    APO_STATE,
    OFF_TARGET_STATE,
    TARGET_STATE,
    grouped_source_constrained_pose_weights,
    grouped_state_contrast_loss,
    grouped_teacher_consistency_loss,
    source_constrained_pose_weights,
)
from disorderflow.utils.source_balanced_sampler import (
    SourceBalancedCompleteGroupBatchSampler,
)
from disorderflow.models.statecontrast_v2 import StateContrastV2Objective
from disorderflow.models.statecontrast_v2 import masked_mean
from disorderflow.models.statecontrast_v2 import AntibodyBFNStateContrastV2


def test_state_contrast_loss_rewards_target_above_negative_states():
    groups = torch.zeros(3, dtype=torch.long)
    states = torch.tensor([TARGET_STATE, APO_STATE, OFF_TARGET_STATE])
    weights = torch.ones(3)
    good, good_metrics = grouped_state_contrast_loss(
        torch.tensor([2.0, 0.0, -1.0], requires_grad=True),
        groups, states, weights,
    )
    bad_scores = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    bad, _ = grouped_state_contrast_loss(bad_scores, groups, states, weights)
    assert good < bad
    assert good_metrics["mean_target_gap"] > 0
    bad.backward()
    assert bad_scores.grad is not None
    assert bad_scores.grad.abs().sum() > 0


def test_pose_weights_balance_sources_and_remain_differentiable():
    logits = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    sources = torch.tensor([0, 0, 1])
    weights, kl = source_constrained_pose_weights(
        logits, sources, minimum_source_mass=0.25, maximum_pose_weight=0.60,
    )
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[sources == 1].sum() >= 0.25
    assert weights.max() <= 0.60 + 1e-6
    (weights[0] + kl).backward()
    assert logits.grad is not None


def test_pose_weights_accept_bfloat16_amp_logits():
    logits = torch.tensor([1.0, 0.0], dtype=torch.bfloat16, requires_grad=True)
    weights, kl = source_constrained_pose_weights(
        logits, torch.tensor([0, 1]), maximum_pose_weight=0.9)
    assert weights.dtype == torch.float32
    (weights[0] + kl).backward()
    assert logits.grad is not None


def test_grouped_pose_weights_normalize_each_state_group():
    weights, _ = grouped_source_constrained_pose_weights(
        torch.tensor([1.0, 0.0, 2.0, -1.0]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 0]),
        minimum_source_mass=0.1,
        maximum_pose_weight=0.9,
    )
    assert torch.isclose(weights[:2].sum(), torch.tensor(1.0))
    assert torch.isclose(weights[2:].sum(), torch.tensor(1.0))


def test_grouped_pose_weights_accept_bfloat16_logits():
    weights, _ = grouped_source_constrained_pose_weights(
        torch.tensor([1.0, 0.0], dtype=torch.bfloat16, requires_grad=True),
        torch.tensor([0, 0]), torch.tensor([0, 1]),
        maximum_pose_weight=0.9)
    assert weights.dtype == torch.float32


def test_masked_mean_returns_zero_for_apo_empty_antigen():
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pooled = masked_mean(values, torch.zeros((1, 2), dtype=torch.bool))
    assert torch.equal(pooled, torch.zeros((1, 2)))


def test_sequence_state_score_rewards_lower_candidate_nll():
    batch = {
        "aa": torch.tensor([[0, 1], [0, 1]]),
        "generate_flag": torch.ones((2, 2), dtype=torch.bool),
    }
    good = torch.full((2, 2, 22), -5.0)
    good[:, 0, 0] = 5.0
    good[:, 1, 1] = 5.0
    bad = torch.zeros((2, 2, 22))
    assert torch.all(
        AntibodyBFNStateContrastV2._sequence_scores(good, batch)
        > AntibodyBFNStateContrastV2._sequence_scores(bad, batch))


def test_teacher_consistency_ignores_ties_and_penalizes_inversion():
    groups = torch.zeros(3, dtype=torch.long)
    teacher = torch.tensor([2.0, 1.0, 1.0])
    correct = grouped_teacher_consistency_loss(
        torch.tensor([2.0, 0.0, -1.0]), teacher, groups)
    inverted = grouped_teacher_consistency_loss(
        torch.tensor([-1.0, 1.0, 2.0]), teacher, groups)
    assert correct < inverted


def test_combined_objective_backpropagates_state_and_pose_heads():
    objective = StateContrastV2Objective({
        "minimum_source_mass": 0.1,
        "maximum_pose_weight": 0.9,
    })
    state_scores = torch.tensor([1.0, 0.0, -0.5], requires_grad=True)
    pose_logits = torch.tensor([0.2, 0.1, -0.1], requires_grad=True)
    losses, weights = objective(state_scores, pose_logits, {
        "state_group_id": torch.tensor([0, 0, 0]),
        "state_type": torch.tensor([TARGET_STATE, APO_STATE, OFF_TARGET_STATE]),
        "pose_source_id": torch.tensor([0, 1, 1]),
        "state_sample_weight": torch.ones(3),
        "independent_teacher_score": torch.tensor([1.0, 0.2, -0.2]),
        "independent_teacher_valid": torch.ones(3, dtype=torch.bool),
    })
    total = losses["state_contrast_v2"] + 0.05 * losses["pose_weight_kl"]
    total = total + 0.2 * losses["independent_consistency"]
    total.backward()
    assert state_scores.grad is not None and state_scores.grad.abs().sum() > 0
    assert pose_logits.grad is not None
    assert torch.isclose(weights.sum(), torch.tensor(3.0))


class _Dataset:
    group_indices = ((0, 1), (2, 3), (4, 5), (6, 7))
    group_source_ids = ("a", "a", "b", "c")


def test_source_balanced_sampler_keeps_groups_complete_and_changes_epoch_order():
    sampler = SourceBalancedCompleteGroupBatchSampler(
        _Dataset(), max_batch_records=4, seed=7)
    epoch_batches = []
    for _epoch in range(5):
        epoch_batches.append(list(sampler))
    groups = [set(group) for group in _Dataset.group_indices]
    for batch in [row for epoch in epoch_batches for row in epoch]:
        for group in groups:
            assert group <= set(batch) or group.isdisjoint(batch)
    assert len({str(batches) for batches in epoch_batches}) > 1
