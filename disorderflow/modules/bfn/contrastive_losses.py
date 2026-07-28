"""Losses for matched observed/counterfactual antibody-antigen groups."""

import torch
import torch.nn.functional as F


def grouped_contrastive_margin(
        scores, group_ids, ranks, sample_weights, valid_samples=None, margin=0.5,
        min_rank_gap=1e-6):
    """Rank every ordered evidence pair within each complete group.

    Binary ranks retain the original behavior. Continuous ranks support ordered
    conformational states; ties and gaps below ``min_rank_gap`` are ignored.
    The requested score margin scales with the target-rank gap.
    """
    scores = scores.reshape(-1)
    group_ids = group_ids.to(scores.device).reshape(-1)
    ranks = ranks.to(scores.device).reshape(-1)
    sample_weights = sample_weights.to(scores.device).reshape(-1)
    if not (scores.numel() == group_ids.numel() == ranks.numel() == sample_weights.numel()):
        raise ValueError("Grouped contrastive tensors must have equal lengths")
    if margin < 0:
        raise ValueError("Grouped contrastive margin must be non-negative")
    if min_rank_gap < 0:
        raise ValueError("Grouped contrastive min_rank_gap must be non-negative")

    valid = (
        torch.isfinite(scores) & torch.isfinite(ranks) & torch.isfinite(sample_weights)
        & (sample_weights > 0)
    )
    if valid_samples is not None:
        valid = valid & valid_samples.to(scores.device).bool().reshape(-1)

    group_losses = []
    for group_id in torch.unique(group_ids[valid]):
        group = valid & (group_ids == group_id)
        group_scores = scores[group]
        group_ranks = ranks[group]
        group_weights = sample_weights[group]
        rank_gap = group_ranks.unsqueeze(1) - group_ranks.unsqueeze(0)
        ordered = rank_gap >= min_rank_gap
        if not ordered.any():
            continue
        score_gap = group_scores.unsqueeze(1) - group_scores.unsqueeze(0)
        pair_loss = F.softplus(margin * rank_gap[ordered] - score_gap[ordered])
        pair_weight = (
            group_weights.unsqueeze(1) * group_weights.unsqueeze(0)
        )[ordered]
        group_losses.append((pair_loss * pair_weight).sum() / pair_weight.sum())

    if not group_losses:
        return scores.sum() * 0.0
    return torch.stack(group_losses).mean()
