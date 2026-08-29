"""Differentiable state-contrast objectives for grouped conformational data."""

from __future__ import annotations

import torch
import torch.nn.functional as F


TARGET_STATE = 0
APO_STATE = 1
OFF_TARGET_STATE = 2


def source_constrained_pose_weights(
        quality_logits, source_ids, valid=None, minimum_source_mass=0.10,
        maximum_pose_weight=0.60, prior_weights=None):
    """Return normalized pose weights without rewarding source replication.

    A softmax first allocates mass between unique sources and then distributes
    each source's mass among its poses. Consequently, adding sampled replicas
    does not automatically outweigh a single experimental pose. Source floors
    and a pose cap prevent collapse onto one source or one easy conformer.
    """
    # Pose aggregation is numerically cheap and should remain fp32 under AMP;
    # source floors and KL are unstable and dtype-incompatible in bfloat16.
    logits = quality_logits.reshape(-1).float()
    sources = source_ids.to(logits.device).reshape(-1)
    valid = (torch.isfinite(logits) if valid is None else valid.to(logits.device).bool())
    if valid.numel() != logits.numel() or sources.numel() != logits.numel():
        raise ValueError("Pose-weight tensors must have equal lengths")
    weights = torch.zeros_like(logits)
    unique = torch.unique(sources[valid])
    if unique.numel() == 0:
        return weights, logits.sum() * 0.0
    floor = float(minimum_source_mass)
    if floor < 0 or floor * unique.numel() > 1.0 + 1e-8:
        raise ValueError("minimum_source_mass is infeasible for the source count")

    source_logits = torch.stack([
        torch.logsumexp(logits[valid & (sources == source)], dim=0)
        - torch.log(torch.tensor(
            int((valid & (sources == source)).sum()), device=logits.device,
            dtype=logits.dtype))
        for source in unique
    ])
    free_mass = max(0.0, 1.0 - floor * unique.numel())
    source_mass = floor + free_mass * torch.softmax(source_logits, dim=0)
    for index, source in enumerate(unique):
        mask = valid & (sources == source)
        weights[mask] = source_mass[index] * torch.softmax(logits[mask], dim=0)

    cap = float(maximum_pose_weight)
    if not 0 < cap <= 1:
        raise ValueError("maximum_pose_weight must be in (0, 1]")
    # A singleton state must carry weight one. More generally, the tightest
    # feasible cap is 1/n; relax only to that mathematical lower bound.
    cap = max(cap, 1.0 / int(valid.sum()))
    maximum = weights.max()
    if maximum > cap:
        # Blend toward uniform rather than hard-clipping, which would break the
        # sum-to-one constraint and remove useful gradients from capped poses.
        uniform = valid.to(weights.dtype) / valid.sum().clamp(min=1)
        alpha = ((maximum - cap) / (maximum - uniform.max()).clamp(min=1e-8)).clamp(0, 1)
        weights = (1 - alpha) * weights + alpha * uniform

    if prior_weights is None:
        prior = valid.to(weights.dtype) / valid.sum().clamp(min=1)
    else:
        prior = prior_weights.to(weights.device, weights.dtype).reshape(-1) * valid
        prior = prior / prior.sum().clamp(min=1e-8)
    kl = (weights[valid] * (
        torch.log(weights[valid].clamp(min=1e-8))
        - torch.log(prior[valid].clamp(min=1e-8))
    )).sum()
    return weights, kl


def grouped_source_constrained_pose_weights(
        quality_logits, group_ids, source_ids, valid=None,
        minimum_source_mass=0.10, maximum_pose_weight=0.60,
        prior_weights=None):
    """Apply source-constrained weighting independently inside each group."""
    logits = quality_logits.reshape(-1).float()
    groups = group_ids.to(logits.device).reshape(-1)
    sources = source_ids.to(logits.device).reshape(-1)
    valid = (
        torch.isfinite(logits) if valid is None
        else valid.to(logits.device).bool().reshape(-1)
    )
    weights = torch.zeros_like(logits)
    divergences = []
    for group in torch.unique(groups[valid]):
        member = valid & (groups == group)
        member_prior = None if prior_weights is None else prior_weights.reshape(-1)[member]
        group_weights, divergence = source_constrained_pose_weights(
            logits[member], sources[member], minimum_source_mass=minimum_source_mass,
            maximum_pose_weight=maximum_pose_weight, prior_weights=member_prior,
        )
        weights[member] = group_weights
        divergences.append(divergence)
    kl = torch.stack(divergences).mean() if divergences else logits.sum() * 0.0
    return weights, kl


def grouped_state_contrast_loss(
        scores, group_ids, state_types, pose_weights, sample_weights=None,
        apo_margin=0.10, off_target_margin=0.20, off_target_temperature=0.25):
    """Require target-state scores to exceed apo and off-target states."""
    scores = scores.reshape(-1)
    groups = group_ids.to(scores.device).reshape(-1)
    states = state_types.to(scores.device).reshape(-1)
    weights = pose_weights.to(scores.device, scores.dtype).reshape(-1)
    sample_weights = (
        torch.ones_like(scores) if sample_weights is None
        else sample_weights.to(scores.device, scores.dtype).reshape(-1)
    )
    if not (scores.numel() == groups.numel() == states.numel()
            == weights.numel() == sample_weights.numel()):
        raise ValueError("State-contrast tensors must have equal lengths")

    losses, target_gaps, used_groups = [], [], []
    for group in torch.unique(groups):
        member = groups == group
        target = member & (states == TARGET_STATE)
        apo = member & (states == APO_STATE)
        off = member & (states == OFF_TARGET_STATE)
        if not target.any() or (not apo.any() and not off.any()):
            continue
        target_weight = weights[target] * sample_weights[target]
        target_score = (scores[target] * target_weight).sum() / target_weight.sum().clamp(min=1e-8)
        group_terms = []
        negative_scores = []
        if apo.any():
            apo_weight = weights[apo] * sample_weights[apo]
            apo_score = (scores[apo] * apo_weight).sum() / apo_weight.sum().clamp(min=1e-8)
            group_terms.append(F.softplus(float(apo_margin) + apo_score - target_score))
            negative_scores.append(apo_score)
        if off.any():
            off_scores = scores[off]
            off_weights = (weights[off] * sample_weights[off]).clamp(min=1e-8)
            temperature = float(off_target_temperature)
            worst_off = temperature * torch.logsumexp(
                off_scores / temperature + torch.log(off_weights), dim=0
            ) - temperature * torch.log(off_weights.sum())
            group_terms.append(F.softplus(
                float(off_target_margin) + worst_off - target_score))
            negative_scores.append(worst_off)
        losses.append(torch.stack(group_terms).mean())
        target_gaps.append(target_score - torch.stack(negative_scores).max())
        used_groups.append(group)
    if not losses:
        zero = scores.sum() * 0.0
        return zero, {"mean_target_gap": zero.detach(), "valid_groups": 0}
    return torch.stack(losses).mean(), {
        "mean_target_gap": torch.stack(target_gaps).mean().detach(),
        "valid_groups": len(used_groups),
    }


def grouped_teacher_consistency_loss(
        model_scores, teacher_scores, group_ids, valid=None, tie_epsilon=1e-4,
        temperature=0.25):
    """Match an independent frozen teacher's within-group pair ordering."""
    model = model_scores.reshape(-1)
    teacher = teacher_scores.to(model.device, model.dtype).reshape(-1).detach()
    groups = group_ids.to(model.device).reshape(-1)
    valid = (
        torch.isfinite(model) & torch.isfinite(teacher)
        if valid is None else valid.to(model.device).bool()
    )
    losses = []
    for group in torch.unique(groups[valid]):
        member = valid & (groups == group)
        m = model[member]
        t = teacher[member]
        teacher_gap = t.unsqueeze(1) - t.unsqueeze(0)
        ordered = teacher_gap.abs() >= float(tie_epsilon)
        if not ordered.any():
            continue
        model_gap = m.unsqueeze(1) - m.unsqueeze(0)
        losses.append(F.softplus(
            -teacher_gap[ordered].sign() * model_gap[ordered] / float(temperature)
        ).mean())
    return torch.stack(losses).mean() if losses else model.sum() * 0.0
