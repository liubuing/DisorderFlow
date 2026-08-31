"""Group-aware confidence losses for BFN training (Direction B, V14+).

Extracted from core.py to keep the forward pass focused on the
BFN diffusion/receiver loop. All functions operate on (N,) tensors
with an optional scaffold_id for grouped (within-backbone) stats.
"""
import torch
import torch.nn.functional as F


def within_group_pair_mask(scaffold_id, n, device):
    """Boolean (N,N) mask: True where i,j belong to the same scaffold group.

    Returns None when scaffold_id is None (legacy/non-grouped batch) so callers
    can fall back to whole-batch behaviour.
    """
    if scaffold_id is None:
        return None
    sid = scaffold_id.to(device).view(-1)
    return sid.unsqueeze(0) == sid.unsqueeze(1)


def grouped_std(values, scaffold_id):
    """Mean of per-group stds of `values` (N,). Gradient flows through values.

    Maximising this (via the variance loss) pushes designs that share a backbone
    toward distinct confidences — the opposite of the constant-output failure.
    """
    if scaffold_id is None:
        return values.std()
    device = values.device
    sid = scaffold_id.to(device).view(-1)
    stds = []
    for g in torch.unique(sid):
        v = values[sid == g]
        if v.numel() > 1:
            stds.append(v.std())
    if not stds:
        return torch.zeros((), device=device)
    return torch.stack(stds).mean()


def grouped_margin_ranking(
        pred_iptm, af2_iptm, scaffold_id, margin=0.05,
        target_noise=None, sample_weight=None):
    """Pairwise margin-ranking of predicted ipTM against REAL af2_iptm labels,
    restricted to designs sharing a scaffold.

    For each within-group pair (i,j) with af2_iptm[i] > af2_iptm[j], penalises
    when pred_iptm[i] is not higher than pred_iptm[j] by at least `margin`.
    This is the supervisory signal that anchors the confidence head to real
    design-specific AF2 quality (absent in the synthetic-label dataset).
    """
    if scaffold_id is None:
        return torch.zeros((), device=pred_iptm.device)
    device = pred_iptm.device
    sid = scaffold_id.to(device).view(-1)
    total = torch.zeros((), device=device)
    weight_total = torch.zeros((), device=device)
    for g in torch.unique(sid):
        idx = (sid == g).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        p = pred_iptm[idx]      # (k,)
        a = af2_iptm[idx]       # (k,)
        # Pairs where a_i > a_j: target sign +1
        a_i, a_j = a.unsqueeze(1), a.unsqueeze(0)
        p_i, p_j = p.unsqueeze(1), p.unsqueeze(0)
        sign = torch.sign(a_i - a_j)
        valid = sign != 0
        if target_noise is not None:
            noise = target_noise.to(device).view(-1)[idx]
            threshold = torch.sqrt(
                noise.unsqueeze(1).square() + noise.unsqueeze(0).square())
            valid = valid & ((a_i - a_j).abs() > threshold)
        if valid.sum() == 0:
            continue
        # margin ranking loss: max(0, -sign * (p_i - p_j) + margin)
        loss = torch.clamp(-sign * (p_i - p_j) + margin, min=0.0)
        pair_weight = valid.float()
        if sample_weight is not None:
            weights = sample_weight.to(device).view(-1)[idx]
            pair_weight = pair_weight * torch.minimum(
                weights.unsqueeze(1), weights.unsqueeze(0))
        total = total + (loss * pair_weight).sum()
        weight_total = weight_total + pair_weight.sum()
    if weight_total <= 0:
        return torch.zeros((), device=device)
    return total / weight_total


def within_protein_disorder_ranking(
        pred_logits, target, mask, temperature=1.0, min_target_gap=0.05):
    """Pairwise residue ranking averaged equally across proteins.

    Only within-protein residue pairs with a meaningful target gap contribute.
    The target magnitude is never regressed; it determines ordering only.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if min_target_gap < 0:
        raise ValueError('min_target_gap must be non-negative')

    protein_losses = []
    for prediction, labels, valid_residues in zip(
            pred_logits, target, mask, strict=True):
        valid_residues = valid_residues.bool() & torch.isfinite(labels)
        indices = valid_residues.nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            continue
        prediction = prediction[indices]
        labels = labels[indices]
        label_delta = labels.unsqueeze(1) - labels.unsqueeze(0)
        prediction_delta = prediction.unsqueeze(1) - prediction.unsqueeze(0)
        pair_mask = torch.triu(
            label_delta.abs() >= min_target_gap, diagonal=1)
        if not pair_mask.any():
            continue
        direction = torch.sign(label_delta[pair_mask])
        ordered_delta = direction * prediction_delta[pair_mask]
        protein_losses.append(F.softplus(-ordered_delta / temperature).mean())

    if not protein_losses:
        return pred_logits.sum() * 0.0
    return torch.stack(protein_losses).mean()
