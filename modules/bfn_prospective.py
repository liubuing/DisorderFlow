"""Successor-only batched BFN scoring without modifying frozen ECLS sources."""

from __future__ import annotations

import torch

from modules.bfn_loader import build_region_batch, inject_candidate_sequence, load_bfn


def inject_candidate_sequences(batch, candidate_sequences):
    """Expand one structural batch and inject one candidate per batch row."""
    candidates = [str(sequence).strip().upper() for sequence in candidate_sequences]
    if not candidates:
        raise ValueError("At least one candidate sequence is required")
    expanded = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            repeats = [len(candidates)] + [1] * (value.ndim - 1)
            expanded[key] = value.repeat(*repeats)
        elif isinstance(value, list) and len(value) == 1:
            expanded[key] = value * len(candidates)
        else:
            expanded[key] = value
    for row, candidate in enumerate(candidates):
        row_batch = {
            "aa": expanded["aa"][row:row + 1],
            "generate_flag": expanded["generate_flag"][row:row + 1],
        }
        inject_candidate_sequence(row_batch, candidate)
        expanded["aa"][row] = row_batch["aa"][0]
    return expanded


def score_bfn_candidates(
    pdb_path,
    region_spec,
    candidate_sequences,
    context_chains=None,
    device=None,
    fixed_t=0.5,
    model=None,
    antigen_chains=None,
):
    """Score supplied sequences against one fixed structural state."""
    if model is None:
        model, _ = load_bfn(device)
    if device is None:
        device = next(model.parameters()).device
    batch = build_region_batch(
        pdb_path,
        region_spec,
        context_chains=context_chains,
        device=device,
        antigen_chains=antigen_chains,
    )
    batch = inject_candidate_sequences(batch, candidate_sequences)
    with torch.no_grad():
        return model.score(batch, fixed_t=fixed_t)
