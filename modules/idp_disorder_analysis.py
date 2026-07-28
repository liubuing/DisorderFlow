#!/usr/bin/env python
"""IDP Disorder Analysis — identify ordered segments using BFN disorder head.

Core capability unique to BFN: per-residue disorder prediction during
sequence design. For intrinsically disordered proteins (IDPs), this
identifies which segments are ordered enough for structure-based design.

Disorder < 0.3 → "ordered" (suitable for antibody epitope design)
Disorder >= 0.3 → "disordered" (unreliable for structure-based methods)
"""

import torch
import torch.nn.functional as F
import numpy as np

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def predict_disorder(model, config, pdb_path, chain_id='A', device='cuda'):
    """Run BFN disorder prediction on a protein structure.

    Uses mask_region with no design regions so the model processes the
    entire protein as context-only. Disorder head outputs per-residue
    logits which we convert to probabilities via sigmoid.

    Args:
        model: Loaded BFN model (must have disorder head)
        config: BFN config
        pdb_path: Path to protein PDB file
        chain_id: Chain to analyze
        device: torch device

    Returns:
        dict with keys:
            disorder_scores: (L,) float32 array, per-residue disorder prob [0,1]
            sequence: str, amino acid sequence
            residue_ids: list of int, 1-indexed residue numbers
    """
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform

    sampling_config = getattr(config, 'sampling', None)
    seed_all(getattr(sampling_config, 'seed', 42))

    structure = preprocess_protein_structure(pdb_path, chain_ids=[chain_id])
    if structure is None:
        raise ValueError(f"Cannot parse structure: {pdb_path}")

    transform = get_transform([
        {'type': 'mask_region', 'regions': {}},  # no design regions, context-only
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])

    batch = recursive_to(PaddingCollate()([transform(structure)]), device)
    mask_res = batch['mask'][0].bool()
    n_valid = mask_res.sum().item()

    sample_opt = {'deterministic': True, 'return_disorder': True, 'num_recycles': 1}
    with torch.no_grad():
        traj = model.sample(batch, sample_opt=sample_opt)

    if 'disorder' not in traj:
        raise RuntimeError(
            "Model does not have a disorder head. "
            "The checkpoint must be trained with disorder loss weight > 0."
        )

    disorder_logits = traj['disorder'][0][mask_res]
    disorder_scores = torch.sigmoid(disorder_logits).cpu().numpy()

    aa = batch['aa'][0][mask_res]
    sequence = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in aa.cpu())

    # Extract residue IDs from structure seqmap
    residue_ids = _extract_residue_ids(structure, chain_id, n_valid)

    return {
        'disorder_scores': disorder_scores.astype(np.float32),
        'sequence': sequence,
        'residue_ids': residue_ids,
    }


def _extract_residue_ids(structure, chain_id, expected_len):
    """Extract 1-indexed residue numbers from structure data."""
    for chain in structure['chains']:
        if chain['chain_id'] == chain_id:
            data = chain['data']
            if hasattr(data, 'resseq') and len(data.resseq) >= expected_len:
                return [int(r) for r in data.resseq[:expected_len].tolist()]
    return list(range(1, expected_len + 1))


def find_ordered_segments(disorder_scores, residue_ids=None, threshold=0.3,
                          min_window=8, gap_tolerance=2):
    """Find contiguous ordered segments from per-residue disorder scores.

    A residue is "ordered" if its disorder score is below the threshold.
    Gaps of up to gap_tolerance disordered residues within an otherwise
    ordered stretch are tolerated (bridged into the segment).

    Args:
        disorder_scores: (L,) array of disorder probabilities [0,1]
        residue_ids: optional list of 1-indexed residue IDs
        threshold: max disorder score to be considered "ordered"
        min_window: minimum number of ordered residues in a segment
        gap_tolerance: max consecutive disordered residues to bridge

    Returns:
        list of dicts sorted by (mean_disorder asc, length desc):
            start, end: 1-indexed residue boundaries
            length: count of ordered residues in segment
            total_span: end - start + 1 (includes gaps)
            mean_disorder: average disorder score
            residues: list of (resid, disorder_score) tuples
    """
    if residue_ids is None:
        residue_ids = list(range(1, len(disorder_scores) + 1))

    scores = np.asarray(disorder_scores, dtype=np.float64)
    n = len(scores)
    ordered = scores < threshold

    segments = []
    i = 0
    while i < n:
        if not ordered[i]:
            i += 1
            continue

        seg_start = i
        gaps = 0
        j = i
        while j < n:
            if ordered[j]:
                gaps = 0
                j += 1
            elif gaps < gap_tolerance and j + 1 < n and ordered[j + 1]:
                gaps += 1
                j += 1
            else:
                break
        seg_end = j  # exclusive

        ordered_count = int(ordered[seg_start:seg_end].sum())
        if ordered_count >= min_window:
            seg_scores = scores[seg_start:seg_end]
            segments.append({
                'start': residue_ids[seg_start],
                'end': residue_ids[seg_end - 1],
                'length': ordered_count,
                'total_span': seg_end - seg_start,
                'mean_disorder': float(seg_scores.mean()),
                'min_disorder': float(seg_scores.min()),
                'max_disorder': float(seg_scores.max()),
                'residues': [(residue_ids[k], float(scores[k]))
                             for k in range(seg_start, seg_end)],
            })
        i = seg_end

    segments.sort(key=lambda s: (s['mean_disorder'], -s['length']))
    return segments


def format_disorder_report(disorder_scores, sequence, residue_ids=None,
                           threshold=0.3, top_n=60):
    """Human-readable disorder analysis report."""
    if residue_ids is None:
        residue_ids = list(range(1, len(disorder_scores) + 1))

    scores = np.asarray(disorder_scores)
    n_ordered = int((scores < threshold).sum())

    lines = [
        "=" * 70,
        "  BFN Disorder Analysis",
        "=" * 70,
        f"  Residues: {len(scores)}",
        f"  Mean disorder: {scores.mean():.4f}",
        f"  Ordered (D < {threshold}): {n_ordered}/{len(scores)}",
        f"  Disordered (D >= {threshold}): {len(scores) - n_ordered}/{len(scores)}",
        "",
        f"  Per-residue scores (top {top_n} most ordered):",
    ]

    sorted_idx = np.argsort(scores)
    for rank, idx in enumerate(sorted_idx[:top_n]):
        tag = " [ORDERED]" if scores[idx] < threshold else ""
        lines.append(
            f"  #{rank+1:2d}: {residue_ids[idx]:4d} {sequence[idx]}  "
            f"D={scores[idx]:.4f}{tag}"
        )

    segments = find_ordered_segments(scores, residue_ids, threshold)
    if segments:
        lines.append("")
        lines.append(f"  Ordered Segments: {len(segments)} found")
        lines.append(f"  {'─' * 55}")
        for i, seg in enumerate(segments):
            start_idx = residue_ids.index(seg['start'])
            end_idx = residue_ids.index(seg['end']) + 1
            seg_seq = sequence[start_idx:end_idx]
            if len(seg_seq) > 50:
                seg_seq = seg_seq[:24] + '...' + seg_seq[-24:]
            lines.append(
                f"  [{i+1}] {seg['start']}-{seg['end']}  "
                f"ordered={seg['length']}/{seg['total_span']}  "
                f"D_mean={seg['mean_disorder']:.4f}  "
                f"D_range=[{seg['min_disorder']:.3f}, {seg['max_disorder']:.3f}]"
            )
            lines.append(f"       {seg_seq}")
    else:
        lines.append("")
        lines.append("  No ordered segments found. This protein appears largely disordered.")

    lines.append("=" * 70)
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from bfn_loader import load_bfn

    model, config = load_bfn()
    result = predict_disorder(
        model, config,
        'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
        chain_id='A'
    )
    print(format_disorder_report(
        result['disorder_scores'], result['sequence'], result['residue_ids']
    ))
