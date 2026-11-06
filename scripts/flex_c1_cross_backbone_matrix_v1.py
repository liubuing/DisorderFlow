"""FLEX C1 cross-backbone + peptide-swap controls (v1).

Purpose: decide what the corrected-role judge's 26/26 native-top-1 result means.

Two controls, both cheap and purely computational:

1. Cross-backbone matrix (within H3 length groups).
   Score every native H3 sequence of a length group on every backbone of the
   same group, with the corrected Fragment encoding. If a native only wins on
   its own deposited backbone, the judge is reading sequence-backbone
   self-consistency. If it wins on heterologous backbones too, the signal is
   not backbone-specific and the C1 endpoint is even weaker.

2. Peptide-sequence swap (same coordinates, scrambled peptide sequence).
   Keep the peptide backbone coordinates and replace the peptide residues with
   a composition-matched derangement. If the native H3 still scores as well,
   the model is not using peptide identity at all.

Frozen artifacts are never modified; outputs go to a new directory.

Usage:
  python scripts/flex_c1_cross_backbone_matrix_v1.py --out results/flex_cross_backbone_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_pae_dense_v6 import read, freeze  # noqa: E402
from scripts.flex_j0_score_v1 import SCORING_SEEDS, candidate_nll  # noqa: E402

C1 = ROOT / 'results/flex_c1_v1'
J0 = ROOT / 'results/flex_j0_v1'
DEFAULT_OUT = ROOT / 'results/flex_cross_backbone_v1'
AA = 'ACDEFGHIKLMNPQRSTVWY'
SWAP_SEED = 20261101


def build_batches(entry, device, conformer=None):
    from modules.bfn_loader import build_region_batch

    pdb = str(conformer or (ROOT / entry['component_pdb']))
    complex_batch = build_region_batch(
        pdb, entry['region_spec'], context_chains=['P'],
        antigen_chains=['P'], device=device)
    stripped_batch = build_region_batch(
        pdb, entry['region_spec'], context_chains=[],
        antigen_chains=[], device=device)
    return complex_batch, stripped_batch


def mean_nll(model, batch, sequence, device):
    return float(np.mean([
        candidate_nll(model, batch, sequence, seed, device)[0]
        for seed in SCORING_SEEDS]))


def scrambled(sequence, rng):
    letters = list(sequence)
    while True:
        rng.shuffle(letters)
        candidate = ''.join(letters)
        if candidate != sequence:
            return candidate


def swap_peptide_sequence(batch, new_peptide):
    """Replace the antigen-chain residues in an already-built batch."""
    import torch
    from disorderflow.utils.protein.constants import Fragment

    aa = batch['aa']
    fragment = batch['fragment_type'][0]
    mask = (fragment == int(Fragment.Antigen))
    if int(mask.sum()) != len(new_peptide):
        raise RuntimeError(
            f'peptide length mismatch: batch has {int(mask.sum())}, '
            f'new peptide has {len(new_peptide)}')
    encoded = torch.tensor([AA.index(r) for r in new_peptide],
                           dtype=aa.dtype, device=aa.device)
    batch['aa'][0, mask] = encoded
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prereg = read(C1 / 'preregistration.json')
    components = list(prereg['components'])
    groups: dict[int, list[dict]] = {}
    for entry in components:
        groups.setdefault(len(entry['region_sequence']), []).append(entry)

    import torch
    from modules.bfn_loader import load_bfn, _get_checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _cfg = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    # ---------- control 1: cross-backbone matrix, within length groups ----------
    matrix_rows = []
    for length, entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        for backbone in entries:
            complex_batch, stripped_batch = build_batches(backbone, device)
            for candidate_entry in entries:
                sequence = candidate_entry['region_sequence']
                complex_nll = mean_nll(model, complex_batch, sequence, device)
                stripped_nll = mean_nll(model, stripped_batch, sequence, device)
                matrix_rows.append({
                    'length': length,
                    'backbone': backbone['instance'],
                    'backbone_pdb': backbone['pdb_id'],
                    'sequence_of': candidate_entry['instance'],
                    'sequence_pdb': candidate_entry['pdb_id'],
                    'sequence': sequence,
                    'complex_nll': complex_nll,
                    'stripped_nll': stripped_nll,
                    'ecls': complex_nll - stripped_nll,
                })
            print(f"length {length} backbone {backbone['instance']} done", flush=True)

    # ranking analysis
    ranking = []
    for length, entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        rows = [r for r in matrix_rows if r['length'] == length]
        for backbone in entries:
            pool = [r for r in rows if r['backbone'] == backbone['instance']]
            pool.sort(key=lambda r: r['ecls'])
            self_rank = next(i + 1 for i, r in enumerate(pool)
                             if r['sequence_of'] == backbone['instance'])
            ranking.append({
                'length': length,
                'backbone': backbone['instance'],
                'n_sequences': len(pool),
                'self_rank': self_rank,
                'self_ecls': next(r['ecls'] for r in pool
                                  if r['sequence_of'] == backbone['instance']),
                'best_foreign_ecls': next(r['ecls'] for r in pool
                                          if r['sequence_of'] != backbone['instance']),
                'self_minus_best_foreign': (
                    next(r['ecls'] for r in pool
                         if r['sequence_of'] == backbone['instance'])
                    - next(r['ecls'] for r in pool
                           if r['sequence_of'] != backbone['instance'])),
            })

    # ---------- control 2: peptide-sequence swap ----------
    rng = np.random.default_rng(SWAP_SEED)
    swap_rows = []
    for entry in components:
        complex_batch, _ = build_batches(entry, device)
        native = entry['region_sequence']
        peptide = entry['peptide_sequence']
        own = mean_nll(model, complex_batch, native, device)
        partner = scrambled(peptide, rng)
        swapped = swap_peptide_sequence(complex_batch, partner)
        heterologous = mean_nll(model, swapped, native, device)
        swap_rows.append({
            'instance': entry['instance'],
            'native_h3': native,
            'peptide': peptide,
            'scrambled_peptide': partner,
            'complex_nll_own_peptide': own,
            'complex_nll_scrambled_peptide': heterologous,
            'prefers_own_peptide': bool(own < heterologous),
            'delta_own_minus_scrambled': own - heterologous,
        })
        print(f"{entry['instance']} peptide swap done", flush=True)

    summary = {
        'schema_version': 'flex_cross_backbone_v1',
        'checkpoint_path': checkpoint_path,
        'role_encoding': 'H=Heavy(1), P=Antigen(3)',
        'cross_backbone': {
            'n_rows': len(matrix_rows),
            'length_groups': {str(k): len(v) for k, v in sorted(groups.items()) if len(v) > 1},
            'self_rank_one': sum(1 for r in ranking if r['self_rank'] == 1),
            'n_backbones': len(ranking),
            'median_self_rank': float(np.median([r['self_rank'] for r in ranking])) if ranking else None,
            'mean_self_minus_best_foreign': float(np.mean(
                [r['self_minus_best_foreign'] for r in ranking])) if ranking else None,
            'per_backbone': ranking,
        },
        'peptide_swap': {
            'n': len(swap_rows),
            'prefers_own_peptide': sum(1 for r in swap_rows if r['prefers_own_peptide']),
            'mean_delta_own_minus_scrambled': float(np.mean(
                [r['delta_own_minus_scrambled'] for r in swap_rows])),
            'rows': swap_rows,
        },
        'classification': 'control_experiment_not_confirmatory_claim',
        'provenance': 're-scores frozen results/flex_c1_v1/preregistration.json',
    }
    freeze(out_dir / 'matrix.json', {'rows': matrix_rows})
    freeze(out_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != 'peptide_swap' and k != 'cross_backbone'}, indent=1))
    print(json.dumps({k: v for k, v in summary['cross_backbone'].items()
                      if k != 'per_backbone'}, indent=1))
    print(json.dumps({k: v for k, v in summary['peptide_swap'].items()
                      if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
