"""FLEX cross-backbone matrix, expanded cohort (code-fix report 7.4).

The v1 cross-backbone control (results/flex_cross_backbone_v1) was limited to
the 26-component C1 cohort, whose H3 length groups hold at most 2 DISTINCT
deposited sequences - so the control rested on 5-6 independent pairs and
could not separate epitope specificity from sequence-backbone
self-consistency.  This v2 expands the pool to every component that passed
the A1 coordinate-level gates (107 components / 36 structures from the frozen
scaffold-audit manifest), keeps only length groups with >= 4 distinct
deposited H3 sequences, and runs the full within-group matrix.

Disclosure (protocol-level, must travel with any citation):
* Expanded components bypass the campaign-level exclusions (exposure
  cleanliness, isolation families, prior-panel overlap, clone collisions).
  Those gates protect the NATIVE-RECOGNITION contrast (an epitope must have
  no other binding partner in the deposited asymmetric unit).  This control
  asks a different question - ON a fixed [H, P] scoring construct, does the
  judge rank the backbone's own deposited H3 above other deposited H3s of
  the same length? - for which only the coordinate-level gates matter.  The
  expanded rows therefore support the control, never the native-recognition
  claim.
* Scoring constructs and encoding are identical to v1 (corrected Fragment
  roles: H=Heavy, P=Antigen; same seeds; same checkpoint).

Output (write-once): results/flex_cross_backbone_v2_expanded/

Usage:
  python scripts/flex_cross_backbone_matrix_v2_expanded.py --build-only
  python scripts/flex_cross_backbone_matrix_v2_expanded.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_pae_dense_v6 import read, freeze, digest  # noqa: E402
from scripts.flex_j0_score_v1 import SCORING_SEEDS  # noqa: E402
from scripts.flex_j0_build_cohort_v1 import load_numbering  # noqa: E402
from scripts.materialize_pae_analog_structures_v26 import (  # noqa: E402
    region_indices, write_component_pdb)

A1 = ROOT / 'results/flex_a1_scaffold_audit_v1'
J0_PDB = ROOT / 'results/flex_j0_v1/pdb'
OUT = ROOT / 'results/flex_cross_backbone_v2_expanded'
MIN_DISTINCT = 4


def build_entries():
    """Materialize [H, P] scoring PDBs for every coordinate-gate-passing
    component and return entries with region specs (fail-closed on
    numbering/sequence mismatches)."""
    from Bio.PDB import MMCIFParser
    from scripts.flex_a1_scaffold_audit_v1 import strict_residues

    manifest = read(A1 / 'manifest.json')
    records = [r for r in manifest['records'] if r['passes']]
    labels = load_numbering()
    pdb_dir = OUT / 'pdb'
    pdb_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = A1 / 'cif'
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)

    entries, skipped = [], []
    for record in sorted(records, key=lambda r: r['instance']):
        instance = record['instance']
        pdb_id = record['pdb_id']
        heavy_seq = record['sequences']['heavy']
        label = labels.get(heavy_seq)
        if not label or label.get('error') or not label.get('chothia_numbering'):
            skipped.append({'instance': instance, 'reason': 'no_chothia_numbering'})
            continue
        indices = region_indices(label['chothia_numbering'],
                                 label.get('chothia_query_start',
                                           label['query_start']), 93, 102)
        if not indices:
            skipped.append({'instance': instance, 'reason': 'empty_h3_region'})
            continue
        if indices != list(range(indices[0], indices[0] + len(indices))):
            skipped.append({'instance': instance,
                            'reason': f'non_contiguous_indices_{indices}'})
            continue
        region_seq = ''.join(heavy_seq[i] for i in indices)
        if region_seq != record['sequences']['region']:
            skipped.append({'instance': instance, 'reason': 'region_mismatch'})
            continue

        cohort_pdb = J0_PDB / f'{instance}.pdb'
        if cohort_pdb.exists():
            component_pdb = f'results/flex_j0_v1/pdb/{instance}.pdb'
        else:
            cif_path = cif_dir / f'{pdb_id}.cif'
            structure = parser.get_structure(pdb_id, str(cif_path))
            model = next(structure.get_models())
            heavy_rows = strict_residues(model[record['chains']['heavy']],
                                         allow_mse=True)
            peptide_rows = strict_residues(model[record['chains']['peptide']],
                                           allow_mse=True)
            out_path = pdb_dir / f'{instance}.pdb'
            write_component_pdb(out_path, [('H', heavy_rows), ('P', peptide_rows)])
            component_pdb = f'results/flex_cross_backbone_v2_expanded/pdb/{instance}.pdb'

        entries.append({
            'instance': instance,
            'pdb_id': pdb_id,
            'component_pdb': component_pdb,
            'component_pdb_sha256': digest(ROOT / component_pdb),
            'region_spec': f'H:{indices[0] + 1}-{indices[-1] + 1}',
            'region_sequence': region_seq,
            'peptide_sequence': record['sequences']['peptide'],
            'in_c1_cohort': cohort_pdb.exists(),
        })
    return entries, skipped


def build_batches(entry, device):
    from modules.bfn_loader import build_region_batch

    pdb = str(ROOT / entry['component_pdb'])
    complex_batch = build_region_batch(
        pdb, entry['region_spec'], context_chains=['P'],
        antigen_chains=['P'], device=device)
    stripped_batch = build_region_batch(
        pdb, entry['region_spec'], context_chains=[],
        antigen_chains=[], device=device)
    return complex_batch, stripped_batch


def mean_nll(model, batch, sequence, device):
    from scripts.flex_j0_score_v1 import candidate_nll
    return float(np.mean([
        candidate_nll(model, batch, sequence, seed, device)[0]
        for seed in SCORING_SEEDS]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, skipped = build_entries()
    groups: dict[int, list[dict]] = {}
    for entry in entries:
        groups.setdefault(len(entry['region_sequence']), []).append(entry)
    selected = {}
    for length, members in sorted(groups.items()):
        distinct = {m['region_sequence'] for m in members}
        if len(distinct) >= MIN_DISTINCT:
            selected[length] = members
    protocol = {
        'schema_version': 'flex_cross_backbone_v2_expanded',
        'source': 'results/flex_a1_scaffold_audit_v1/manifest.json '
                  '(coordinate-gate-passing components)',
        'n_components_built': len(entries),
        'n_skipped': len(skipped),
        'skipped': skipped,
        'min_distinct_per_group': MIN_DISTINCT,
        'selected_length_groups': {str(k): {
            'n_backbones': len(v),
            'n_distinct_sequences': len({m['region_sequence'] for m in v}),
        } for k, v in selected.items()},
        'disclosure': (
            'expanded components bypass campaign-level exclusions '
            '(exposure/isolation/prior-panel/clone); they support only the '
            'within-backbone sequence-preference control, never the '
            'native-recognition claim'),
        'scoring_seeds': SCORING_SEEDS,
        'classification': 'control_experiment_not_confirmatory_claim',
    }
    freeze(out_dir / 'protocol.json', protocol)
    freeze(out_dir / 'entries.json', {'components': entries})
    print(json.dumps({k: protocol[k] for k in
                      ['n_components_built', 'n_skipped',
                       'selected_length_groups']}, indent=1))
    if args.build_only:
        return
    if not selected:
        raise SystemExit('no length group reaches the distinct-sequence floor')

    import torch
    from modules.bfn_loader import load_bfn, _get_checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _cfg = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    matrix_rows = []
    scoring_dir = out_dir / 'scoring'
    scoring_dir.mkdir(parents=True, exist_ok=True)
    for length, members in sorted(selected.items()):
        for backbone in members:
            cache_path = scoring_dir / f'{backbone["instance"]}.json'
            if cache_path.exists():
                matrix_rows.extend(read(cache_path)['rows'])
                print(f'length {length} backbone {backbone["instance"]} cached',
                      flush=True)
                continue
            complex_batch, stripped_batch = build_batches(backbone, device)
            rows = []
            for candidate in members:
                sequence = candidate['region_sequence']
                complex_nll = mean_nll(model, complex_batch, sequence, device)
                stripped_nll = mean_nll(model, stripped_batch, sequence, device)
                rows.append({
                    'length': length,
                    'backbone': backbone['instance'],
                    'backbone_pdb': backbone['pdb_id'],
                    'sequence_of': candidate['instance'],
                    'sequence_pdb': candidate['pdb_id'],
                    'sequence': sequence,
                    'complex_nll': complex_nll,
                    'stripped_nll': stripped_nll,
                    'ecls': complex_nll - stripped_nll,
                })
            freeze(cache_path, {'rows': rows})
            matrix_rows.extend(rows)
            print(f'length {length} backbone {backbone["instance"]} scored '
                  f'({len(members)} candidates)', flush=True)

    # ---- ranking analysis: self vs DISTINCT foreign sequences ----
    ranking = []
    for length, members in sorted(selected.items()):
        for backbone in members:
            pool = [r for r in matrix_rows
                    if r['length'] == length
                    and r['backbone'] == backbone['instance']]
            self_row = next(r for r in pool
                            if r['sequence_of'] == backbone['instance'])
            self_seq = self_row['sequence']
            foreign = [r for r in pool if r['sequence'] != self_seq]
            distinct_foreign = {r['sequence'] for r in foreign}
            best_foreign = min(r['ecls'] for r in foreign)
            ranking.append({
                'length': length,
                'backbone': backbone['instance'],
                'n_rows': len(pool),
                'n_distinct_sequences': len(distinct_foreign) + 1,
                'self_ecls': self_row['ecls'],
                'best_distinct_foreign_ecls': best_foreign,
                'self_minus_best_distinct_foreign': self_row['ecls'] - best_foreign,
                'self_strictly_best': bool(self_row['ecls'] < best_foreign),
                'in_c1_cohort': backbone['in_c1_cohort'],
            })

    def frac(key):
        values = [r[key] for r in ranking]
        return float(np.mean(values)) if values else None

    n_groups = {str(k): {'n_backbones': len(v),
                         'self_strictly_best': sum(1 for r in v if r['self_strictly_best']),
                         'chance_if_exchangeable': 1.0 / len({r['sequence'] for r in
                              matrix_rows if r['length'] == k})}
                for k, v in sorted(selected.items())
                for v in [[r for r in ranking if r['length'] == k]]}
    summary = {
        'schema_version': 'flex_cross_backbone_v2_expanded',
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': digest(Path(checkpoint_path)),
        'role_encoding': 'H=Heavy(1), P=Antigen(3)',
        'n_matrix_rows': len(matrix_rows),
        'n_backbones': len(ranking),
        'self_strictly_best': sum(1 for r in ranking if r['self_strictly_best']),
        'self_strictly_best_fraction': frac('self_strictly_best'),
        'mean_self_minus_best_distinct_foreign': frac('self_minus_best_distinct_foreign'),
        'median_self_minus_best_distinct_foreign': float(np.median(
            [r['self_minus_best_distinct_foreign'] for r in ranking])),
        'per_length': n_groups,
        'interpretation_note': (
            'under exchangeability a backbone self-ranks strictly best with '
            'probability 1/n_distinct; a fraction far above that indicates '
            'sequence-backbone self-consistency, while a fraction near it '
            'would mean the corrected C1 26/26 result does not even reflect '
            'self-consistency'),
        'per_backbone': ranking,
        'classification': 'control_experiment_not_confirmatory_claim',
        'provenance': 'expands results/flex_cross_backbone_v1 per code-fix '
                      'report 7.4; frozen v1 artifacts untouched',
    }
    freeze(out_dir / 'matrix.json', {'rows': matrix_rows})
    freeze(out_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('per_backbone',)}, indent=1))


if __name__ == '__main__':
    main()
