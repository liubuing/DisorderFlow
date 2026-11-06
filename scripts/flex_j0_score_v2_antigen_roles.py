"""FLEX J0 re-scoring with corrected chain-role encoding (v2).

Bug found 2026-11-01 (audit): ``build_region_batch`` called without
``antigen_chains`` leaves ``fragment_type`` at the generic chain index produced
by ``MergeProtein`` (0, 1, 2 ...), but the residue encoder is
``nn.Embedding(10, padding_idx=0)`` with the documented convention
1 = Heavy, 2 = Light, 3 = Antigen.  Consequences in the frozen J0a / J0b / C1
scoring path:

* the antibody heavy chain was embedded as index 0 = padding;
* the peptide was embedded as index 1 = Heavy;
* ``mask_antigen`` (fragment_type == 3) was empty, so the receiver's antigen
  branch never engaged.

This script re-scores the very same frozen J0 cohort and the very same
composition-matched shuffles with the intended encoding (H -> Heavy,
P -> Antigen, other context -> Light), writes to a NEW directory, and never
touches the frozen J0a artifacts.

Usage:
  python scripts/flex_j0_score_v2_antigen_roles.py --instances 2x6m_A_B,8op0_A_B --out results/flex_j0_v2_antigen_roles
  python scripts/flex_j0_score_v2_antigen_roles.py --all
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_pae_dense_v6 import read, freeze  # noqa: E402
from scripts.flex_j0_score_v1 import SCORING_SEEDS, FIXED_T, candidate_nll  # noqa: E402

J0 = ROOT / 'results/flex_j0_v1'
DEFAULT_OUT = ROOT / 'results/flex_j0_v2_antigen_roles'


def build_batches(entry, device):
    """Same two contexts as J0, but with the intended Fragment encoding."""
    from modules.bfn_loader import build_region_batch

    pdb = str(ROOT / entry['component_pdb'])
    complex_batch = build_region_batch(
        pdb, entry['region_spec'], context_chains=['P'],
        antigen_chains=['P'], device=device)
    stripped_batch = build_region_batch(
        pdb, entry['region_spec'], context_chains=[],
        antigen_chains=[], device=device)
    return complex_batch, stripped_batch


def role_report(batch):
    """Record the fragment_type layout actually fed to the model."""
    from disorderflow.utils.protein.constants import Fragment
    ft = batch['fragment_type'][0]
    chains = [c[0] if isinstance(c, (list, tuple)) else c for c in batch['chain_id']]
    per_chain: dict[str, list[int]] = {}
    for index, chain in enumerate(chains):
        per_chain.setdefault(chain, [])
        value = int(ft[index])
        if value not in per_chain[chain]:
            per_chain[chain].append(value)
    return {
        'length': int(ft.shape[0]),
        'fragment_type_by_chain': {k: sorted(v) for k, v in per_chain.items()},
        'n_heavy': int((ft == int(Fragment.Heavy)).sum()),
        'n_light': int((ft == int(Fragment.Light)).sum()),
        'n_antigen': int((ft == int(Fragment.Antigen)).sum()),
    }


def score_component(model, entry, device):
    native = entry['region_sequence']
    candidates = [native] + entry['shuffles']
    complex_batch, stripped_batch = build_batches(entry, device)
    per_candidate = {'complex': {}, 'stripped': {}}
    for context, batch in (('complex', complex_batch), ('stripped', stripped_batch)):
        for candidate in candidates:
            repeats = [candidate_nll(model, batch, candidate, seed, device)[0]
                       for seed in SCORING_SEEDS]
            per_candidate[context][candidate] = float(np.mean(repeats))
        print(f"  {entry['instance']} {context} scored {len(candidates)}", flush=True)

    ecls_native = per_candidate['complex'][native] - per_candidate['stripped'][native]
    shuffle_ecls = [
        per_candidate['complex'][s] - per_candidate['stripped'][s]
        for s in entry['shuffles']]
    advantage = float(np.mean(shuffle_ecls)) - ecls_native
    return {
        'instance': entry['instance'],
        'family': entry.get('family'),
        'native': native,
        'region_spec': entry['region_spec'],
        'peptide_sequence': entry['peptide_sequence'],
        'scoring_seeds': SCORING_SEEDS,
        'fixed_t': FIXED_T,
        'role_encoding': {
            'complex': role_report(complex_batch),
            'stripped': role_report(stripped_batch),
        },
        'nll': per_candidate,
        'ecls_native': ecls_native,
        'advantage': advantage,
        'shuffle_advantage_samples': shuffle_ecls,
    }


def family_table(rows):
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row.get('family')), []).append(row['advantage'])
    return {k: {'mean_advantage': float(np.mean(v)), 'n_components': len(v)}
            for k, v in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', default='')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--limit-shuffles', type=int, default=0,
                        help='debug only: use the first N shuffles (0 = all 50)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = read(J0 / 'cohort.json')
    prereg = read(J0 / 'preregistration.json')
    family_of = {}
    for family_key, members in prereg['cohort']['antigen_families'].items():
        for peptide in members:
            family_of[peptide] = family_key
    entries = list(cohort['components'])
    for entry in entries:
        entry['family'] = family_of.get(entry['peptide_sequence'])
    if not args.all:
        wanted = [x for x in args.instances.split(',') if x]
        if not wanted:
            raise SystemExit('pass --instances a,b or --all')
        entries = [e for e in entries if e['instance'] in wanted]
        missing = set(wanted) - {e['instance'] for e in entries}
        if missing:
            raise SystemExit(f'unknown instances: {sorted(missing)}')
    if args.limit_shuffles:
        for entry in entries:
            entry['shuffles'] = entry['shuffles'][:args.limit_shuffles]

    import torch
    from modules.bfn_loader import load_bfn, _get_checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _cfg = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    rows = []
    for entry in entries:
        cache = out_dir / 'scoring' / f"{entry['instance']}.json"
        if cache.exists():
            row = read(cache)
        else:
            print(entry['instance'], 'scoring', flush=True)
            row = score_component(model, entry, device)
            freeze(cache, row)
        rows.append(row)

    families = family_table(rows)
    values = [r['advantage'] for r in rows]
    summary = {
        'gate': 'j0a_single_pose_corrected_roles',
        'n_components': len(rows),
        'n_families': len(families),
        'mean_advantage': float(np.mean(values)) if values else None,
        'positive_components': int(sum(1 for v in values if v > 0)),
        'per_family': families,
        'role_encoding_note': (
            'H -> Fragment.Heavy(1), P -> Fragment.Antigen(3), '
            'other context -> Fragment.Light(2)'),
        'checkpoint_path': checkpoint_path,
        'classification': 'development_gate_not_confirmatory_claim',
        'provenance': 're-scores frozen results/flex_j0_v1/cohort.json; '
                      'frozen J0a artifacts untouched',
    }
    freeze(out_dir / 'j0a_v2_summary.json', summary)
    freeze(out_dir / 'j0a_v2_results.json', {'components': rows})
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
