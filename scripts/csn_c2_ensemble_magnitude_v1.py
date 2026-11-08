"""CSN Phase C2 (v1): ensemble-magnitude diagnostic for the T1-vs-J0b tension.

Question: the frozen T1 record (restrained-MD ensembles, 7 development clusters)
reports mean pairwise peptide RMSD 0.63 A and an ECLS attenuation of ~0.20, while
the frozen corrected J0b v2 gate reports ensemble marginalization with NO
attenuation (delta -0.126 vs J0a v2). First diagnostic: how much peptide motion
do the J0b generated conformers actually contain? If they are near-rigid, the
"no attenuation" result is expected and uninformative about ensembles; if they
move at T1-like amplitude, the tension is real and lives elsewhere (judge or
cohort).

Three reading conventions are frozen:
- corrected complex-frame (primary): heavy chains identical across seeds -> no
  realignment; peptide CA RMSD directly in the complex frame;
- corrected peptide-superposed (secondary): Kabsch on the peptide itself;
- legacy peptide-superposed (disclosure only): the legacy
  scripts.flex_a2_nmr_rmsf_v1.kabsch (transposed-rotation bug, see campaign doc
  section 6), kept so that future comparisons against any legacy-instrument
  reference values can be made apples-to-apples.

Red lines: no model inference; the frozen J0a/J0b v2 gate numbers are read-only
references and are neither rerun nor revised (classification:
development_diagnostic_not_gate_revision).
"""
import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402
from scripts.flex_a2_nmr_rmsf_v1 import kabsch as legacy_kabsch  # noqa: E402
from scripts.csn_c0_conformer_sharing_v1 import kabsch_row  # noqa: E402

QUEUE = ROOT / 'results/flex_j0_v1/preregistration.json'
CONF_DIR = ROOT / 'results/flex_j0b_v1/conformer_component_pdbs'
J0B_SUMMARY = ROOT / 'results/flex_j0b_v2_antigen_roles/j0b_v2_summary.json'
OUT = ROOT / 'results/csn_c2_v1'

DATE = '2026-11-08'
CLASSIFICATION = 'development_diagnostic_not_gate_revision'
T1_REFERENCE_A = 0.63
FRAME_TOLERANCE_A = 0.1
MIN_SEEDS = 4
NEAR_RIGID_A = 0.3
T1_LIKE_A = 0.6


def tier(spread):
    if spread < NEAR_RIGID_A:
        return 'near_rigid'
    if spread < T1_LIKE_A:
        return 'modest'
    return 't1_like'


def parse_pdb(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def ca_coords(chain):
    out = []
    for res in chain:
        if res.id[0] != ' ' or 'CA' not in res:
            continue
        out.append(res['CA'].coord.astype(float).copy())
    return np.asarray(out)


def load_component_groups():
    groups = {}
    for path in sorted(CONF_DIR.glob('*.pdb')):
        key = path.name.split('_seed_')[0]
        groups.setdefault(key, []).append(path)
    return groups


def map_instances(keys):
    """Map doubled-instance keys to queue instances (exact prefix match)."""
    comps = json.loads(QUEUE.read_text(encoding='utf-8'))['cohort']['components']
    instances = [c['instance'] for c in comps]
    mapping = {}
    for key in keys:
        hits = [i for i in instances if key.startswith(i + '_')]
        if len(hits) != 1:
            raise SystemExit(f'ambiguous conformer key {key}: {hits}')
        mapping[key] = hits[0]
    return mapping


def build_protocol():
    return {
        'classification': CLASSIFICATION,
        'campaign': 'CSN v1',
        'phase': 'C2',
        'date': DATE,
        'question': ('Do the frozen J0b generated conformers contain peptide motion at '
                     'T1-like amplitude (0.63 A mean pairwise reference), or are they '
                     'near-rigid (which would render the J0b "no attenuation" result '
                     'expected and uninformative about ensembles)?'),
        'inputs': {
            'conformer_pdbs': str(CONF_DIR.relative_to(ROOT)),
            'queue': str(QUEUE.relative_to(ROOT)),
            'frozen_gate_reference': str(J0B_SUMMARY.relative_to(ROOT)),
        },
        'metrics': {
            'corrected_complex_frame': ('primary; heavy-chain CA RMSD across seeds '
                                        f'<= {FRAME_TOLERANCE_A} A without realignment '
                                        'is required, else component flagged '
                                        'frame_shifted and only superposed readings apply'),
            'corrected_peptide_superposed': 'secondary; Kabsch (row convention, unit-tested)',
            'legacy_peptide_superposed': ('disclosure only; legacy flex kabsch with the '
                                          'transposed-rotation bug; enables future '
                                          'apples-to-apples checks against legacy-instrument '
                                          'reference numbers'),
        },
        'tiers': {
            'near_rigid': 'mean pairwise spread < 0.3 A',
            'modest': '0.3 <= spread < 0.6 A',
            't1_like': 'spread >= 0.6 A (T1 reference 0.63 A, read-only)',
            'interpretation_closed_rules': {
                'median_component_spread < 0.3': 'marginalization_trivial_explanation',
                'median_component_spread >= 0.6': 'magnitude_parity_preserved',
                'otherwise': 'intermediate_requires_deeper_resolution',
            },
        },
        'red_lines': [
            'the frozen J0a/J0b v2 gate numbers are read-only references',
            'no model inference, no network, no scores',
            'diagnostic only; never revises or reinterprets a frozen gate',
        ],
        'claims_boundary': ('This diagnostic characterizes conformer amplitude only. It '
                            'does not re-evaluate the ECLS endpoint, does not revise the '
                            'frozen J0b v2 gate, and does not establish any binding or '
                            'design claim.'),
        'provenance': {'script_sha256': digest(Path(__file__)),
                       'queue_sha256': digest(QUEUE),
                       'j0b_summary_sha256': digest(J0B_SUMMARY)},
    }


def run():
    protocol_path = OUT / 'protocol.json'
    if not protocol_path.exists():
        raise SystemExit('protocol.json missing; run --protocol first')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['provenance']['script_sha256'] != digest(Path(__file__)):
        raise SystemExit('script changed after protocol freeze; open results/csn_c2_v2')

    groups = load_component_groups()
    mapping = map_instances(groups.keys())
    comps = json.loads(QUEUE.read_text(encoding='utf-8'))['cohort']['components']
    all_instances = [c['instance'] for c in comps]
    missing = sorted(set(all_instances) - set(mapping.values()))
    gate = json.loads(J0B_SUMMARY.read_text(encoding='utf-8'))

    parser = PDBParser(QUIET=True)
    rows = []
    for key in sorted(groups):
        paths = groups[key]
        if len(paths) != MIN_SEEDS:
            raise SystemExit(f'{key}: expected {MIN_SEEDS} conformers, found {len(paths)}')
        peptides, heavies = [], []
        for p in paths:
            st = parse_pdb(p)
            chain_p = [ch for ch in st[0] if ch.id == 'P']
            chain_h = [ch for ch in st[0] if ch.id == 'H']
            if not chain_p or not chain_h:
                raise SystemExit(f'{p.name}: missing chain P or H')
            peptides.append(ca_coords(chain_p[0]))
            heavies.append(ca_coords(chain_h[0]))
        n_res = {len(x) for x in peptides}
        if len(n_res) != 1:
            raise SystemExit(f'{key}: peptide length differs across seeds: {n_res}')
        if len({len(x) for x in heavies}) != 1:
            raise SystemExit(f'{key}: heavy length differs across seeds')

        frame_shift = max(float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))
                          for a, b in combinations(heavies, 2))
        frame_ok = frame_shift <= FRAME_TOLERANCE_A
        pairs = list(combinations(range(MIN_SEEDS), 2))
        corrected_frame = [float(np.sqrt(((peptides[a] - peptides[b]) ** 2)
                                         .sum(axis=1).mean())) for a, b in pairs]
        corrected_super = [float(np.sqrt(((kabsch_row(peptides[a], peptides[b])
                                           - peptides[b]) ** 2).sum(axis=1).mean()))
                           for a, b in pairs]
        legacy_super = [float(np.sqrt(((legacy_kabsch(peptides[a], peptides[b])
                                        - peptides[b]) ** 2).sum(axis=1).mean()))
                        for a, b in pairs]
        rows.append({
            'instance': mapping[key], 'pdb_key': key,
            'n_peptide_residues': int(next(iter(n_res))),
            'frame_shift_max': round(frame_shift, 4), 'frame_ok': frame_ok,
            'corrected_complex_frame': {
                'mean': round(float(np.mean(corrected_frame)), 4),
                'max': round(float(np.max(corrected_frame)), 4)},
            'corrected_peptide_superposed': {
                'mean': round(float(np.mean(corrected_super)), 4),
                'max': round(float(np.max(corrected_super)), 4)},
            'legacy_peptide_superposed': {
                'mean': round(float(np.mean(legacy_super)), 4),
                'max': round(float(np.max(legacy_super)), 4)},
            'tier_corrected_complex_frame': tier(float(np.mean(corrected_frame))),
        })
    freeze(OUT / 'conformer_spread.json', {'classification': CLASSIFICATION,
                                           'rows': rows, 'missing_components': missing})

    def med(field):
        return float(np.median([r[field]['mean'] for r in rows]))

    median_frame = med('corrected_complex_frame')
    median_super = med('corrected_peptide_superposed')
    median_legacy = med('legacy_peptide_superposed')
    if median_frame < NEAR_RIGID_A:
        verdict = 'marginalization_trivial_explanation'
    elif median_frame >= T1_LIKE_A:
        verdict = 'magnitude_parity_preserved'
    else:
        verdict = 'intermediate_requires_deeper_resolution'
    t1_like_n = sum(1 for r in rows if r['tier_corrected_complex_frame'] == 't1_like')
    near_rigid_n = sum(1 for r in rows if r['tier_corrected_complex_frame'] == 'near_rigid')
    summary = {
        'classification': CLASSIFICATION,
        'date': DATE,
        'n_components': len(rows),
        'missing_components': missing,
        'n_frame_ok': sum(1 for r in rows if r['frame_ok']),
        'median_spread_corrected_complex_frame': round(median_frame, 4),
        'median_spread_corrected_peptide_superposed': round(median_super, 4),
        'median_spread_legacy_peptide_superposed': round(median_legacy, 4),
        'tier_counts': {'near_rigid': near_rigid_n,
                        'modest': len(rows) - near_rigid_n - t1_like_n,
                        't1_like': t1_like_n},
        't1_reference_mean_pairwise_rmsd_a': T1_REFERENCE_A,
        't1_reference_instrument_note': ('frozen read-only reference from 7 development '
                                         'clusters (INTEGRATIVE_ANALYSIS section 2.2); its '
                                         'alignment convention is not recoverable from the '
                                         'frozen record, so the legacy column above is the '
                                         'apples-to-apples bridge if T1 used the legacy '
                                         'function'),
        'frozen_gate_reference': {
            'j0b_v2_mean_advantage': gate['mean_advantage'],
            'j0b_v2_attenuation_vs_j0a_v2': gate['attenuation_vs_j0a_v2'],
            'usage': 'read-only context; not recomputed here'},
        'verdict': verdict,
        'claims_boundary': 'see protocol.json claims_boundary (frozen)',
    }
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'protocol_sha256': digest(protocol_path),
        'script_sha256': digest(Path(__file__)),
        'artifacts': ['conformer_spread.json', 'summary.json', 'completion.json'],
    })
    print('C2 frozen:', verdict, '| median complex-frame spread',
          round(median_frame, 3), 'A | tiers', near_rigid_n, '/',
          len(rows) - near_rigid_n - t1_like_n, '/', t1_like_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.protocol:
        freeze(OUT / 'protocol.json', build_protocol())
        print('protocol frozen:', digest(OUT / 'protocol.json'))
    if args.run:
        run()


if __name__ == '__main__':
    main()
