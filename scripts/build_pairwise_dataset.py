"""Build the within-complex pairwise supervision dataset (v1).

Groups (one complex each, H3-only variants, experimentally measured binding):

  1mlc   (ABB002, lysozyme)       AbBiBench binding_score    n~132
  1n8z   (ABB003, HER2)           AbBiBench binding_score    n~397
  2fjg   (ABB004, VEGF)           AbBiBench binding_score    n~247
  AAYL49 (AlphaSeq anti-HR2)      9 - median(log10 Kd nM)    n=281
  AAYL51 (AlphaSeq anti-HR2)      9 - median(log10 Kd nM)    n=227

Only substitutions inside the H3 window are kept (everything else must equal
the parent heavy chain, light chain and antigen).

Output: data/pairwise_v1/variants.json
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/abbibench_raw_v1'
OUT = ROOT / 'data/pairwise_v1'
COHORT = ROOT / 'data/aayl_original_retrospective_v1/cohort.json'
STRUCT = ROOT / 'data/abbibench_structural_audit_v1/structures'

H3_RE = re.compile(r'YYC(.+?)WGQG')

GROUPS = {
    '1mlc': {
        'structure': STRUCT / '1mlc_bae.pdb',
        'heavy': 'B', 'light': 'A', 'antigen': ['E'],
        'region_spec': 'B:97-105',
        'label_source': 'AbBiBench binding_score (1mlc)',
    },
    '1n8z': {
        'structure': STRUCT / '1n8z_bac.pdb',
        'heavy': 'B', 'light': 'A', 'antigen': ['C'],
        'region_spec': 'B:97-109',
        'label_source': 'AbBiBench binding_score (1n8z)',
    },
    '2fjg': {
        'structure': STRUCT / '2fjg_hlv.pdb',
        'heavy': 'H', 'light': 'L', 'antigen': ['V'],
        'region_spec': 'H:97-109',
        'label_source': 'AbBiBench binding_score (2fjg)',
    },
    'AAYL49': {
        'structure': STRUCT / 'AAYL49_bca.pdb',
        'heavy': 'B', 'light': 'C', 'antigen': ['A'],
        'region_spec': 'B:97-107',
        'label_source': '9 - median(original log10 estimated Kd in nM)',
    },
    'AAYL51': {
        'structure': STRUCT / 'AAYL51_bca.pdb',
        'heavy': 'B', 'light': 'C', 'antigen': ['A'],
        'region_spec': 'B:97-108',
        'label_source': '9 - median(original log10 estimated Kd in nM)',
    },
}


def extract_h3(heavy):
    match = H3_RE.search(heavy)
    return match.group(1) if match else None


def context_of(heavy):
    match = H3_RE.search(heavy)
    if not match:
        return None
    return heavy[:match.start(1)] + heavy[match.end(1):]


def build_abbibench_group(name, cfg):
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    model = next(PDBParser(QUIET=True).get_structure(name, str(cfg['structure'])).get_models())
    residues = [r for r in model[cfg['heavy']] if r.id[0] == ' ']
    parent_heavy = ''.join(seq1(r.resname) for r in residues)
    parent_match = H3_RE.search(parent_heavy)
    if parent_match is None:
        raise RuntimeError(f'{name}: parent heavy chain has no YYC..WGQG motif')
    parent_h3 = parent_match.group(1)
    prefix = parent_heavy[:parent_match.start(1)]
    suffix = parent_heavy[parent_match.end(1):]

    rows = list(csv.DictReader((RAW / f'{name}.csv').open(encoding='utf-8')))
    kept = []
    for row in rows:
        heavy = row['heavy_chain_seq'].strip()
        match = H3_RE.search(heavy)
        if match is None:
            continue
        if heavy[:match.start(1)] != prefix or heavy[match.end(1):] != suffix:
            continue
        h3 = match.group(1)
        if len(h3) != len(parent_h3):
            continue
        kept.append({'h3': h3, 'label': float(row['binding_score'])})
    # dedupe identical H3 (average label)
    grouped = {}
    for record in kept:
        grouped.setdefault(record['h3'], []).append(record['label'])
    variants = [{'h3': h3, 'label': sum(v) / len(v), 'n_measurements': len(v)}
                for h3, v in grouped.items()]
    variants.sort(key=lambda x: x['h3'])
    print(f'{name}: rows {len(rows)} -> H3-only on the deposited parent {len(kept)} '
          f'-> unique H3 {len(variants)} | parent H3 {parent_h3} ({len(parent_h3)} aa)')
    return variants


def build_aayl_group(family, cohort):
    variants = []
    for row in cohort['eligible']:
        if row['family'] != family:
            continue
        variants.append({
            'h3': row['h3_sequence'],
            'label': float(row['endpoint']),
            'n_measurements': len(row['observations']),
        })
    # dedupe
    grouped = {}
    for record in variants:
        grouped.setdefault(record['h3'], []).append(record['label'])
    out = [{'h3': h3, 'label': sum(v) / len(v), 'n_measurements': len(v)}
           for h3, v in grouped.items()]
    out.sort(key=lambda x: x['h3'])
    print(f'{family}: eligible {len(variants)} -> unique H3 {len(out)}')
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(COHORT.read_text(encoding='utf-8'))
    groups = {}
    for name in ('1mlc', '1n8z', '2fjg'):
        variants = build_abbibench_group(name, GROUPS[name])
        if variants:
            groups[name] = {**GROUPS[name], 'variants': variants}
    for name in ('AAYL49', 'AAYL51'):
        variants = build_aayl_group(name, cohort)
        groups[name] = {**GROUPS[name], 'variants': variants}

    total = sum(len(g['variants']) for g in groups.values())
    payload = {
        'schema_version': 'pairwise_v1',
        'unit': 'within-complex H3-only variants',
        'groups': {k: {'structure': str(v['structure'].relative_to(ROOT)),
                       'heavy': v['heavy'], 'light': v['light'],
                       'antigen': v['antigen'], 'region_spec': v['region_spec'],
                       'label_source': v['label_source'],
                       'n_variants': len(v['variants']),
                       'variants': v['variants']}
                   for k, v in groups.items()},
        'n_groups': len(groups),
        'n_variants': total,
        'notes': [
            'labels are not comparable across groups; only within-group order is used',
            'AbBiBench binding_score is higher-is-better (benchmark convention); '
            'AAYL endpoint is 9 - median(log10 Kd nM), higher-is-better',
            'H3-only filter: heavy chain outside H3, light chain and antigen must match parent',
        ],
    }
    (OUT / 'variants.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: len(v['variants']) for k, v in groups.items()}, indent=1))
    print('total', total, '->', OUT / 'variants.json')


if __name__ == '__main__':
    main()
