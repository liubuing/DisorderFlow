"""Design v1: filter ProteinMPNN H3 designs and build the synthesis/AF2 package.

Inputs
  results/design_v1_20261101/mpnn_designs.json      (from benchmark_4hix_mpnn_matched.py)
  data/anti_abeta_refs/4HIX.pdb                    (frozen scaffold)

Outputs (results/design_v1_20261101/)
  shortlist.csv / shortlist.fasta   -> AF2 + PAE shortlist
  af2_input.fasta                   -> colabfold multimer input (VH:VL:A)
  synthesis_aa.fasta                -> Fab chains + scFv (VH-130 / VL-115 / His6)
  package.json                      -> provenance and filter decisions

Design region: 4HIX H3, 1-based positions 96-107 (12 residues), native
VRYDHYSGSSDY.  Everything else fixed.

This is a developability + logistics filter, NOT a binding predictor: the
project's own retrospective (AAYL, 508 variants) showed the ECLS judge has no
usable affinity correlation, so no model score is used to rank candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / 'data/anti_abeta_refs/4HIX.pdb'
DESIGNS = ROOT / 'results/design_v1_20261101/mpnn_designs.json'
OUT = ROOT / 'results/design_v1_20261101'
H3_POSITIONS_1BASED = list(range(96, 108))
NATIVE_H3 = 'VRYDHYSGSSDY'
LINKER = 'GGGGSGGGGSGGGGS'
HIS = 'HHHHHH'


def read_chains():
    model = next(PDBParser(QUIET=True).get_structure('4hix', str(PDB)).get_models())
    out = {}
    for chain in model:
        residues = [r for r in chain if r.id[0] == ' ']
        out[chain.id] = ''.join(seq1(r.resname) for r in residues)
    return out


def net_charge(h3):
    return sum(h3.count(a) for a in 'KR') - sum(h3.count(a) for a in 'DE')


def has_n_glyc(seq):
    return any(seq[i] == 'N' and seq[i + 1] != 'P' and seq[i + 2] in 'ST'
               for i in range(len(seq) - 2))


def max_run(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def evaluate(h3):
    reasons = []
    if len(h3) != 12:
        reasons.append('length!=12')
    if 'C' in h3:
        reasons.append('cysteine')
    if has_n_glyc(h3):
        reasons.append('n_glyc_sequon')
    if max_run(h3) > 3:
        reasons.append('homopolymer_run>3')
    if not (-4 <= net_charge(h3) <= 6):
        reasons.append('net_charge_out_of_range')
    if h3.count('P') > 3:
        reasons.append('proline_rich')
    return reasons


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep', type=int, default=50)
    args = parser.parse_args()

    chains = read_chains()
    vh_full, vl_full, antigen = chains['H'], chains['L'], chains['A']
    assert vh_full[95:107] == NATIVE_H3, vh_full[95:107]

    data = json.loads(DESIGNS.read_text(encoding='utf-8'))
    rows = data['results']

    seen = set()
    kept, dropped = [], []
    for row in sorted(rows, key=lambda r: r['score']):
        h3 = row['sequence']
        if h3 in seen:
            continue
        seen.add(h3)
        reasons = evaluate(h3)
        record = {
            'h3': h3,
            'mpnn_score': row['score'],
            'global_score': row['global_score'],
            'seed': row['seed'],
            'sample_index': row['sample_index'],
            'native_recovery': row['native_recovery'],
            'net_charge': net_charge(h3),
            'reasons': reasons,
        }
        if reasons:
            dropped.append(record)
        else:
            kept.append(record)

    # diversity-aware pick: greedily keep sequences at Hamming >= 3 from picked
    picked = []
    for record in kept:
        if all(hamming(record['h3'], other['h3']) >= 3 for other in picked):
            picked.append(record)
        if len(picked) >= args.keep:
            break
    if len(picked) < args.keep:
        for record in kept:
            if record not in picked:
                picked.append(record)
            if len(picked) >= args.keep:
                break

    # build sequences
    vh_var = vh_full[:130]
    vl_var = vl_full[:115]
    for index, record in enumerate(picked, start=1):
        h3 = record['h3']
        record['design_id'] = f'4HIX-D{index:03d}'
        record['vh_full'] = vh_full[:95] + h3 + vh_full[107:]
        record['vl_full'] = vl_full
        record['vh_130'] = vh_var[:95] + h3 + vh_var[107:]
        record['vl_115'] = vl_var
        record['scfv'] = record['vh_130'] + LINKER + record['vl_115'] + HIS
        record['fab_vh'] = record['vh_full']
        record['fab_vl'] = vl_full
        record['af2_complex'] = f"{record['vh_full']}:{vl_full}:{antigen}"

    native_record = {
        'design_id': '4HIX-native',
        'h3': NATIVE_H3,
        'mpnn_score': None,
        'native_recovery': 1.0,
        'vh_full': vh_full,
        'vl_full': vl_full,
        'vh_130': vh_var,
        'vl_115': vl_var,
        'scfv': vh_var + LINKER + vl_var + HIS,
        'fab_vh': vh_full,
        'fab_vl': vl_full,
        'af2_complex': f'{vh_full}:{vl_full}:{antigen}',
        'note': 'deposited native H3; positive control',
    }

    # scrambled-H3 counterfactual (explicitly NOT an experimental non-binder)
    scrambled_h3 = 'YDVRSGHYSSDY'
    scrambled_record = dict(native_record)
    scrambled_record.update({
        'design_id': '4HIX-H3scramble',
        'h3': scrambled_h3,
        'fab_vh': vh_full[:95] + scrambled_h3 + vh_full[107:],
        'vh_full': vh_full[:95] + scrambled_h3 + vh_full[107:],
        'vh_130': vh_var[:95] + scrambled_h3 + vh_var[107:],
        'scfv': vh_var[:95] + scrambled_h3 + vh_var[107:] + LINKER + vl_var + HIS,
        'af2_complex': f"{vh_full[:95] + scrambled_h3 + vh_full[107:]}:{vl_full}:{antigen}",
        'note': 'composition-matched H3 scramble; counterfactual control, not an experimental non-binder',
    })

    # write outputs
    OUT.mkdir(parents=True, exist_ok=True)
    all_records = [native_record] + picked + [scrambled_record]

    af2_lines = []
    for record in all_records:
        af2_lines.append(f">{record['design_id']}")
        af2_lines.append(record['af2_complex'])
    (OUT / 'af2_input.fasta').write_text('\n'.join(af2_lines) + '\n', encoding='ascii')

    syn_lines = []
    for record in all_records:
        syn_lines.append(f">{record['design_id']}_VH | Fab heavy | H3={record['h3']}")
        syn_lines.append(record['fab_vh'])
        syn_lines.append(f">{record['design_id']}_VL | Fab light")
        syn_lines.append(record['fab_vl'])
        syn_lines.append(f">{record['design_id']}_scFv | VH-GS linker-VL-His6")
        syn_lines.append(record['scfv'])
    (OUT / 'synthesis_aa.fasta').write_text('\n'.join(syn_lines) + '\n', encoding='ascii')

    cols = ['design_id', 'h3', 'mpnn_score', 'native_recovery', 'net_charge']
    with (OUT / 'shortlist.csv').open('w', encoding='utf-8') as handle:
        handle.write(','.join(cols) + '\n')
        for record in all_records:
            handle.write(','.join('' if record.get(c) is None else str(record[c])
                                  for c in cols) + '\n')

    package = {
        'schema_version': 'design_v1_20261101',
        'scaffold_pdb': str(PDB.relative_to(ROOT)),
        'design_region': 'H:96-107 (12 aa, Chothia H3)',
        'native_h3': NATIVE_H3,
        'n_generated': len(rows),
        'n_unique': len(seen),
        'n_passed_filters': len(kept),
        'n_dropped_by_filters': len(dropped),
        'filter_rules': [
            'no cysteine', 'no N-X-S/T sequon', 'no homopolymer run > 3',
            'net charge between -4 and +6', 'proline count <= 3',
        ],
        'diversity_rule': 'greedy, Hamming >= 3 to already picked',
        'selection_basis': 'developability only; no model affinity score used',
        'blacklist_boundary': (
            'AAYL retrospective (508 variants) found Spearman +0.02 between the '
            'BFN ECLS judge and AlphaSeq-estimated affinity; the judge is not used here.'),
        'records': all_records,
        'dropped_examples': dropped[:30],
    }
    (OUT / 'package.json').write_text(json.dumps(package, indent=2, ensure_ascii=False) + '\n',
                                      encoding='utf-8')
    print(json.dumps({k: package[k] for k in (
        'n_generated', 'n_unique', 'n_passed_filters', 'n_dropped_by_filters')}, indent=1))
    print('wrote', OUT / 'af2_input.fasta')
    print('wrote', OUT / 'synthesis_aa.fasta')
    print('wrote', OUT / 'shortlist.csv')
    print('wrote', OUT / 'package.json')


if __name__ == '__main__':
    main()
