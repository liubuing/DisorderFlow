"""Design v1: parse ColabFold/AlphaFold2 outputs into interface-PAE ranking.

For every design in results/design_v1_20261101/af2_out/, locate the rank-1
scores JSON (multimer model), compute:

  - antibody->antigen interface PAE   mean(pae[ab, ag])
  - antigen->antibody interface PAE   mean(pae[ag, ab])
  - symmetric mean                    average of the two
  - ipTM, mean pLDDT

Chain layout per input FASTA: VH : VL : antigen  (219 : 219 : 6), so the
antibody block is [0, 438) and the antigen block is [438, 444).

Output: results/design_v1_20261101/af2_pae_ranking.csv + .json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'results/design_v1_20261101/af2_out'
AB_LEN = 438
AG_LEN = 6


def find_scores(out_dir: Path, design_id: str):
    candidates = sorted(out_dir.glob(f'*{design_id}*scores_rank_001*.json'))
    if not candidates:
        candidates = sorted(out_dir.glob(f'*{design_id}*.json'))
    return candidates[0] if candidates else None


def interface_pae(pae):
    pae = np.asarray(pae, dtype=float)
    ab_to_ag = float(pae[:AB_LEN, AB_LEN:AB_LEN + AG_LEN].mean())
    ag_to_ab = float(pae[AB_LEN:AB_LEN + AG_LEN, :AB_LEN].mean())
    return ab_to_ag, ag_to_ab, (ab_to_ag + ag_to_ab) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=str(DEFAULT_DIR))
    parser.add_argument('--out-prefix', default=str(ROOT / 'results/design_v1_20261101/af2_pae_ranking'))
    args = parser.parse_args()
    out_dir = Path(args.dir)

    rows = []
    for path in sorted(out_dir.glob('*_scores_rank_001*.json')):
        design_id = path.name.split('_scores_rank_001')[0]
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'pae' not in data:
            continue
        ab_to_ag, ag_to_ab, sym = interface_pae(data['pae'])
        plddt = data.get('plddt')
        rows.append({
            'design_id': design_id,
            'interface_pae_ab_to_ag': ab_to_ag,
            'interface_pae_ag_to_ab': ag_to_ab,
            'interface_pae_symmetric': sym,
            'iptm': data.get('iptm'),
            'ptm': data.get('ptm'),
            'mean_plddt': float(np.mean(plddt)) if plddt else None,
            'scores_json': str(path.relative_to(ROOT)),
        })
    rows.sort(key=lambda r: r['interface_pae_symmetric'])

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cols = ['design_id', 'interface_pae_symmetric', 'interface_pae_ab_to_ag',
            'interface_pae_ag_to_ab', 'iptm', 'ptm', 'mean_plddt', 'scores_json']
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if row.get(k) is None else row[k]) for k in cols})
    prefix.with_suffix('.json').write_text(
        json.dumps({'n': len(rows), 'rows': rows}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'n': len(rows), 'n_designs': len([r for r in rows if 'native' not in r['design_id']])}, indent=1))
    for row in rows[:12]:
        print(f"{row['design_id']:20s} symPAE {row['interface_pae_symmetric']:.2f} "
              f"ipTM {row['iptm'] if row['iptm'] is None else round(row['iptm'], 3)} "
              f"pLDDT {row['mean_plddt'] if row['mean_plddt'] is None else round(row['mean_plddt'], 1)}")


if __name__ == '__main__':
    main()
