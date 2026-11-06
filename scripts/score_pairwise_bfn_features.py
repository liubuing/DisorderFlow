"""Score the within-complex variant dataset with the corrected-role BFN judge.

For every variant in data/pairwise_v1/variants.json compute, with the intended
Fragment encoding (heavy -> Heavy, light -> Light, antigen -> Antigen):

    complex NLL, stripped NLL, ECLS = complex - stripped

Three scoring seeds per variant, mean reported.  Output:
results/pairwise_v1/bfn_features.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_pae_dense_v6 import read  # noqa: E402
from scripts.flex_j0_score_v1 import SCORING_SEEDS, candidate_nll  # noqa: E402

DATA = ROOT / 'data/pairwise_v1/variants.json'
DEFAULT_OUT = ROOT / 'results/pairwise_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--instances', default='')
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read(DATA)
    wanted = [x for x in args.instances.split(',') if x]

    import torch
    from modules.bfn_loader import build_region_batch, load_bfn, _get_checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _cfg = load_bfn(device)
    checkpoint = _get_checkpoint_path()

    records = []
    for group, info in data['groups'].items():
        if wanted and group not in wanted:
            continue
        pdb = ROOT / info['structure']
        heavy = info['heavy']
        context = [c for c in (info['light'], *info['antigen'])]
        complex_batch = build_region_batch(
            str(pdb), info['region_spec'], context_chains=context,
            antigen_chains=list(info['antigen']), device=device)
        stripped_batch = build_region_batch(
            str(pdb), info['region_spec'], context_chains=[], antigen_chains=[],
            device=device)
        expected = len(info['variants'][0]['h3'])
        assert int(complex_batch['generate_flag'].sum()) == expected, group
        print(f'{group}: {info["n_variants"]} variants, region {info["region_spec"]}',
              flush=True)
        for variant in info['variants']:
            h3 = variant['h3']
            if len(h3) != expected:
                continue
            complex_nll = float(np.mean([
                candidate_nll(model, complex_batch, h3, seed, device)[0]
                for seed in SCORING_SEEDS]))
            stripped_nll = float(np.mean([
                candidate_nll(model, stripped_batch, h3, seed, device)[0]
                for seed in SCORING_SEEDS]))
            records.append({
                'group': group,
                'h3': h3,
                'label': variant['label'],
                'n_measurements': variant.get('n_measurements'),
                'complex_nll': complex_nll,
                'stripped_nll': stripped_nll,
                'ecls': complex_nll - stripped_nll,
            })
        print(f'{group}: scored {sum(1 for r in records if r["group"] == group)}',
              flush=True)

    payload = {
        'schema_version': 'pairwise_bfn_features_v1',
        'checkpoint': checkpoint,
        'scoring_seeds': SCORING_SEEDS,
        'role_encoding': 'H=Heavy(1), L=Light(2), A=Antigen(3)',
        'n_records': len(records),
        'records': records,
    }
    (out_dir / 'bfn_features.json').write_text(
        json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({g: sum(1 for r in records if r['group'] == g)
                      for g in sorted({r['group'] for r in records})}, indent=1))


if __name__ == '__main__':
    main()
