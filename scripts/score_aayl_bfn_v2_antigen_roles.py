"""AAYL / AbBiBench functional retrospective with the corrected-role BFN judge.

Repeats the 2026-09-16 retrospective (508 H3-only variants, two parent
lineages, one antigen, AlphaSeq-estimated Kd) but with the BFN judge built with
the corrected Fragment encoding (H -> Heavy, L -> Light, A -> Antigen).

Reference (frozen, ProteinMPNN-based):
  results/aayl_original_retrospective_v1/report.json
  AAYL49: ECLS Spearman -0.1214 | negative complex NLL +0.0635
  AAYL51: ECLS Spearman -0.0715 | negative complex NLL +0.1318

Endpoint: 9 - median(original log10 estimated Kd in nM); higher = stronger.
Desired direction: Spearman(ECLS, endpoint) < 0 (lower ECLS = better binder).

Outputs: results/aayl_bfn_v2_antigen_roles/{scores.json,report.json}
Frozen artifacts are never modified.
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

COHORT = ROOT / 'data/aayl_original_retrospective_v1/cohort.json'
STRUCT_DIR = ROOT / 'data/abbibench_structural_audit_v1/structures'
FROZEN = ROOT / 'results/aayl_original_retrospective_v1/scores.json'
DEFAULT_OUT = ROOT / 'results/aayl_bfn_v2_antigen_roles'
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261101


def rankdata(values):
    order = np.argsort(np.asarray(values, dtype=float), kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # average ties
    sorted_vals = np.asarray(values, dtype=float)[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float('nan')


def bootstrap_spearman(x, y, rng):
    n = len(x)
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 3:
            continue
        values.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = read(COHORT)
    frozen = {rec['poi']: rec for rec in read(FROZEN)['records']}

    import torch
    from modules.bfn_loader import build_region_batch, load_bfn, _get_checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _cfg = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    records = []
    for family, info in cohort['structures'].items():
        pdb = STRUCT_DIR / Path(info['file']).name
        heavy = info['heavy_chain']
        idx = info['h3_indices_zero_based_observed_heavy']
        region_spec = f"{heavy}:{idx[0] + 1}-{idx[-1] + 1}"
        expected_h3 = info['cdr_h3_sequence']
        variants = [v for v in cohort['eligible'] if v['family'] == family]

        complex_batch = build_region_batch(
            str(pdb), region_spec, context_chains=[
                c for c in info['sequences'] if c != heavy],
            antigen_chains=list(info['antigen_chains']), device=device)
        stripped_batch = build_region_batch(
            str(pdb), region_spec, context_chains=[], antigen_chains=[],
            device=device)
        assert int(complex_batch['generate_flag'].sum()) == len(expected_h3), (
            family, int(complex_batch['generate_flag'].sum()), len(expected_h3))
        print(f'{family}: {len(variants)} variants, region {region_spec}, '
              f'H3 len {len(expected_h3)}', flush=True)

        for variant in variants:
            h3 = variant['h3_sequence']
            if len(h3) != len(expected_h3):
                raise RuntimeError(f'{family} {variant["poi"]}: H3 length {len(h3)}')
            complex_nll = float(np.mean([
                candidate_nll(model, complex_batch, h3, seed, device)[0]
                for seed in SCORING_SEEDS]))
            stripped_nll = float(np.mean([
                candidate_nll(model, stripped_batch, h3, seed, device)[0]
                for seed in SCORING_SEEDS]))
            frozen_rec = frozen.get(variant['poi'], {})
            records.append({
                'family': family,
                'poi': variant['poi'],
                'h3_sequence': h3,
                'endpoint': float(variant['endpoint']),
                'bfn_complex_nll': complex_nll,
                'bfn_stripped_nll': stripped_nll,
                'bfn_ecls': complex_nll - stripped_nll,
                'frozen_proteinmpnn_ecls': frozen_rec.get('ECLS'),
                'frozen_negative_complex_nll': frozen_rec.get('negative_complex_NLL'),
            })
        print(f'{family}: scored {len(variants)}', flush=True)

    freeze(out_dir / 'scores.json', {'records': records, 'checkpoint': checkpoint_path,
                                     'scoring_seeds': SCORING_SEEDS,
                                     'role_encoding': 'H=Heavy(1), L=Light(2), A=Antigen(3)'})

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = []
    for family in sorted({r['family'] for r in records}):
        rows = [r for r in records if r['family'] == family]
        endpoint = [r['endpoint'] for r in rows]
        bfn_ecls = [r['bfn_ecls'] for r in rows]
        bfn_neg_complex = [-r['bfn_complex_nll'] for r in rows]
        frozen_ecls = [r['frozen_proteinmpnn_ecls'] for r in rows
                       if r['frozen_proteinmpnn_ecls'] is not None]
        summary.append({
            'family': family,
            'n_variants': len(rows),
            'spearman': {
                'bfn_ecls_vs_endpoint': spearman(bfn_ecls, endpoint),
                'bfn_negative_complex_nll_vs_endpoint': spearman(bfn_neg_complex, endpoint),
                'frozen_proteinmpnn_ecls_vs_endpoint': (
                    spearman(frozen_ecls,
                             [r['endpoint'] for r in rows
                              if r['frozen_proteinmpnn_ecls'] is not None])
                    if len(frozen_ecls) > 2 else None),
            },
            'bootstrap_ci_bfn_ecls': bootstrap_spearman(bfn_ecls, endpoint, rng),
            'bfn_ecls_mean': float(np.mean(bfn_ecls)),
            'bfn_ecls_sd': float(np.std(bfn_ecls)),
        })
        print(json.dumps(summary[-1], indent=1), flush=True)

    report = {
        'schema_version': 'aayl_bfn_v2_antigen_roles',
        'classification': 'retrospective_single_antigen_descriptive',
        'endpoint': '9 - median(original log10 estimated Kd in nM); higher = stronger',
        'desired_direction': 'spearman(ecls, endpoint) < 0',
        'references_frozen_proteinmpnn': {
            'AAYL49': {'ECLS': -0.12137340587783532,
                       'negative_complex_NLL': 0.06346850696059449},
            'AAYL51': {'ECLS': -0.07147922607901694,
                       'negative_complex_NLL': 0.1317821126759523},
        },
        'families': summary,
        'independent_antigens': 1,
        'structure_source': 'AF3 predicted parent complexes (AbBiBench)',
        'limitations': [
            'one antigen, two parent lineages',
            'AlphaSeq-estimated Kd, not SPR/ITC',
            'AF3 predicted rather than experimental coordinates',
            'retrospective and not blinded (labels previously inspected)',
            'complete-readings subset excludes weak/no-reading variants',
        ],
    }
    freeze(out_dir / 'report.json', report)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
