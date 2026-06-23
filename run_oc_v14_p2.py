#!/usr/bin/env python
"""Phase 2: AF2 validation + overconfidence (OC) ratio for V14 designs.

Reads oc_validation_results/oc_v14_phase1.json, runs AF2 multimer on each
design, and computes the OC ratio = bfn_iptm_mean / af2_iptm_mean.

Target: OC iptm ≤ 2x (was 9.47x at V12). Also reports the within-design BFN
iptm standard deviation (V14 should raise it above ~0.05, vs V12's <1e-4) and
the Spearman rank correlation between BFN and AF2 iptm across designs on the
same scaffold (V14 target ρ > 0.4).

Usage:
  python run_oc_v14_p2.py
"""
import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import numpy as np

INPUT_PATH = 'oc_validation_results/oc_v14_phase1.json'
OUTPUT_PATH = 'oc_validation_results/oc_v14_phase2.json'


def _spearman(a, b):
    """Spearman ρ between two equal-length lists (simple rank-correlation)."""
    if len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float(np.dot(ra, rb) / denom) if denom > 0 else float('nan')


def run_phase2(input_path=None, output_path=None):
    input_path = input_path or INPUT_PATH
    output_path = output_path or OUTPUT_PATH

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    from af2_jax_runner import validate_antibody_epitope

    epi_seq = data['epi_seq']
    models = data['models']

    for model_name, model_data in models.items():
        af2_inputs = model_data.get('af2_input', [])
        if not af2_inputs:
            print(f'  {model_name}: no AF2 inputs')
            continue

        print(f'\n{"=" * 60}')
        print(f'  Phase 2: AF2 Validation — {model_data["label"]}')
        print(f'  {len(af2_inputs)} designs, recycle=3')
        print(f'{"=" * 60}')

        results = []
        for i, inp in enumerate(af2_inputs):
            full_ab = inp['full_ab']
            print(f'  [{i+1}/{len(af2_inputs)}] {len(full_ab)}+{len(epi_seq)} AA... ',
                  end='', flush=True)
            t0 = time.time()
            try:
                af2_r = validate_antibody_epitope(full_ab, epi_seq, num_recycle=3, verbose=False)
                dt = time.time() - t0
                if af2_r.get('success'):
                    print(f'ipTM={af2_r["iptm"]:.3f} pLDDT={af2_r["plddt"]:.3f} ({dt:.0f}s)')
                else:
                    print(f'FAILED ({dt:.0f}s)')
                result = dict(inp)
                result['af2_plddt'] = af2_r.get('plddt')
                result['af2_iptm'] = af2_r.get('iptm')
                result['af2_ptm'] = af2_r.get('ptm')
                result['af2_interface_pae'] = af2_r.get('interface_pae')
                results.append(result)
            except Exception as e:
                print(f'ERROR: {e}')

        model_data['af2_results'] = results

        valid = [r for r in results if r.get('af2_iptm') is not None]
        if valid:
            ratios_iptm = [r['bfn_iptm'] / max(r['af2_iptm'], 0.001) for r in valid]
            ratios_plddt = [r['bfn_plddt'] / max(r['af2_plddt'], 0.001) for r in valid]
            model_data['af2_iptm_mean'] = float(np.mean([r['af2_iptm'] for r in valid]))
            model_data['af2_plddt_mean'] = float(np.mean([r['af2_plddt'] for r in valid]))
            model_data['oc_ratio_iptm_mean'] = float(np.mean(ratios_iptm))
            model_data['oc_ratio_plddt_mean'] = float(np.mean(ratios_plddt))
            model_data['bfn_iptm_within_design_std'] = float(
                np.std([r['bfn_iptm'] for r in valid]))
            model_data['spearman_bfn_vs_af2_iptm'] = _spearman(
                [r['bfn_iptm'] for r in valid], [r['af2_iptm'] for r in valid])

            print('\n  ── V14 Summary ──')
            print(f'  BFN ipTM mean:     {model_data["bfn_iptm_mean"]:.4f}')
            print(f'  AF2 ipTM mean:     {model_data["af2_iptm_mean"]:.4f}')
            print(f'  OC ratio ipTM:     {model_data["oc_ratio_iptm_mean"]:.2f}x   (target ≤ 2.0x)')
            print(f'  OC ratio pLDDT:    {model_data["oc_ratio_plddt_mean"]:.2f}x')
            print(f'  within-design std: {model_data["bfn_iptm_within_design_std"]:.6f}  (target > 0.05)')
            print(f'  Spearman ρ (BFN vs AF2 ipTM): {model_data["spearman_bfn_vs_af2_iptm"]:.3f}  (target > 0.4)')
        else:
            print('\n  No successful AF2 results!')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'\n  Phase 2 output saved to: {output_path}')
    return data


if __name__ == '__main__':
    run_phase2()
