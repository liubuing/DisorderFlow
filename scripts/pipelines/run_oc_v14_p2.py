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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

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

    epi_seq = data['epi_seq']
    models = data['models']

    # Choose AF2 backend. WSL2 GPU batch is ~100x faster than Windows CPU JAX
    # and is the validated data-pipeline path (see machine-setup memory).
    use_wsl = os.environ.get('OC_AF2_BACKEND', 'wsl').lower() in ('wsl', '1', 'true')
    if use_wsl:
        from build_design_variant_dataset import _batch_af2_wsl
        af2_recycle = int(os.environ.get('OC_AF2_RECYCLE', '3'))
        print(f'[AF2] backend = WSL2 GPU batch (recycle={af2_recycle})')

    for model_name, model_data in models.items():
        af2_inputs = model_data.get('af2_input', [])
        if not af2_inputs:
            print(f'  {model_name}: no AF2 inputs')
            continue

        print(f'\n{"=" * 60}')
        print(f'  Phase 2: AF2 Validation — {model_data["label"]}')
        print(f'  {len(af2_inputs)} designs, recycle={os.environ.get("OC_AF2_RECYCLE","3")}')
        print(f'{"=" * 60}')

        if use_wsl:
            # Batch all designs through WSL GPU at once (handles chunking internally).
            seqs = [inp['full_ab'] for inp in af2_inputs]
            print(f'  [AF2] {len(seqs)} sequences ({len(seqs[0])}+{len(epi_seq)} AA each), warmup...')
            t0 = time.time()
            af2_results = _batch_af2_wsl(seqs, epi_seq, af2_recycle, warmup_seq=seqs[0])
            print(f'  [AF2] batch done in {time.time()-t0:.0f}s')
            results = []
            for inp, af2_r in zip(af2_inputs, af2_results):
                result = dict(inp)
                if af2_r and af2_r.get('success'):
                    result['af2_plddt'] = float(np.mean(af2_r.get('plddt_seq', [0.0])))
                    result['af2_iptm'] = float(af2_r.get('iptm', 0.0))
                    result['af2_ptm'] = float(af2_r.get('ptm', 0.0))
                    result['af2_interface_pae'] = float(af2_r.get('interface_pae', 0.0))
                    print(f"  rank{inp['rank']}: AF2 ipTM={result['af2_iptm']:.4f} "
                          f"pLDDT={result['af2_plddt']:.3f}")
                else:
                    err = (af2_r or {}).get('error', 'failed')
                    result['af2_plddt'] = None
                    result['af2_iptm'] = None
                    print(f"  rank{inp['rank']}: AF2 FAILED ({err})")
                results.append(result)
        else:
            from af2_jax_runner import validate_antibody_epitope
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
