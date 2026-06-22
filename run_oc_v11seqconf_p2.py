#!/usr/bin/env python3
"""Phase 2: AF2 validation on V11 SeqConf designs.
Run directly (no conda needed if JAX AF2 works in current env).

Usage:
  python run_oc_v11seqconf_p2.py
"""

import os, sys, json, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import numpy as np

INPUT_PATH = 'oc_validation_results/oc_v11seqconf_phase1.json'
OUTPUT_PATH = 'oc_validation_results/oc_v11seqconf_phase2.json'


def run_phase2(input_path=None, output_path=None):
    if input_path is None:
        input_path = INPUT_PATH
    if output_path is None:
        output_path = OUTPUT_PATH

    with open(input_path) as f:
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
                af2_r = validate_antibody_epitope(
                    full_ab, epi_seq, num_recycle=3, verbose=False)
                dt = time.time() - t0

                if af2_r.get('success'):
                    print(f'ipTM={af2_r["iptm"]:.3f} pLDDT={af2_r["plddt"]:.3f} '
                          f'iPAE={af2_r.get("interface_pae", "?")} ({dt:.0f}s)')
                else:
                    print(f'FAILED: {af2_r.get("error", "unknown")[:80]} ({dt:.0f}s)')

                result = dict(inp)
                result['af2_plddt'] = af2_r.get('plddt')
                result['af2_iptm'] = af2_r.get('iptm')
                result['af2_ptm'] = af2_r.get('ptm')
                result['af2_max_pae'] = af2_r.get('max_pae')
                result['af2_interface_pae'] = af2_r.get('interface_pae')
                result['af2_elapsed'] = dt
                results.append(result)

            except Exception as e:
                dt = time.time() - t0
                print(f'ERROR: {e} ({dt:.0f}s)')
                import traceback
                traceback.print_exc()

        model_data['af2_results'] = results

        # Compute OC ratios
        valid = [r for r in results if r.get('af2_iptm') is not None]
        if valid:
            ratios_plddt = [r['bfn_plddt'] / max(r['af2_plddt'], 0.001) for r in valid]
            ratios_iptm = [r['bfn_iptm'] / max(r['af2_iptm'], 0.001) for r in valid]
            model_data['af2_plddt_mean'] = float(np.mean([r['af2_plddt'] for r in valid]))
            model_data['af2_iptm_mean'] = float(np.mean([r['af2_iptm'] for r in valid]))
            model_data['oc_ratio_plddt_mean'] = float(np.mean(ratios_plddt))
            model_data['oc_ratio_iptm_mean'] = float(np.mean(ratios_iptm))
            model_data['oc_ratio_plddt_std'] = float(np.std(ratios_plddt))
            model_data['oc_ratio_iptm_std'] = float(np.std(ratios_iptm))
            model_data['af2_success_count'] = len(valid)

            print(f'\n  ── Summary ──')
            print(f'  AF2 pLDDT mean: {model_data["af2_plddt_mean"]:.4f}')
            print(f'  AF2 ipTM  mean: {model_data["af2_iptm_mean"]:.4f}')
            print(f'  BFN pLDDT mean: {model_data["bfn_plddt_mean"]:.4f}')
            print(f'  BFN ipTM  mean: {model_data["bfn_iptm_mean"]:.4f}')
            print(f'  OC ratio pLDDT: {model_data["oc_ratio_plddt_mean"]:.2f}x')
            print(f'  OC ratio ipTM:  {model_data["oc_ratio_iptm_mean"]:.2f}x')
        else:
            print(f'\n  No successful AF2 results!')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'\n  Phase 2 output saved to: {output_path}')

    return data


if __name__ == '__main__':
    run_phase2()
