#!/usr/bin/env python
"""Standalone AF2 batch validation — runs in bfn-msa conda env (no torch required).

Expects pre-grafted full antibody sequences in the input JSON.
Only depends on af2_jax_runner (JAX + haiku).

Usage:
  conda activate bfn-msa
  python af2_batch_validate.py input.json output.json
"""

import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))

from af2_jax_runner import run_multimer_prediction


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input.json> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        data = json.load(f)

    epi_seq = data['epi_seq']
    designs = data['designs']
    num_recycle = data.get('num_recycle', 3)
    data_dir = data.get('data_dir') or os.path.expanduser('~/.cache/colabfold')

    print(f'AF2 batch validation: {len(designs)} designs, '
          f'epitope={len(epi_seq)} AA, recycle={num_recycle}')
    print()

    results = []
    n = len(designs)
    for i, d in enumerate(designs):
        cdr_seq = d.get('sequence', '')
        full_ab = d.get('full_ab', cdr_seq)  # Pre-grafted by Phase 1
        t0 = time.time()
        try:
            r = run_multimer_prediction(
                full_ab, epi_seq, num_recycle=num_recycle, data_dir=data_dir)
            elapsed = time.time() - t0
            if r['success']:
                results.append({
                    'rank_bfn': d.get('rank_bfn', i + 1),
                    'sequence': cdr_seq,
                    'af2_plddt': r['plddt'],
                    'af2_iptm': r['iptm'],
                    'af2_ptm': r['ptm'],
                    'af2_max_pae': r['max_pae'],
                    'af2_interface_pae': r.get('interface_pae'),
                    'af2_elapsed': elapsed,
                    'success': True,
                })
                print(f'  [{i+1}/{n}] AF2 pLDDT={r["plddt"]:.4f} '
                      f'ipTM={r["iptm"]:.4f} ({elapsed:.0f}s)')
            else:
                results.append({
                    'rank_bfn': d.get('rank_bfn', i + 1),
                    'sequence': cdr_seq,
                    'success': False,
                    'error': r.get('error', 'unknown'),
                    'af2_elapsed': elapsed,
                })
                print(f'  [{i+1}/{n}] FAILED: {r.get("error", "?")[:80]}')
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                'rank_bfn': d.get('rank_bfn', i + 1),
                'sequence': cdr_seq,
                'success': False,
                'error': str(e)[:200],
                'af2_elapsed': elapsed,
            })
            print(f'  [{i+1}/{n}] EXCEPTION: {e}')

    # Summary
    success_results = [r for r in results if r.get('success')]
    summary = {
        'n_total': len(results),
        'n_success': len(success_results),
        'n_failed': len(results) - len(success_results),
    }
    if success_results:
        summary['af2_plddt_mean'] = sum(r['af2_plddt'] for r in success_results) / len(success_results)
        summary['af2_iptm_mean'] = sum(r['af2_iptm'] for r in success_results) / len(success_results)
        summary['af2_ptm_mean'] = sum(r['af2_ptm'] for r in success_results) / len(success_results)
        ipae_vals = [r['af2_interface_pae'] for r in success_results if r.get('af2_interface_pae') is not None]
        if ipae_vals:
            summary['af2_interface_pae_mean'] = sum(ipae_vals) / len(ipae_vals)

        ranked = sorted(success_results, key=lambda r: r['af2_iptm'], reverse=True)
        for i, r in enumerate(ranked):
            r['rank_af2'] = i + 1
        summary['top_af2_iptm'] = ranked[0]['af2_iptm']
        summary['top_af2_plddt'] = ranked[0]['af2_plddt']
        summary['top_sequence'] = ranked[0]['sequence']

    output = {
        'input_config': data.get('_config', {}),
        'summary': summary,
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nAF2 validation done. {summary["n_success"]}/{summary["n_total"]} succeeded.')
    if success_results:
        print(f'Top AF2 ipTM: {summary["top_af2_iptm"]:.4f}')
        print(f'Top AF2 pLDDT: {summary["top_af2_plddt"]:.4f}')
    print(f'Saved to: {output_path}')


if __name__ == '__main__':
    main()
