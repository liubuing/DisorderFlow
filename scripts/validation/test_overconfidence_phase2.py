#!/usr/bin/env python
"""Phase 2: AF2 validation on designs from Phase 1. Run in bfn-msa conda env.

Usage:
  conda activate bfn-msa
  python test_overconfidence_phase2.py
"""
import os, sys, json, time
os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import numpy as np
from af2_jax_runner import run_multimer_prediction

PHASE1_FILE = '/tmp/overconfidence_phase1.json'

with open(PHASE1_FILE) as f:
    data = json.load(f)

epi_seq = data['epi_seq']
designs = data['designs']

print('=' * 80)
print('  Phase 2: AF2 Validation')
print('=' * 80)
print(f'  Epitope: {len(epi_seq)} AA')
print(f'  AF2 recycle: 1')
print()

results = {}

for label in ['phase3', 'phase4']:
    print(f'--- {label} ---')
    af2_input = designs[label]['af2_input']
    label_results = []

    for item in af2_input:
        full_ab = item['full_ab']
        print(f'  [{item["rank"]}] {len(full_ab)}+{len(epi_seq)} AA ...', end=' ', flush=True)
        t0 = time.time()
        r = run_multimer_prediction(full_ab, epi_seq, num_recycle=1)
        elapsed = time.time() - t0

        if r['success']:
            label_results.append({
                **item,
                'af2_plddt': r['plddt'],
                'af2_iptm': r['iptm'],
                'af2_ptm': r['ptm'],
                'af2_max_pae': r['max_pae'],
                'af2_interface_pae': r.get('interface_pae'),
                'af2_elapsed': elapsed,
            })
            print(f'pLDDT={r["plddt"]:.3f} ipTM={r["iptm"]:.3f} ({elapsed:.0f}s)')
        else:
            label_results.append({**item, 'af2_error': r.get('error', '?')})
            print(f'FAILED: {r.get("error", "?")[:80]}')

    # Compute ratios
    ratios_plddt = []
    ratios_iptm = []
    for item in label_results:
        if 'af2_plddt' in item and item['bfn_plddt'] > 0:
            ratios_plddt.append(item['bfn_plddt'] / max(item['af2_plddt'], 0.001))
        if 'af2_iptm' in item and item['bfn_iptm'] > 0:
            ratios_iptm.append(item['bfn_iptm'] / max(item['af2_iptm'], 0.001))

    results[label] = {
        'af2_results': label_results,
        'af2_plddt_mean': float(np.mean([r['af2_plddt'] for r in label_results if 'af2_plddt' in r])),
        'af2_iptm_mean': float(np.mean([r['af2_iptm'] for r in label_results if 'af2_iptm' in r])),
        'oc_ratio_plddt_mean': float(np.mean(ratios_plddt)) if ratios_plddt else None,
        'oc_ratio_iptm_mean': float(np.mean(ratios_iptm)) if ratios_iptm else None,
    }
    print(f'  AF2 pLDDT mean: {results[label]["af2_plddt_mean"]:.4f}')
    print(f'  AF2 ipTM mean:  {results[label]["af2_iptm_mean"]:.4f}')
    print(f'  OC ratio pLDDT: {results[label]["oc_ratio_plddt_mean"]:.2f}x (BFN/AF2)')
    print(f'  OC ratio ipTM:  {results[label]["oc_ratio_iptm_mean"]:.2f}x (BFN/AF2)')
    print()

# Save
phase2_path = '/tmp/overconfidence_phase2.json'
with open(phase2_path, 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print('=' * 80)
print('  OVERCONFIDENCE COMPARISON')
print('=' * 80)

p3 = results['phase3']
p4 = results['phase4']

for metric, label_text in [
    ('oc_ratio_plddt_mean', 'pLDDT overconfidence (BFN/AF2)'),
    ('oc_ratio_iptm_mean', 'ipTM overconfidence (BFN/AF2)'),
]:
    v3 = p3.get(metric)
    v4 = p4.get(metric)
    if v3 and v4:
        delta = v4 - v3
        pct = f'{delta/v3*100:+.1f}%' if v3 != 0 else 'N/A'
        print(f'  {label_text}: {v3:.2f}x -> {v4:.2f}x  ({pct})')
    else:
        print(f'  {label_text}: N/A')

if p3.get('oc_ratio_plddt_mean') and p4.get('oc_ratio_plddt_mean'):
    if p4['oc_ratio_plddt_mean'] < p3['oc_ratio_plddt_mean']:
        reduction = (1 - p4['oc_ratio_plddt_mean']/p3['oc_ratio_plddt_mean']) * 100
        print(f'\n  Overconfidence REDUCED by {reduction:.1f}% —Phase 4 negative sample training WORKS')
    else:
        print(f'\n  Overconfidence did NOT decrease —may need more negative samples or higher weight')

print(f'\n  Results saved to: {phase2_path}')
print(f'{"="*80}')
