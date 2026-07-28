#!/usr/bin/env python
"""Quick overconfidence test: Phase 3 vs Phase 4 negative sample training.

Phase 1 (this script, venv): BFN design with both checkpoints →save to JSON
Phase 2 (bfn-msa env): AF2 validation on saved designs →save AF2 scores
Phase 3 (this script): Load results, compare overconfidence ratios
"""
import os, sys, json, time
os.chdir('/home/liubuzi/disorderflow-main/disorderflow-main')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import torch
import numpy as np
import yaml
from bfn_loader import run_bfn_design
from idp_antibody_design import _extract_sequence_from_pdb, _parse_cdr_ranges, graft_cdrs

# Config
SCAFFOLD_PDB = 'data/misfolding_targets/5IMK.pdb'
SCAFFOLD_CHAIN = 'B'
TARGET_PDB = 'data/misfolding_targets/2NAO_model1_A_1-42.pdb'
TARGET_CHAIN = 'A'
CDR_SPEC = 'B:26-33,51-58,97-113'
N_SAMPLES = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PHASE3_CKPT = 'logs/bfn_phase3_disorder_2026_05_31__14_52_47/checkpoints/best.pt'
PHASE4_CKPT = 'logs/bfn_phase4_negative_xpu_2026_05_31__18_27_56/checkpoints/best.pt'

# Extract sequences
epi_seq = _extract_sequence_from_pdb(TARGET_PDB, TARGET_CHAIN)
scaffold_seq = _extract_sequence_from_pdb(SCAFFOLD_PDB, SCAFFOLD_CHAIN)
cdr_ranges = _parse_cdr_ranges(CDR_SPEC)

print('=' * 80)
print('  Phase 1: BFN Design —Phase 3 vs Phase 4')
print('=' * 80)
print(f'  Scaffold: {len(scaffold_seq)} AA')
print(f'  Target:   {len(epi_seq)} AA')
print()

designs_data = {}

for label, ckpt_path in [('phase3', PHASE3_CKPT), ('phase4', PHASE4_CKPT)]:
    print(f'--- {label} ---')
    print(f'  Checkpoint: {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}')

    # Update config
    cfg_path = 'app_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    orig_ckpt = cfg['models']['bfn']['checkpoint']
    cfg['models']['bfn']['checkpoint'] = ckpt_path
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f)

    try:
        t0 = time.time()
        designs = run_bfn_design(SCAFFOLD_PDB, CDR_SPEC, num_samples=N_SAMPLES,
                                 stochastic=True, context_chains=None, device=DEVICE)
        elapsed = time.time() - t0

        bfn_plddts = [d['plddt'] for d in designs]
        bfn_iptms = [d['iptm'] for d in designs]

        print(f'  Generated {len(designs)} designs in {elapsed:.0f}s')
        print(f'  BFN pLDDT: {np.mean(bfn_plddts):.4f} +/- {np.std(bfn_plddts):.4f}')
        print(f'  BFN ipTM:  {np.mean(bfn_iptms):.4f} +/- {np.std(bfn_iptms):.4f}')

        # Prepare AF2 input: graft CDRs onto scaffold for top-N
        af2_input = []
        for i, d in enumerate(designs[:5]):
            full_ab, mutations = graft_cdrs(scaffold_seq, d['sequence'], cdr_ranges)
            af2_input.append({
                'rank': i + 1,
                'sequence': d['sequence'],
                'full_ab': full_ab,
                'bfn_plddt': d['plddt'],
                'bfn_iptm': d['iptm'],
                'bfn_ppl': d.get('ppl', 0),
                'mutations': len(mutations),
            })
            print(f'  [{i+1}] BFN pLDDT={d["plddt"]:.4f} ipTM={d["iptm"]:.4f} '
                  f'PPL={d.get("ppl",0):.1f} | {len(mutations)} mutations '
                  f'| CDR={d["sequence"][:30]}...')

        designs_data[label] = {
            'ckpt': ckpt_path,
            'n_designs': len(designs),
            'bfn_plddt_mean': float(np.mean(bfn_plddts)),
            'bfn_plddt_std': float(np.std(bfn_plddts)),
            'bfn_iptm_mean': float(np.mean(bfn_iptms)),
            'bfn_iptm_std': float(np.std(bfn_iptms)),
            'af2_input': af2_input,
        }

    finally:
        cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f)

    print()

# Save for Phase 2
phase1_path = '/tmp/overconfidence_phase1.json'
with open(phase1_path, 'w') as f:
    json.dump({
        'epi_seq': epi_seq,
        'scaffold_seq': scaffold_seq,
        'cdr_ranges': [[s, e, l] for s, e, l in cdr_ranges],
        'designs': designs_data,
    }, f, indent=2)

# Compare BFN self-confidence
p3 = designs_data['phase3']
p4 = designs_data['phase4']

print('=' * 80)
print('  BFN Self-Confidence Comparison')
print('=' * 80)
print(f'  {"Metric":<20} {"Phase 3":<15} {"Phase 4":<15} {"Change":<15}')
print(f'  {"-"*65}')
for key, label_text in [('bfn_plddt_mean', 'BFN pLDDT mean'),
                         ('bfn_iptm_mean', 'BFN ipTM mean')]:
    v3 = p3[key]
    v4 = p4[key]
    delta = v4 - v3
    pct = f'{delta/v3*100:+.1f}%' if v3 != 0 else 'N/A'
    print(f'  {label_text:<20} {v3:<15.4f} {v4:<15.4f} {delta:+.4f} ({pct})')

print()
if p4['bfn_plddt_mean'] < p3['bfn_plddt_mean']:
    print(f'  BFN pLDDT DECREASED: Phase 4 is less overconfident')
    reduction = (1 - p4['bfn_plddt_mean']/p3['bfn_plddt_mean']) * 100
    print(f'  Overconfidence reduction: {reduction:.1f}%')
else:
    print(f'  BFN pLDDT did NOT decrease —negative samples may need more weight')

print()
print(f'  Phase 1 data saved to: {phase1_path}')
print(f'  Run Phase 2: conda activate bfn-msa && python test_overconfidence_phase2.py')
print(f'{"="*80}')
