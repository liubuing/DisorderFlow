import os
#!/usr/bin/env python
"""FixBB vs Complex overconfidence comparison.
THE definitive test: does Complex mode actually use the epitope?
"""
import sys, os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import yaml, time, json, numpy as np

SCAFFOLD_PDB = 'data/misfolding_targets/5IMK.pdb'
SCAFFOLD_CHAIN = 'B'
CDR_SPEC = 'B:26-33,51-58,97-113'
V11_CKPT = 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt'

def auto_detect_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def run_test(label, mode, pdb_path, context_chains):
    from bfn_loader import run_bfn_design, load_bfn

    cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    with open(cfg_path) as f:
        app_cfg = yaml.safe_load(f)
    orig_ckpt = app_cfg['models']['bfn']['checkpoint']
    app_cfg['models']['bfn']['checkpoint'] = V11_CKPT
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)

    try:
        import bfn_loader
        bfn_loader._bfn_model = None
        bfn_loader._bfn_config = None

        t0 = time.time()
        designs = run_bfn_design(
            pdb_path, CDR_SPEC, num_samples=10, stochastic=True,
            context_chains=context_chains, device=auto_detect_device(),
        )
        dt = time.time() - t0

        plddts = [d['plddt'] for d in designs]
        iptms = [d['iptm'] for d in designs]
        ppls = [d.get('ppl', 0) for d in designs]
        entropies = [d.get('entropy', 0) for d in designs]
        seqs = [d['sequence'] for d in designs]

        print(f'  [{label}] {mode}: {len(designs)} designs in {dt:.0f}s')
        print(f'    pLDDT: {np.mean(plddts):.4f} std={np.std(plddts):.8f}')
        print(f'    ipTM:  {np.mean(iptms):.4f} std={np.std(iptms):.8f}')
        print(f'    PPL:   {np.mean(ppls):.1f} [{min(ppls):.0f}-{max(ppls):.0f}]')
        print(f'    Entropy: {np.mean(entropies):.4f} std={np.std(entropies):.4f}')

        # Show top sequence
        print(f'    Top3 seqs: {seqs[0][:30]}..., {seqs[1][:30]}..., {seqs[2][:30]}...')

        return {
            'mode': mode, 'n_designs': len(designs),
            'plddt_mean': float(np.mean(plddts)), 'plddt_std': float(np.std(plddts)),
            'iptm_mean': float(np.mean(iptms)), 'iptm_std': float(np.std(iptms)),
            'ppl_mean': float(np.mean(ppls)), 'ppl_range': [float(min(ppls)), float(max(ppls))],
            'entropy_mean': float(np.mean(entropies)),
            'sequences': [s[:30] for s in seqs[:3]],
        }
    finally:
        app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(app_cfg, f, default_flow_style=False)

print('=' * 70)
print('  CRITICAL TEST: FixBB vs Complex Mode')
print('  Does BFN actually use epitope context?')
print('=' * 70)

# Test 1: 5IMK FixBB (chain B only)
print('\n── Test 1: FixBB (scaffold only, no antigen)')
r1 = run_test('FixBB', 'FixBB', SCAFFOLD_PDB, [])

# Test 2: 5IMK with chain A as context
# 5IMK chain A is a 349aa protein that the nanobody was co-crystallized with
print('\n── Test 2: Complex (scaffold + co-crystallized protein A)')
r2 = run_test('5IMK_native', 'Complex', SCAFFOLD_PDB, ['A'])

# Test 3: Abeta42 as epitope (build combined PDB)
print('\n── Test 3: Complex (scaffold + Abeta42 epitope)')
from antibody_epitope_complex import position_epitope
abeta_info = position_epitope(
    SCAFFOLD_PDB, 'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
    scaffold_chain='B', epitope_chain='A', distance=6.0,
)
print(f'  Combined PDB: {abeta_info["pdb_path"]}')
r3 = run_test('Abeta42', 'Complex', abeta_info['pdb_path'], ['A'])

# Test 4: VEGF as epitope (ordered protein control)
print('\n── Test 4: Complex (scaffold + VEGF ordered protein)')
vegf_info = position_epitope(
    SCAFFOLD_PDB, 'data/antibody_complexes/1BJ1.pdb',
    scaffold_chain='B', epitope_chain='V', distance=6.0,
)
print(f'  Combined PDB: {vegf_info["pdb_path"]}')
r4 = run_test('VEGF', 'Complex', vegf_info['pdb_path'], ['V'])

# ── Comparison ──
print('\n' + '=' * 70)
print('  COMPARISON: Are designs DIFFERENT for different targets?')
print('=' * 70)
print(f'  {"Mode":<25s} {"pLDDT std":<14s} {"ipTM std":<14s} {"PPL range":<14s} {"Top sequence":<35s}')
print(f'  {"─"*25} {"─"*14} {"─"*14} {"─"*14} {"─"*35}')
for r in [r1, r2, r3, r4]:
    ppl_r = '%d-%d' % (int(r['ppl_range'][0]), int(r['ppl_range'][1]))
    print('  %-25s %-14s %-14s %-14s %-35s' % (
        r['mode'], '%.8f' % r['plddt_std'], '%.8f' % r['iptm_std'],
        ppl_r, r['sequences'][0]))

# Key question: are the sequences different?
seqs_by_mode = {}
for r in [r1, r2, r3, r4]:
    seqs_by_mode[r['mode']] = set(tuple(s) for s in r['sequences'])

all_same = True
for m1 in seqs_by_mode:
    for m2 in seqs_by_mode:
        if m1 < m2:
            overlap = seqs_by_mode[m1] & seqs_by_mode[m2]
            if overlap:
                print(f'\n  ⚠ {m1} and {m2} share {len(overlap)} identical sequences!')
                all_same = False

if all_same or len(set(frozenset(v) for v in seqs_by_mode.values())) == 1:
    print(f'\n  ❌ ALL MODES PRODUCE IDENTICAL SEQUENCES!')
    print(f'  BFN is NOT using epitope context to differentiate designs.')
else:
    # Check if sequence sets differ between modes
    unique_seqs = set()
    for mode, seqs in seqs_by_mode.items():
        for s in seqs:
            unique_seqs.add(tuple(s))
    print(f'\n  ✓ Different modes produce different sequences ({len(unique_seqs)} unique)')

print(f'\n  ── Key Question ──')
print(f'  FixBB vs Complex: does epitope context change BFN design output?')
print(f'  If NOT → epitope integration is broken → need to fix model, not scoring')
print(f'  If YES → Complex mode works → need to fix confidence scoring for Complex mode')
