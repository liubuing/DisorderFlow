#!/usr/bin/env python3
"""AF2 validation on V17i designs via WSL GPU batch.

Reads oc_v17i_phase1_complex.json, extracts grafted sequences,
runs AF2 via WSL, computes OC + Spearman ρ.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
from collections import defaultdict

INPUT = 'oc_validation_results/oc_v17i_phase1_complex.json'
OUTPUT = 'oc_validation_results/oc_v17i_phase2_complex.json'

with open(INPUT, encoding='utf-8') as f:
    designs = json.load(f)
print(f'Loaded {len(designs)} designs')

# Get epitope sequence (Aβ42)
EPITOPE_SEQ = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'

# Get full grafted sequences from PDBs
def get_pdb_sequence(pdb_path):
    aa3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E',
              'GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
              'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
              'TYR':'Y','VAL':'V'}
    seq, seen = [], set()
    with open(pdb_path) as fh:
        for l in fh:
            if l.startswith('ATOM') and l[12:16].strip()=='CA':
                ri = l[22:27]
                if ri not in seen:
                    seen.add(ri)
                    seq.append(aa3to1.get(l[17:20].strip(),'X'))
    return ''.join(seq)

seqs = []
for d in designs:
    if 'grafted_pdb' in d and os.path.exists(d['grafted_pdb']):
        seqs.append(get_pdb_sequence(d['grafted_pdb']))
    else:
        seqs.append(d.get('cdr_seq',''))
    # Also store the epitope
    d['epitope_seq'] = EPITOPE_SEQ

print(f'Extracted {len(seqs)} full sequences')
print(f'Sequence lengths: {set(len(s) for s in seqs)}')

# Run AF2 via WSL
from build_design_variant_dataset import _batch_af2_wsl
print('Running AF2 via WSL GPU...')
t0 = time.time()
af2_results = _batch_af2_wsl(seqs, EPITOPE_SEQ, num_recycle=1,
                              warmup_seq=seqs[0] if seqs else None)
elapsed = time.time() - t0
print(f'AF2 done in {elapsed:.0f}s')

# Merge results
af2_iptms, bfn_iptms = [], []
for i, d in enumerate(designs):
    r = af2_results[i] if i < len(af2_results) else None
    d['af2_success'] = r is not None and r.get('success', False)
    d['af2_iptm'] = r.get('iptm', 0.0) if r else 0.0
    d['af2_plddt'] = r.get('plddt', 0.0) if r else 0.0
    d['af2_pae'] = r.get('pae', 0.0) if r else 0.0
    if d['af2_success']:
        af2_iptms.append(d['af2_iptm'])
        bfn_iptms.append(d['bfn_iptm'])

# Compute metrics
oc_ratios = [b/a for b,a in zip(bfn_iptms, [d['af2_iptm'] for d in designs if d['af2_success']]) if a>0.001]

def spearman(a,b):
    if len(a)<3: return float('nan')
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra-=ra.mean(); rb-=rb.mean()
    d2=((ra-rb)**2).sum()
    return 1-6*d2/(len(a)*(len(a)**2-1)) if len(a)>1 else float('nan')

rho=spearman(np.array(bfn_iptms), np.array([d['af2_iptm'] for d in designs if d['af2_success']]))

print(f'\n=== V17i AF2 Results ({len(designs)} designs) ===')
print(f'Success: {sum(1 for d in designs if d["af2_success"])}/{len(designs)}')
if af2_iptms:
    print(f'AF2 ipTM: mean={np.mean(af2_iptms):.4f} max={np.max(af2_iptms):.4f} min={np.min(af2_iptms):.4f}')
    print(f'BFN ipTM: mean={np.mean(bfn_iptms):.4f}')
    if oc_ratios: print(f'OC ratio: {np.mean(oc_ratios):.2f}x')
    print(f'Spearman ρ: {rho:.3f}')

    # Per-group breakdown
    groups=defaultdict(list)
    for d in designs:
        key=(os.path.basename(d['scaffold']),os.path.basename(d['antigen']))
        groups[key].append(d)
    for key,gd in sorted(groups.items()):
        ga=[d['af2_iptm'] for d in gd if d['af2_success']]
        if ga: print(f'  {key[0]}+{key[1]}: AF2 iptm mean={np.mean(ga):.4f} max={np.max(ga):.4f}')

# Save
os.makedirs('oc_validation_results',exist_ok=True)
with open(OUTPUT,'w',encoding='utf-8') as f:
    json.dump({
        'designs':designs,
        'summary':{
            'n_total':len(designs),
            'n_success':sum(1 for d in designs if d['af2_success']),
            'af2_iptm_mean':float(np.mean(af2_iptms)) if af2_iptms else 0,
            'af2_iptm_max':float(np.max(af2_iptms)) if af2_iptms else 0,
            'bfn_iptm_mean':float(np.mean(bfn_iptms)) if bfn_iptms else 0,
            'oc_ratio_mean':float(np.mean(oc_ratios)) if oc_ratios else 0,
            'spearman_rho':float(rho) if not np.isnan(rho) else 0,
        },
    },f,indent=2)
print(f'\nSaved to {OUTPUT}')
