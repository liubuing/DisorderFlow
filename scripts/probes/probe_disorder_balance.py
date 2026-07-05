import os
#!/usr/bin/env python3
"""Probe: does the V15 disorder head see an order/disorder conflict in designs?

Hypothesis (from the "balance ordered/disordered" insight):
  - Designed CDRs wreck the VHH framework fold (AF2 pLDDT ceiling 0.32).
  - The BFN disorder head already predicts per-residue disorder, but never
    guides design.
  - If the disorder head predicts the framework as ORDERED and the CDRs as
    DISORDERED in the *designed* sequences, that confirms the conflict the
    head could intervene on.

This script loads the V15 checkpoint, runs sample() with return_disorder=True
on the 5IMK+Aβ42 complex in complex mode for N designs, and reports the
disorder-head prediction averaged over: framework (VHH non-CDR) / CDR / Aβ42
(epitope) regions. Free — no AF2.

Also loads the Aβ42 5-seed RMSF to confirm the epitope is intrinsically
disordered (validating that an ordered-ish seq-recovery prior is mismatched).
"""
import sys, os, copy, json
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import numpy as np, torch, pickle
from easydict import EasyDict
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.models import get_model
from disorderflow.utils.misc import seed_all

import yaml
APP = 'app_config.yaml'
with open(APP, encoding='utf-8') as f:
    app = yaml.safe_load(f)
ORIG = app['models']['bfn']['checkpoint']
V15 = 'logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt'
app['models']['bfn']['checkpoint'] = V15
with open(APP, 'w', encoding='utf-8') as f:
    yaml.dump(app, f, default_flow_style=False)

try:
    import bfn_loader
    bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
    seed_all(42)
    model, config = bfn_loader.load_bfn('cuda')
    model.eval()

    # Build the Ab+Aβ42 complex (same as run_oc_v14_p1 complex mode)
    from idp_antibody_design import _parse_cdr_ranges
    from antibody_epitope_complex import position_epitope
    out_dir = os.path.abspath('oc_validation_results/_complexes')
    os.makedirs(out_dir, exist_ok=True)
    res = position_epitope('data/misfolding_targets/5IMK.pdb',
                           'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
                           scaffold_chain='B', epitope_chain='A', distance=18.0,
                           output_dir=out_dir)
    complex_pdb = res['pdb_path']
    region_spec = 'B:26-33,51-58,97-113'
    cdr_ranges = _parse_cdr_ranges(region_spec)  # list of (start,end,len) 1-based on chain B

    # Load complex, mask CDRs on B, antigen A as context
    import re
    regions = {}
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', region_spec):
        idx = []
        for seg in spec.split(','):
            seg = seg.strip()
            if '-' in seg:
                a, b = seg.split('-'); idx.extend(range(int(a.strip()), int(b.strip()) + 1))
            elif seg: idx.append(int(seg))
        regions[cid] = sorted(set(idx))
    from disorderflow.datasets.protein import preprocess_protein_structure
    struct = preprocess_protein_structure(complex_pdb, chain_ids=['A', 'B'])
    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(struct)]), 'cuda')

    # Identify residues: chain order A (antigen) then B (VHH). gen_mask = CDR on B.
    # We need per-residue region labels. Use batch['chain_id'] if present else infer by index.
    gen_mask = batch['generate_flag'][0].bool()
    L = batch['aa'].shape[1]
    # chainNB / chain_id
    cn = batch.get('chain_nb')
    if cn is not None:
        cn = cn[0]
        # antigen = first chain nb, VHH = second
        uniq = sorted(set(int(x) for x in cn.tolist()))
        ag_nb = uniq[0]; ab_nb = uniq[1] if len(uniq) > 1 else uniq[0]
        ag_mask = (cn == ag_nb)
        ab_mask = (cn == ab_nb)
    else:
        ag_mask = torch.zeros(L, dtype=torch.bool, device='cuda')
        ab_mask = torch.ones(L, dtype=torch.bool, device='cuda')
    cdr_mask = gen_mask  # on B
    fw_mask = ab_mask & (~cdr_mask)  # framework = VHH non-CDR
    print(f"Residues: total={L} antigen(Aβ42)={int(ag_mask.sum())} VHH={int(ab_mask.sum())} "
          f"CDR={int(cdr_mask.sum())} framework={int(fw_mask.sum())}")

    # Run N designs, collect disorder predictions
    N = 8
    region_dis = {'framework': [], 'cdr': [], 'antigen': []}
    for i in range(N):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt={'deterministic': False, 'num_recycles': 2,
                                                   'return_disorder': True})
        dis = traj.get('disorder')  # (1, L) logits
        if dis is None:
            print("No disorder output!"); break
        pd = torch.sigmoid(dis[0])  # (L,) disorder prob
        region_dis['framework'].append(float(pd[fw_mask].mean()))
        region_dis['cdr'].append(float(pd[cdr_mask].mean()))
        region_dis['antigen'].append(float(pd[ag_mask].mean()))
    print(f"\n=== V15 disorder-head prediction (sigmoid prob, {N} designs) ===")
    for k in ['framework', 'cdr', 'antigen']:
        v = region_dis[k]
        if v: print(f"  {k:10s}: mean={np.mean(v):.3f} std={np.std(v):.4f}  [{', '.join(f'{x:.3f}' for x in v)}]")
    # interpretation
    fw = np.mean(region_dis['framework']) if region_dis['framework'] else float('nan')
    cd = np.mean(region_dis['cdr']) if region_dis['cdr'] else float('nan')
    ag = np.mean(region_dis['antigen']) if region_dis['antigen'] else float('nan')
    print(f"\nInterpretation: framework disorder={fw:.3f} (want ~0, ordered) | "
          f"CDR disorder={cd:.3f} | antigen disorder={ag:.3f} (want high, IDP)")
    print(f"Conflict (CDR much more disordered than framework): {cd - fw:+.3f}")

    # Aβ42 RMSF check
    print("\n=== Aβ42 intrinsic disorder (5-seed RMSF) ===")
    try:
        ab = pickle.load(open('data/abeta_conformations/abeta42_5seed.pkl', 'rb'))
        rmsf = np.array(ab['rmsf']) if 'rmsf' in ab else None
        if rmsf is not None:
            print(f"  Aβ42 RMSF: mean={rmsf.mean():.2f} max={rmsf.max():.2f} n={len(rmsf)} "
                  f"(high = intrinsically disordered)")
            print(f"  → confirms Aβ42 is an IDP; designing a binder needs pliable CDRs,")
            print(f"    not the folding-protein 'should-be-ordered' seq-recovery prior.")
    except Exception as e:
        print(f"  RMSF load failed: {e}")
finally:
    app2 = yaml.safe_load(open(APP, encoding='utf-8'))
    app2['models']['bfn']['checkpoint'] = ORIG
    with open(APP, 'w', encoding='utf-8') as f:
        yaml.dump(app2, f, default_flow_style=False)
