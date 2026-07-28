#!/usr/bin/env python
"""B1 diagnostic: why does within-design ipTM collapse to identical values?

Loads the V14-unfreeze best.pt and runs design sampling on the OC scaffold
(5IMK + Abeta42), capturing the decomposed ipTM components per design:
    iptm_seq  = v12_iptm(pooled sequence embedding)   -- should vary w/ CDR seq
    iptm_bb   = v14_iptm_bb(pooled backbone features) -- suspect: fixBB→constant
    pred_iptm = sigmoid(iptm_seq + iptm_bb)

If iptm_bb is identical across designs → backbone pathway is blind to CDR
variants (root: fixBB design keeps backbone fixed, v14_iptm_bb reads pooled
encoder features that barely change). If iptm_seq is identical → the sequence
pathway is also flattened. This tells us WHERE to fix design-time discrimination.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

CKPT = os.environ.get('V14_CKPT',
    'logs/bfn_v14_grouped_conf_xpu_2026_06_26__01_06_36/checkpoints/best.pt')

# Point app_config at the V14-unfreeze checkpoint so load_bfn picks it up.
import yaml
cfg_path = 'app_config.yaml'
with open(cfg_path, encoding='utf-8') as f:
    app_cfg = yaml.safe_load(f)
orig = app_cfg['models']['bfn']['checkpoint']
app_cfg['models']['bfn']['checkpoint'] = CKPT
with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.dump(app_cfg, f, default_flow_style=False)

captured = []  # list of (iptm_seq, iptm_bb) per design
try:
    import bfn_loader
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    from bfn_loader import run_bfn_design, load_bfn
    from disorderflow.utils.misc import seed_all
    seed_all(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = load_bfn(device)

    # Enable component stash in receiver
    model.bfn.receiver._dbg_iptm_components = []  # signal: stash mode on

    # Patch sample to capture components after each design.
    # run_bfn_design calls model.sample(...) per design. We wrap receiver so
    # the LAST forward's components are grabbed.
    orig_forward = model.bfn.receiver.forward
    def wrapped_forward(*a, **kw):
        out = orig_forward(*a, **kw)
        # receiver stashed into self._dbg_iptm_components as (seq, bb) tensors
        st = model.bfn.receiver._dbg_iptm_components
        if isinstance(st, tuple) and len(st) == 2:
            s, b = st
            captured.append((float(s.squeeze().item()) if s.numel()==1 else float(s.mean().item()),
                             float(b.squeeze().item()) if b.numel()==1 else float(b.mean().item())))
        return out
    model.bfn.receiver.forward = wrapped_forward

    print(f'[B1] device={device}  ckpt={CKPT}')
    designs = run_bfn_design(
        'data/misfolding_targets/5IMK.pdb', 'B:26-33,51-58,97-113',
        num_samples=8, stochastic=True, context_chains=None, device=device,
    )

    print(f'\n=== {len(designs)} designs, ipTM component decomposition ===')
    print(f'{"i":>3} {"pred_iptm":>11} {"iptm_seq":>11} {"iptm_bb":>11} {"PPL":>6}')
    for i, d in enumerate(designs):
        if i < len(captured):
            s, b = captured[i]
        else:
            s, b = float('nan'), float('nan')
        print(f'{i+1:>3} {d["iptm"]:>11.6f} {s:>11.6f} {b:>11.6f} {d.get("ppl",0):>6.0f}')

    if captured:
        seqs = np.array([c[0] for c in captured])
        bbs = np.array([c[1] for c in captured])
        preds = np.array([d['iptm'] for d in designs[:len(captured)]])
        print('\n=== std across designs ===')
        print(f'  iptm_seq std = {seqs.std():.6f}  range={seqs.max()-seqs.min():.6f}')
        print(f'  iptm_bb  std = {bbs.std():.6f}   range={bbs.max()-bbs.min():.6f}')
        print(f'  pred_iptm std = {preds.std():.6f}')
        if bbs.std() < 1e-4 and seqs.std() < 1e-4:
            print('\n  >> BOTH components constant → encoder features carry no CDR signal')
        elif bbs.std() < 1e-4:
            print('\n  >> iptm_bb is constant (fixBB backbone pathway blind to CDR variants)')
        elif seqs.std() < 1e-4:
            print('\n  >> iptm_seq is constant (sequence pathway flattened)')
        else:
            print('\n  >> both vary but pred collapses → check sigmoid saturation')

finally:
    app_cfg['models']['bfn']['checkpoint'] = orig
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)
