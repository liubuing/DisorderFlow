#!/usr/bin/env python
"""Confidence-collapse probe (V14).

Loads the V14 best.pt, runs the validation LMDB through the model in eval mode,
and reports the distribution of predicted ipTM/plDDT. If pred variance collapses
to ~0 across the whole val set, the confidence head is constant (collapsed),
which fully explains the 9.8x OC. If val has normal spread but the 5 OC probes
were near-identical, the issue is OOD coverage instead.

Usage:  python probe_conf_collapse.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import torch
import numpy as np
from disorderflow.utils.misc import load_config
from disorderflow.datasets import get_dataset
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.models import get_model
from torch.utils.data import DataLoader

CKPT = os.environ.get('V14_CKPT',
    'logs/bfn_v14_grouped_conf_xpu_2026_06_25__19_55_06/checkpoints/best.pt')
CFG  = 'configs/train/bfn_v14_grouped_conf_xpu.yml'

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config, _ = load_config(CFG)
    # Force eval-time recycling params consistent with training
    model = get_model(config.model).to(device)

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(missing)} unexpected={len(unexpected)}')
    model.eval()

    val_dataset = get_dataset(config.dataset.val)
    print(f'[val] {len(val_dataset)} samples')

    # Hook the receiver to capture the LAST recycle's pred_iptm (item index 5).
    captured = {}
    def hook(_module, _inputs, output):
        # output: (pred_seq, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae, pred_disorder)
        captured['iptm'] = output[5].detach()
        captured['plddt'] = output[4].detach()
    h = model.bfn.receiver.register_forward_hook(hook)

    loader = DataLoader(val_dataset, batch_size=4, collate_fn=PaddingCollate(),
                        shuffle=False, num_workers=0)

    all_pred_iptm, all_true_iptm = [], []
    all_pred_plddt, all_true_plddt = [], []
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = recursive_to(batch, device)
            if batch['aa'].shape[0] == 0:
                continue
            try:
                t = batch.get('fixed_t', None)
                if t is None:
                    batch['fixed_t'] = config.train.get('fixed_t', None)
                _ = model(batch)
            except Exception as e:
                print(f'  [skip batch] {e}')
                continue
            n_batches += 1
            pi = captured.get('iptm')
            pp = captured.get('plddt')
            if pi is None:
                continue
            # mask by residue mask — iptm is per-sequence scalar (N,)
            mask = batch.get('mask')
            # pred_iptm shape might be (N,) or (N,L); collapse to per-sample
            if pi.dim() > 1:
                pi = pi.view(pi.shape[0], -1).mean(-1)
                pp = pp.view(pp.shape[0], -1).mean(-1)
            else:
                pass
            all_pred_iptm.extend(pi.float().cpu().numpy().tolist())
            all_pred_plddt.extend(pp.float().cpu().numpy().tolist())
            if 'af2_iptm' in batch:
                all_true_iptm.extend(batch['af2_iptm'].float().cpu().numpy().reshape(-1).tolist())
            if 'af2_plddt' in batch:
                # per-sequence mean of true plddt
                tp = batch['af2_plddt']
                if tp.dim() > 1:
                    m2 = batch['mask'].bool() if mask is not None else None
                    if m2 is not None:
                        tp = (tp * m2).sum(-1) / (m2.sum(-1) + 1e-8)
                    else:
                        tp = tp.mean(-1)
                all_true_plddt.extend(tp.float().cpu().numpy().reshape(-1).tolist())

    h.remove()
    if not all_pred_iptm:
        print('NO predictions captured.')
        return

    pi = np.array(all_pred_iptm)
    print(f'\n=== PRED ipTM over {len(pi)} val samples, {n_batches} batches ===')
    print(f'  mean={pi.mean():.6f}  std={pi.std():.6f}  min={pi.min():.6f}  max={pi.max():.6f}')
    print(f'  range={pi.max()-pi.min():.6f}  rel_std={pi.std()/ (pi.mean()+1e-9):.4%}')
    print(f'  pct>0.55: {(pi>0.55).mean()*100:.1f}%   unique~: {len(np.unique(np.round(pi,4)))}')

    pp = np.array(all_pred_plddt)
    print(f'\n=== PRED pLDDT over {len(pp)} samples ===')
    print(f'  mean={pp.mean():.6f}  std={pp.std():.6f}  min={pp.min():.6f}  max={pp.max():.6f}')

    if all_true_iptm:
        ti = np.array(all_true_iptm[:len(pi)])
        print(f'\n=== TRUE ipTM ===')
        print(f'  mean={ti.mean():.6f}  std={ti.std():.6f}  min={ti.min():.6f}  max={ti.max():.6f}')
        # correlation between pred and true
        if len(ti) == len(pi) and pi.std() > 1e-9:
            c = np.corrcoef(pi, ti)[0,1]
            print(f'  corr(pred,true) = {c:.4f}')
        else:
            print(f'  corr: N/A (pred std≈0 → constant collapse)')
        print(f'  mean(pred)/mean(true) = {pi.mean()/(ti.mean()+1e-9):.2f}x  ← OC proxy')

    # per-sample dump (first 20)
    print('\n=== first 20 (pred_iptm, true_iptm) ===')
    for i in range(min(20, len(pi))):
        t = all_true_iptm[i] if i < len(all_true_iptm) else float('nan')
        print(f'  [{i:2d}] pred={pi[i]:.6f}  true={t:.6f}')

if __name__ == '__main__':
    main()
