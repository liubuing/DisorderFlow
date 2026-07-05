#!/usr/bin/env python
"""B1 deep diagnostic: WHERE does the sequence pathway die?

Logs, per design, the per-stage values along the v12 ipTM sequence path:
    probs_seq[CDR]   — raw AA-probabilities at designed residues
    seq_emb[CDR]     — v12_seq_emb output (Linear 20->64+LN+ReLU) at CDR
    pooled_seq       — CDR-pooled seq_emb (input to v12_iptm)
    iptm_seq         — v12_iptm(pooled_seq) scalar

Reports cross-design std at each stage. If probs_seq varies but seq_emb
collapses → v12_seq_emb is the dead point. If seq_emb varies but iptm_seq
collapses → v12_iptm is dead. This decides whether to rebuild the embed, the
head, or both.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch, numpy as np
import yaml

CKPT = os.environ.get('V14_CKPT',
    'logs/bfn_v14_grouped_conf_xpu_2026_06_26__02_53_57/checkpoints/best.pt')

cfg_path = 'app_config.yaml'
app = yaml.safe_load(open(cfg_path, encoding='utf-8'))
orig = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = CKPT
yaml.dump(app, open(cfg_path, 'w', encoding='utf-8'), default_flow_style=False)

stages = []  # per design: dict of arrays
try:
    import bfn_loader
    bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
    from bfn_loader import run_bfn_design, load_bfn
    from disorderflow.utils.misc import seed_all
    seed_all(42)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_bfn(dev)
    recv = model.bfn.receiver
    recv._dbg_iptm_components = []  # enable stash
    orig_fwd = recv.forward

    def fwd(theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res,
            backbone_pos=None, prev_conf=None, prev_iptm=None, prev_pae=None,
            prev_seq_emb=None, mask_gen=None):
        # Capture pre-head stages for this forward.
        probs_seq = torch.softmax(theta_seq, dim=-1)  # (N,L,20)
        seq_emb = recv.v12_seq_emb(probs_seq)          # (N,L,64)
        # CDR region
        gm = mask_gen if mask_gen is not None else mask_res
        gm_f = gm.float()
        all_sum = gm_f.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (seq_emb * gm_f.unsqueeze(-1)).sum(dim=1) / all_sum  # (N,64)
        iptm_seq_raw = recv.v12_iptm(pooled)           # (N,1)
        # stash per-sample (N=1 in design sampling)
        stages.append({
            'probs_cdr_mean_var': float(probs_seq[0][gm[0]].var().item()),
            'seqemb_cdr_std': float(seq_emb[0][gm[0]].std().item()),
            'pooled_std': float(pooled[0].std().item()),
            'pooled_norm': float(pooled[0].norm().item()),
            'iptm_seq': float(iptm_seq_raw[0].item()),
        })
        return orig_fwd(theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res,
                        backbone_pos=backbone_pos, prev_conf=prev_conf, prev_iptm=prev_iptm,
                        prev_pae=prev_pae, prev_seq_emb=prev_seq_emb, mask_gen=mask_gen)
    recv.forward = fwd

    print(f'[diag] ckpt={CKPT}')
    designs = run_bfn_design('data/misfolding_targets/5IMK.pdb',
        'B:26-33,51-58,97-113', num_samples=8, stochastic=True,
        context_chains=None, device=dev)

    print(f'\n=== {len(designs)} designs, sequence-pathway per-stage values ===')
    print(f'{"i":>3} {"PPL":>5} {"probs_var":>10} {"seqemb_std":>11} {"pooled_std":>11} {"pooled_norm":>11} {"iptm_seq":>10}')
    for i,(d,s) in enumerate(zip(designs, stages)):
        print(f'{i+1:>3} {d.get("ppl",0):>5.0f} {s["probs_cdr_mean_var"]:>10.6f} '
              f'{s["seqemb_cdr_std"]:>11.6f} {s["pooled_std"]:>11.6f} '
              f'{s["pooled_norm"]:>11.4f} {s["iptm_seq"]:>10.5f}')

    def std(k): return float(np.std([s[k] for s in stages]))
    print('\n=== cross-design std per stage ===')
    for k in ['probs_cdr_mean_var','seqemb_cdr_std','pooled_std','pooled_norm','iptm_seq']:
        print(f'  {k:>20}: std={std(k):.6f}  range={max(s[k] for s in stages)-min(s[k] for s in stages):.6f}')

    print('\n=== dead-point verdict ===')
    if std('probs_cdr_mean_var') < 1e-5:
        print('  >> probs_seq identical across designs (upstream: sampling not varying CDR?)')
    elif std('seqemb_cdr_std') < 1e-5 and std('pooled_norm') < 1e-3:
        print('  >> v12_seq_emb is DEAD: distinct probs → identical embedding. Rebuild emb.')
    elif std('iptm_seq') < 1e-4:
        print('  >> v12_iptm is DEAD: pooled_seq varies but output constant. Rebuild head.')
    else:
        print('  >> sequence pathway actually varies end-to-end — check sigmoid/scale downstream')
finally:
    app['models']['bfn']['checkpoint'] = orig
    yaml.dump(app, open(cfg_path, 'w', encoding='utf-8'), default_flow_style=False)
