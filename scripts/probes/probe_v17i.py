import os
#!/usr/bin/env python3
"""Probe for V17i Pair Routing checkpoints. Loads model from YAML config
(not checkpoint config) so pair_routing=True is properly applied."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import torch, yaml, numpy as np, random, lmdb, pickle, warnings
warnings.filterwarnings('ignore')
from easydict import EasyDict
from disorderflow.utils.misc import seed_all
from disorderflow.modules.bfn.core import AntibodyBFN_Core

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/2200.pt'
CFG = 'configs/train/bfn_v17i_pairrouting_xpu.yml'
N_COMPLEX, N_SAMPLES = 3, 3

cfg = EasyDict(yaml.safe_load(open(CFG, encoding='utf-8')))
m = cfg.model
model = AntibodyBFN_Core(m.res_feat_dim, m.pair_feat_dim, m.diffusion.num_steps,
    eps_net_opt=m.diffusion.eps_net_opt, loss_weight=m.loss_weight,
    beta=m.diffusion.beta, schedule=m.diffusion.schedule)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
print(f'Loaded {CKPT} step {ckpt.get("iteration","?")} pair_routing={model.receiver.pair_routing}')

# Set up bfn_loader
import bfn_loader
full_cfg = EasyDict(yaml.safe_load(open('app_config.yaml', encoding='utf-8')))
full_cfg.models.bfn.checkpoint = CKPT
full_cfg.sampling = EasyDict({'seed': 42, 'temperature': 0.1, 'num_steps': 20, 'stochastic': True})
bfn_loader._bfn_model = model
bfn_loader._bfn_config = full_cfg

from probe_antigen_signal_usage import write_pdb, get_cdr_real
from bfn_loader import run_bfn_design

env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
ids = pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb'))
random.seed(5); random.shuffle(ids)
tmpdir = 'oc_validation_results/_diag'; os.makedirs(tmpdir, exist_ok=True)
agg = {'complex': [], 'fixbb': []}; picked = 0

for sid in ids:
    if picked >= N_COMPLEX: break
    with env.begin() as txn: e = pickle.loads(txn.get(sid.encode()))
    if e.get('heavy') is None or e.get('antigen') is None: continue
    cdrs = get_cdr_real(e, 'H')
    if not cdrs: continue
    pdb_c = os.path.join(tmpdir, f'{sid}_c.pdb'); pdb_f = os.path.join(tmpdir, f'{sid}_f.pdb')
    try: write_pdb(e, pdb_c, True); write_pdb(e, pdb_f, False)
    except Exception as ex: print(f'{sid}: write failed {ex}'); continue
    parts = [f'{s+1}-{e_}' for s, e_, _ in cdrs.values()]
    region_spec = 'H:' + ','.join(parts)
    lengths = [e_ - s for s, e_, _ in cdrs.values()]
    seed_all(42)
    des_c = run_bfn_design(pdb_c, region_spec, N_SAMPLES, stochastic=True, context_chains=['P'], device='cpu', sort_by=None)
    seed_all(42)
    des_f = run_bfn_design(pdb_f, region_spec, N_SAMPLES, stochastic=True, context_chains=[], device='cpu', sort_by=None)
    for ci, nm in enumerate(['H1', 'H2', 'H3']):
        if nm not in cdrs: continue
        s, e_, real = cdrs[nm]; l = e_ - s; off = sum(lengths[:ci])
        rc = [sum(a == b for a, b in zip(d['sequence'][off:off+l], real)) / l for d in des_c if len(d['sequence'][off:off+l]) == l]
        rf = [sum(a == b for a, b in zip(d['sequence'][off:off+l], real)) / l for d in des_f if len(d['sequence'][off:off+l]) == l]
        if rc and rf: agg['complex'].append(np.mean(rc)); agg['fixbb'].append(np.mean(rf))
    picked += 1; print(f'{sid}: OK ({picked}/{N_COMPLEX})')
env.close()

c = np.array(agg['complex']); f = np.array(agg['fixbb']); d = (c.mean() - f.mean()) * 100
step = ckpt.get('iteration', '?')
print(f'\n=== V17i @{step} ===')
print(f'complex={c.mean()*100:.1f}%  fixbb={f.mean()*100:.1f}%  Δ={d:+.2f}pp')
if d > 5: print('BREAKTHROUGH')
elif d > 1: print('MARGINAL improvement')
elif d > 0: print('SLIGHT above zero')
else: print('NEGATIVE')
