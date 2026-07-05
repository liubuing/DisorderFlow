import os
#!/usr/bin/env python3
"""Probe V17i Pair Routing: direct model.sample() call, no bfn_loader."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import torch, yaml, numpy as np, random, lmdb, pickle, warnings
warnings.filterwarnings('ignore')
from easydict import EasyDict
from disorderflow.utils.misc import seed_all
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.models.bfn_model import AntibodyBFN
from probe_antigen_signal_usage import get_cdr_real

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/2200.pt'
cfg = EasyDict(yaml.safe_load(open('configs/train/bfn_v17i_pairrouting_xpu.yml', encoding='utf-8')))
model = AntibodyBFN(cfg.model)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False); model.eval()
print(f'V17i step {ckpt["iteration"]} pair_routing={model.bfn.receiver.pair_routing}')

# Use phase3 dataset to get properly-formatted entries
from disorderflow.datasets.phase3_dataset import Phase3Dataset
ds_cfg = EasyDict({'type': 'phase3', 'lmdb_path': 'data/sabdab_phase3_processed/train.lmdb'})
ds = Phase3Dataset(ds_cfg)
# Get transforms from V17i config
tx = get_transform(cfg.dataset.train.transform)

random.seed(5)
# Pick 3 entries with antigen
agg = {'complex': [], 'fixbb': []}; picked = 0
for idx in random.sample(range(len(ds)), min(len(ds), 50)):
    entry = ds[idx]
    if entry.get('antigen') is None or entry.get('heavy') is None: continue
    cdrs = get_cdr_real(entry, 'H')
    if not cdrs: continue
    lengths = [e_ - s for s, e_, _ in cdrs.values()]

    for include_ag in [True, False]:
        mode = 'complex' if include_ag else 'fixbb'
        seed_all(42)
        e2 = {'heavy': entry['heavy'], 'light': None, 'antigen': entry['antigen'] if include_ag else None}
        batch = recursive_to(PaddingCollate()([tx(e2)]), 'cpu')
        gen_mask = batch['generate_flag'][0].bool()
        if gen_mask.sum() == 0: continue

        with torch.no_grad():
            traj = model.sample(batch, sample_opt={'deterministic': False, 'num_recycles': 3})
        aa_pred = traj['pred_logits'][0, gen_mask.cpu()].argmax(dim=-1).tolist()
        design_seq = ''.join('ACDEFGHIKLMNPQRSTVWY'[a] for a in aa_pred)

        off = 0
        for nm in ['H1', 'H2', 'H3']:
            if nm not in cdrs: continue
            s, e_, real = cdrs[nm]; l = e_ - s
            dc = design_seq[off:off + l]
            if len(dc) == l: agg[mode].append(sum(a == b for a, b in zip(dc, real)) / l)
            off += l
    picked += 1; print(f'{picked}/3 OK')
    if picked >= 3: break

c = np.array(agg['complex']); f = np.array(agg['fixbb']); d = (c.mean() - f.mean()) * 100
print(f'\n=== V17i@{ckpt["iteration"]} ===')
print(f'complex={c.mean()*100:.1f}%  fixbb={f.mean()*100:.1f}%  Δ={d:+.2f}pp')
if d > 5: print('BREAKTHROUGH')
elif d > 1: print('MARGINAL')
elif d > 0: print('SLIGHT +')
else: print('NEGATIVE')
