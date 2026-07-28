import os
#!/usr/bin/env python3
"""Pre-compute per-residue disorder profiles for all SAbDab training antigens.

Creates a lookup dict: {sample_id: per_residue_disorder_array}
Used by the training data pipeline to inject epitope_disorder_profile.

This is the data bridge between P0-fix-A (profile computation) and P0-fix-B
(per-residue receiver integration).
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
import torch, lmdb, yaml

SABDAB_LMDB = 'data/sabdab_phase3_processed/train.lmdb'
SABDAB_IDS = 'data/sabdab_phase3_processed/train.lmdb-ids'
V18_CKPT = 'logs/bfn_v18_disorder_cond_xpu_2026_07_01__15_51_01_v18_way4/checkpoints/best.pt'
OUT_PATH = 'data/sabdab_disorder_lookup.pkl'
MAX_SAMPLES = None  # None = all

print("=" * 60)
print("Pre-computing per-residue disorder for training antigens")
print("=" * 60)

# Load model
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.modules.common.geometry import construct_3d_basis

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = V18_CKPT
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
print("Model loaded")


def predict_chain_disorder(model, chain_data, max_len=380):
    aa = chain_data['aa'][:max_len]; L = aa.shape[0]
    pos = chain_data['pos_heavyatom'][:max_len]
    mask_h = chain_data['mask_heavyatom'][:max_len]
    torsion = chain_data.get('torsion', torch.zeros(L, 4))
    if torsion.shape[0] > max_len: torsion = torsion[:max_len]
    mk_torsion = chain_data.get('mask_torsion', torch.ones(L, 4))
    if mk_torsion.shape[0] > max_len: mk_torsion = mk_torsion[:max_len]

    batch = {
        'aa': aa, 'pos_heavyatom': pos, 'mask_heavyatom': mask_h,
        'torsion': torsion, 'mask_torsion': mk_torsion,
        'generate_flag': torch.zeros(L, dtype=torch.bool),
        'fragment_type': torch.zeros(L, dtype=torch.long),
        'cdr_flag': torch.zeros(L, dtype=torch.long),
        'anchor_flag': torch.zeros(L, dtype=torch.bool),
        'chain_id': ['A'] * L, 'resseq': torch.arange(L),
        'res_nb': torch.arange(L), 'chain_nb': torch.zeros(L, dtype=torch.long),
    }
    batch = recursive_to(PaddingCollate()([batch]), 'cpu')
    batch['mask'] = torch.ones(1, batch['aa'].shape[1]).bool()

    with torch.no_grad():
        N, Lb = batch['aa'].shape; device = batch['aa'].device
        theta_seq = torch.zeros(N, Lb, 22, device=device)
        pm, ps = model.bfn.position_mean, model.bfn.position_scale
        theta_pos_norm = (batch['pos_heavyatom'][:, :, 1].float() - pm) / ps
        theta_ori = construct_3d_basis(
            batch['pos_heavyatom'][:, :, 1],
            batch['pos_heavyatom'][:, :, 2],
            batch['pos_heavyatom'][:, :, 0])
        theta_ang = batch.get('torsion', torch.zeros(N, Lb, 4, device=device))
        t_vec = 0.5 * torch.ones(N, device=device)
        pair_feat = torch.zeros(N, Lb, Lb, 128, device=device)
        out = model.bfn.receiver(
            theta_seq, theta_pos_norm, theta_ori, theta_ang,
            t_vec, pair_feat, batch['mask'].bool(),
            backbone_pos=batch['pos_heavyatom'][:, :, :4])
        pred = out[7]
        if pred is None: return None
        return torch.sigmoid(pred[0])[:L].cpu().numpy()


# Scan all training antigens
ids = pickle.load(open(SABDAB_IDS, 'rb'))
if MAX_SAMPLES:
    ids = ids[:MAX_SAMPLES]

env = lmdb.open(SABDAB_LMDB, subdir=False, readonly=True, lock=False, readahead=False)

lookup = {}
n_ok = 0; n_skip = 0; t0 = time.time()

for i, sid in enumerate(ids):
    with env.begin() as txn:
        e = pickle.loads(txn.get(sid.encode()))
    if e.get('antigen') is None:
        lookup[sid] = None  # no antigen
        n_skip += 1
        continue
    try:
        disorder = predict_chain_disorder(model, e['antigen'])
        if disorder is not None:
            lookup[sid] = disorder.astype(np.float32)
            n_ok += 1
        else:
            lookup[sid] = None
            n_skip += 1
    except Exception as ex:
        lookup[sid] = None
        n_skip += 1

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(ids) - i - 1) / rate
        print(f"  {i+1}/{len(ids)} ({n_ok} ok, {n_skip} skip) {elapsed:.0f}s, ETA {eta:.0f}s")

env.close()

# Summary statistics
ag_lens = [len(v) for v in lookup.values() if v is not None]
disorder_means = [v.mean() for v in lookup.values() if v is not None]
disorder_maxes = [v.max() for v in lookup.values() if v is not None]

print(f"\nResults: {n_ok} antigens scored, {n_skip} skipped")
if ag_lens:
    print(f"  Antigen lengths: {np.min(ag_lens)} - {np.max(ag_lens)}, mean={np.mean(ag_lens):.0f}")
    print(f"  Disorder means:  {np.mean(disorder_means):.4f} +/- {np.std(disorder_means):.4f}")
    print(f"  Disorder maxes:  {np.min(disorder_maxes):.4f} - {np.max(disorder_maxes):.4f}")
    n_flex = sum(1 for m in disorder_maxes if m > 0.4)
    print(f"  Antigens with flex residues (>0.4): {n_flex}/{n_ok}")

# Save
with open(OUT_PATH, 'wb') as f:
    pickle.dump(lookup, f)
print(f"\nSaved: {OUT_PATH} ({len(lookup)} entries, {os.path.getsize(OUT_PATH)/1024:.0f} KB)")
print("Done. Ready for training pipeline integration.")
