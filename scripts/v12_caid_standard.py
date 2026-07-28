#!/usr/bin/env python3
"""V12 CAID Standard Benchmark — multi-IDP balanced evaluation.

Fixes the V8 "红斑": AUC=1.0 from single Aβ42 (42 residues) vs 15327 folded.
Uses balanced set of IDPs from SAbDab disorder lookup + folded SAbDab complexes.
Reports AUC-ROC/PR with proper class balance.
"""
import sys, os, pickle, time, json, random
sys.path.insert(0, '.'); sys.path.insert(0, 'modules')
import numpy as np
import torch, yaml, lmdb
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

V20_CKPT = 'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt'
V15_CKPT = 'logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt'
N_SABDAB = 80  # folded complexes
N_IDP_RESIDUES_TARGET = 5000  # target disordered residues for balance

print("=" * 60)
print("V12 CAID Standard Benchmark — Multi-IDP Balanced")
print("=" * 60)

# ── Load V20 model ──
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig_ckpt = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = V20_CKPT
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, cfg = bfn_loader.load_bfn('cpu')
print(f"V20 model loaded")
app['models']['bfn']['checkpoint'] = orig_ckpt
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))

# ── Predict disorder ──
def predict_disorder(model, batch):
    from disorderflow.modules.common.geometry import construct_3d_basis
    with torch.no_grad():
        N, L = batch['aa'].shape
        device = batch['aa'].device
        theta_seq = torch.zeros(N, L, 22, device=device)
        theta_pos = batch['pos_heavyatom'][:, :, 1].float()
        pos_mean = model.bfn.position_mean
        pos_scale = model.bfn.position_scale
        theta_pos_norm = (theta_pos - pos_mean) / pos_scale
        theta_ori = construct_3d_basis(
            batch['pos_heavyatom'][:,:,1], batch['pos_heavyatom'][:,:,2], batch['pos_heavyatom'][:,:,0])
        theta_ang = batch.get('torsion', torch.zeros(N, L, 4, device=device))
        t = 0.5 * torch.ones(N, device=device)
        pair_feat = batch.get('pair_feat', torch.zeros(N, L, L, 128, device=device))
        mask_res = batch['mask'].bool()
        backbone_pos = batch['pos_heavyatom'][:, :, :4]
        mask_gen = batch.get('generate_flag', torch.zeros(N, L).bool())
        out = model.bfn.receiver(theta_seq, theta_pos_norm, theta_ori, theta_ang, t,
                                 pair_feat, mask_res, backbone_pos=backbone_pos, mask_gen=mask_gen)
        pred_disorder = out[7]
        if pred_disorder is None:
            return np.zeros(L)
        disorder = torch.sigmoid(pred_disorder)
        mask = mask_res[0].cpu().numpy()
        return disorder[0].cpu().numpy()[mask]

# ── ORDERED: SAbDab complexes ──
print(f"\n[1/3] Running on {N_SABDAB} SAbDab complexes (ORDERED)...")
transform = get_transform([{'type': 'merge_chains'}, {'type': 'patch_around_anchor'}])
env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
ids = pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb'))
random.seed(42); random.shuffle(ids)

ordered_preds, ordered_labels = [], []
n_done = 0
with env.begin() as txn:
    for sid in ids:
        if n_done >= N_SABDAB: break
        e = pickle.loads(txn.get(sid.encode()))
        if e.get('heavy') is None: continue
        try:
            batch_data = transform(e)
            batch = recursive_to(PaddingCollate()([batch_data]), 'cpu')
            preds = predict_disorder(model, batch)
            ordered_preds.extend(preds.tolist())
            ordered_labels.extend([0] * len(preds))
            n_done += 1
        except Exception as ex:
            continue
env.close()
print(f"  {n_done} complexes, {len(ordered_preds)} ordered residues")

# ── DISORDERED: Multi-IDP from disorder lookup ──
print(f"\n[2/3] Running on multi-IDP epitopes (DISORDERED)...")
lookup = pickle.load(open('data/sabdab_disorder_lookup.pkl', 'rb'))
# Get high-disorder epitopes (multiple IDPs, max disorder > 0.4)
high_disorder = [(k, v) for k, v in lookup.items() if v.max() > 0.4]
random.seed(42)
random.shuffle(high_disorder)

disorder_preds, disorder_labels_arr = [], []
n_idp = 0
transform2 = get_transform([{'type': 'mask_multiple_cdrs'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}])
env2 = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
valid_ids = set(pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb')))

with env2.begin() as txn:
    for sid, disorder_arr in high_disorder:
        if sid not in valid_ids: continue
        if len(disorder_preds) >= N_IDP_RESIDUES_TARGET: break
        try:
            e = pickle.loads(txn.get(sid.encode()))
            if e.get('heavy') is None or e.get('antigen') is None: continue
            batch_data = transform2(e)
            batch = recursive_to(PaddingCollate()([batch_data]), 'cpu')
            preds = predict_disorder(model, batch)
            # Get antigen residues only
            ag_len = len(e['antigen']['aa'])
            ag_preds = preds[-ag_len:]
            ag_labels = disorder_arr[:ag_len]  # continuous → binarize
            disorder_preds.extend(ag_preds.tolist())
            disorder_labels_arr.extend(ag_labels.tolist())
            n_idp += 1
        except Exception as ex:
            continue
env2.close()
print(f"  {n_idp} antigens, {len(disorder_preds)} disordered residues")

# ── Compute metrics ──
print(f"\n[3/3] Computing AUC...")
print(f"  DEBUG: ordered_preds={len(ordered_preds)}, ordered_labels={len(ordered_labels)}")
print(f"  DEBUG: disorder_preds={len(disorder_preds)}, disorder_labels={len(disorder_labels_arr)}")
all_preds = np.concatenate([np.array(ordered_preds), np.array(disorder_preds)])
all_labels_cont = np.concatenate([np.array(ordered_labels), np.array(disorder_labels_arr)])
# Ensure consistent length
min_len = min(len(all_preds), len(all_labels_cont))
all_preds = all_preds[:min_len]
all_labels_cont = all_labels_cont[:min_len]

# Binarize: disorder > 0.3 → label 1 (disordered)
binary_labels = (all_labels_cont > 0.3).astype(int)

n_pos = binary_labels.sum()
n_neg = len(binary_labels) - n_pos

roc = roc_auc_score(binary_labels, all_preds)
pr = average_precision_score(binary_labels, all_preds)

print(f"\n{'='*60}")
print(f"V12 CAID STANDARD BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"Ordered:   {n_neg} residues ({n_done} SAbDab complexes)")
print(f"Disordered: {n_pos} residues ({n_idp} antigens)")
print(f"Class balance: {n_pos/(n_pos+n_neg)*100:.1f}% disordered")
print(f"\nBFN V20 Disorder Head:")
print(f"  AUC-ROC: {roc:.4f}")
print(f"  AUC-PR:  {pr:.4f}")
print(f"  Ordered mean disorder:  {np.mean([p for p,l in zip(all_preds, binary_labels) if l==0]):.4f}")
print(f"  Disordered mean disorder: {np.mean([p for p,l in zip(all_preds, binary_labels) if l==1]):.4f}")
sep = np.mean([p for p,l in zip(all_preds, binary_labels) if l==1]) - np.mean([p for p,l in zip(all_preds, binary_labels) if l==0])
print(f"  Separation Δ: {sep:+.4f}")
print(f"\nBaseline (random): AUC-ROC=0.500, AUC-PR={n_pos/(n_pos+n_neg):.4f}")

# Save
os.makedirs('idp_design_results', exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
out = f'idp_design_results/caid_standard_v12_{ts}.json'
json.dump({
    'benchmark': 'V12 CAID Standard — Multi-IDP Balanced',
    'model': 'V20 (head_seq unfrozen)',
    'n_ordered_residues': int(n_neg),
    'n_disordered_residues': int(n_pos),
    'n_ordered_complexes': n_done,
    'n_disordered_antigens': n_idp,
    'class_balance': float(n_pos/(n_pos+n_neg)),
    'auc_roc': float(roc),
    'auc_pr': float(pr),
    'ordered_mean': float(np.mean([p for p,l in zip(all_preds, binary_labels) if l==0])),
    'disordered_mean': float(np.mean([p for p,l in zip(all_preds, binary_labels) if l==1])),
    'separation': float(sep),
}, open(out, 'w'), indent=2)
print(f"\nSaved: {out}")
