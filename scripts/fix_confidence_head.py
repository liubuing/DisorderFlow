#!/usr/bin/env python
"""修头: Fine-tune BFN confidence heads with anti-collapse losses.

Uses logs/disorder_head_retrain.pt (40.6MB, full encoder + heads).
Trains on data/confidence_design_variants_v14/ (704 train, 128 val, real AF2 labels).

Strategy:
  Phase 1 (diagnosis): Measure encoder feature variance across designs.
  Phase 2 (head-only): Freeze encoder, retrain heads with correlation + variance loss.
  Phase 3 (encoder unfreeze): Unfreeze last 2 encoder layers, low LR joint training.

This does NOT invalidate previous calibration work:
  - Isotonic calibration remains as post-hoc safety net
  - Sequence calibrator becomes ensemble member
  - CalibratedScorer integrates the fixed head + calibration layers

Usage:
  python scripts/fix_confidence_head.py --phase diagnose
  python scripts/fix_confidence_head.py --phase head-only --epochs 200
  python scripts/fix_confidence_head.py --phase unfreeze --epochs 100
  python scripts/fix_confidence_head.py --phase all  # run all sequentially
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_PATH = PROJECT / "logs" / "disorder_head_retrain.pt"
TRAIN_LMDB = PROJECT / "data" / "confidence_design_variants_v14" / "train_grouped.lmdb"
VAL_LMDB = PROJECT / "data" / "confidence_design_variants_v14" / "val_grouped.lmdb"
OUT_DIR = PROJECT / "calibration_artifacts"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── Losses ──────────────────────────────────────────────────────────────────

class PearsonCorrelationLoss(nn.Module):
    def forward(self, pred, target):
        if pred.numel() < 3:
            return torch.tensor(0.0, device=pred.device)
        pc = pred - pred.mean()
        tc = target - target.mean()
        corr = (pc * tc).sum() / (pc.pow(2).sum().sqrt().clamp(1e-8) *
                                   tc.pow(2).sum().sqrt().clamp(1e-8))
        return -corr


class VarianceAntiCollapse(nn.Module):
    """Hinge loss: penalize pred_std < fraction of target_std."""
    def __init__(self, min_ratio=0.3):
        super().__init__()
        self.min_ratio = min_ratio

    def forward(self, pred, target):
        if pred.numel() < 4:
            return torch.tensor(0.0, device=pred.device)
        desired = target.std() * self.min_ratio
        return F.relu(desired - pred.std())


class GroupedMarginRanking(nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target, groups):
        loss = torch.tensor(0.0, device=pred.device)
        n = 0
        for gid in groups.unique():
            mask = groups == gid
            if mask.sum() < 2:
                continue
            p, t = pred[mask], target[mask]
            # Vectorized pairwise ranking
            diff_p = p.unsqueeze(0) - p.unsqueeze(1)  # (n, n)
            diff_t = t.unsqueeze(0) - t.unsqueeze(1)
            # Where t[i] > t[j], we want p[i] > p[j]
            sign = diff_t.sign()
            violations = F.relu(self.margin - sign * diff_p)
            # Only count pairs where t differs
            valid = (diff_t.abs() > 1e-6).float()
            loss = loss + (violations * valid).sum()
            n += valid.sum().item()
        return loss / max(n, 1)


# ─── Data ────────────────────────────────────────────────────────────────────

def load_lmdb_batches(lmdb_path, max_entries=0, max_residues=300):
    """Load LMDB entries as a list of (batch_dict, af2_plddt_mean, af2_iptm, scaffold_id)."""
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        n = pickle.loads(txn.get(b'__len__'))
    if max_entries > 0:
        n = min(n, max_entries)

    entries = []
    with env.begin() as txn:
        for i in range(n):
            entry = pickle.loads(txn.get(f'{i:08d}'.encode()))
            seq_len = len(entry['sequence'])
            if seq_len > max_residues:
                continue
            entries.append(entry)
    env.close()
    print(f"  Loaded {len(entries)} entries from {Path(lmdb_path).name} "
          f"(filtered >{max_residues} res)")
    return entries


def forward_with_grad(model, batch):
    """Replicate score_fixed logic WITHOUT @torch.no_grad, so heads get gradients."""
    bfn = model.bfn
    x_seq = batch['aa']
    N, L = x_seq.shape
    mask_res = batch['mask'].bool()
    mask_gen = batch['generate_flag'].bool() & mask_res
    if 'mask_antigen' not in batch:
        batch['mask_antigen'] = bfn._mask_antigen(batch, mask_gen)

    inp_seq = F.one_hot(
        x_seq.clamp(min=0, max=bfn.num_classes - 1),
        num_classes=bfn.num_classes,
    ).to(dtype=batch['pair_feat'].dtype)
    from disorderflow.modules.common.geometry import construct_3d_basis
    inp_pos = bfn._normalize_position(batch['pos_heavyatom'][:, :, 1])
    inp_ori = construct_3d_basis(
        batch['pos_heavyatom'][:, :, 1],
        batch['pos_heavyatom'][:, :, 2],
        batch['pos_heavyatom'][:, :, 0],
    )
    inp_ang = batch['torsion']
    t = torch.full((N,), 0.5, device=x_seq.device, dtype=inp_pos.dtype)

    result = bfn.receiver(
        inp_seq, inp_pos, inp_ori, inp_ang, t, batch['pair_feat'], mask_res,
        backbone_pos=batch['pos_heavyatom'][:, :, :4],
        mask_gen=mask_gen,
        mask_antigen=batch.get('mask_antigen'),
        epitope_disorder=batch.get('epitope_disorder_profile',
                                   batch.get('epitope_disorder', None)),
    )
    (_, _, _, _, pred_plddt, pred_iptm, pred_pae, pred_disorder,
     pred_contact, pred_contrastive) = result

    residue_mask = mask_res.to(pred_plddt.dtype)
    sample_mask = mask_res.any(dim=1).to(pred_iptm.dtype)
    return {
        'plddt': pred_plddt * residue_mask,
        'iptm': pred_iptm * sample_mask,
    }


def entry_to_model_input(entry, device):
    """Convert LMDB entry to model-ready batch tensors."""
    batch = entry['batch']
    # Add batch dimension
    model_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            model_batch[k] = v.unsqueeze(0).to(device)
        else:
            model_batch[k] = v
    # Ensure mask exists
    if 'mask' not in model_batch:
        model_batch['mask'] = model_batch['mask_heavyatom'][:, :, 1].bool()
    # generate_flag = all False (full context for encoder)
    model_batch['generate_flag'] = torch.zeros_like(model_batch['aa'], dtype=torch.bool)
    # pair_feat placeholder if missing
    if 'pair_feat' not in model_batch:
        L = model_batch['aa'].shape[1]
        model_batch['pair_feat'] = torch.zeros(1, L, L, 128, device=device)

    # Targets
    af2_plddt = entry['af2_plddt']
    af2_plddt_mean = af2_plddt.mean().item() if af2_plddt.dim() > 0 else af2_plddt.item()
    af2_iptm = entry['af2_iptm']
    af2_iptm_val = af2_iptm.item() if isinstance(af2_iptm, torch.Tensor) else float(af2_iptm)
    scaffold_id = int(entry.get('scaffold_id', 0))

    return model_batch, af2_plddt_mean, af2_iptm_val, scaffold_id


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(device=DEVICE):
    """Load BFN model from disorder_head_retrain.pt."""
    from disorderflow.models import get_model

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)

    model = get_model(mc).to(device)
    # Load with strict=False (some keys may mismatch)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    model.eval()
    return model, ckpt['config']


def get_encoder_features(model, batch):
    """Extract 256-dim encoder features via forward hook."""
    features = {}
    def hook(module, input, output):
        # receiver returns tuple; we need the internal features
        # Actually, let's hook the encoder directly
        pass

    # Run through the receiver's encoder path
    # The receiver.forward() calls self.encoder() internally
    # We hook the encoder output
    encoder = model.bfn.receiver.encoder if hasattr(model.bfn.receiver, 'encoder') else None
    if encoder is None:
        # Try alternative path
        encoder = model.bfn.receiver

    captured = {}
    def capture_hook(module, inp, out):
        if isinstance(out, tuple):
            captured['features'] = out[0].detach() if out[0].dim() == 3 else None
        elif isinstance(out, torch.Tensor) and out.dim() == 3:
            captured['features'] = out.detach()

    # Find the encoder module
    enc_module = None
    for name, mod in model.bfn.receiver.named_modules():
        if 'encoder' in name and name.count('.') <= 1:
            enc_module = mod
            break
    if enc_module is None:
        # Fallback: hook the receiver itself and extract from output
        enc_module = model.bfn.receiver

    h = enc_module.register_forward_hook(capture_hook)
    return captured, h


# ─── Phase 1: Diagnosis ─────────────────────────────────────────────────────

def diagnose(model, val_entries):
    """Measure encoder feature variance across designs."""
    print("\n" + "=" * 70)
    print("PHASE 1: DIAGNOSIS — Encoder feature variance")
    print("=" * 70)

    model.eval()
    all_features = []
    all_plddt = []
    all_iptm = []
    all_scaffolds = []
    all_pred_plddt = []
    all_pred_iptm = []

    with torch.no_grad():
        for i, entry in enumerate(val_entries[:64]):  # Limit for speed
            batch, af2_p, af2_i, sid = entry_to_model_input(entry, DEVICE)
            try:
                result = model.bfn.score_fixed(batch, fixed_t=0.5)
                pred_p = result['plddt'][0].mean().item()
                pred_i = result['iptm'][0].item()
                all_pred_plddt.append(pred_p)
                all_pred_iptm.append(pred_i)
                all_plddt.append(af2_p)
                all_iptm.append(af2_i)
                all_scaffolds.append(sid)
            except Exception as e:
                if i < 3:
                    print(f"  [WARN] Entry {i} failed: {e}")
                continue

            if (i + 1) % 16 == 0:
                print(f"  Processed {i+1}/{min(len(val_entries), 64)}")

    pred_p = np.array(all_pred_plddt)
    pred_i = np.array(all_pred_iptm)
    true_p = np.array(all_plddt)
    true_i = np.array(all_iptm)
    scaffolds = np.array(all_scaffolds)

    print(f"\n  N = {len(pred_p)} designs processed")
    print(f"\n  === BFN head outputs ===")
    print(f"  pred_pLDDT: mean={pred_p.mean():.4f} std={pred_p.std():.6f} "
          f"range=[{pred_p.min():.4f}, {pred_p.max():.4f}]")
    print(f"  pred_ipTM:  mean={pred_i.mean():.4f} std={pred_i.std():.6f} "
          f"range=[{pred_i.min():.4f}, {pred_i.max():.4f}]")
    print(f"\n  === AF2 ground truth ===")
    print(f"  af2_pLDDT:  mean={true_p.mean():.4f} std={true_p.std():.4f} "
          f"range=[{true_p.min():.4f}, {true_p.max():.4f}]")
    print(f"  af2_ipTM:   mean={true_i.mean():.4f} std={true_i.std():.4f} "
          f"range=[{true_i.min():.4f}, {true_i.max():.4f}]")

    # Within-scaffold analysis
    print(f"\n  === Within-scaffold variance (collapse detection) ===")
    unique_scaffolds = np.unique(scaffolds)
    within_stds_p = []
    within_stds_i = []
    for sid in unique_scaffolds:
        mask = scaffolds == sid
        if mask.sum() >= 3:
            within_stds_p.append(pred_p[mask].std())
            within_stds_i.append(pred_i[mask].std())
    if within_stds_p:
        print(f"  Scaffolds with >=3 designs: {len(within_stds_p)}")
        print(f"  Within-scaffold pred_pLDDT std: mean={np.mean(within_stds_p):.6f} "
              f"max={np.max(within_stds_p):.6f}")
        print(f"  Within-scaffold pred_ipTM std:  mean={np.mean(within_stds_i):.6f} "
              f"max={np.max(within_stds_i):.6f}")
        collapsed = np.mean(within_stds_p) < 0.001
        print(f"\n  VERDICT: {'COLLAPSED' if collapsed else 'HAS VARIANCE'} "
              f"(threshold: std < 0.001)")
    else:
        print(f"  No scaffolds with >=3 designs found (all unique scaffolds)")
        print(f"  Cross-scaffold std is the only signal available.")

    # Correlation
    from scipy.stats import spearmanr
    sp_p = spearmanr(pred_p, true_p)[0]
    sp_i = spearmanr(pred_i, true_i)[0]
    print(f"\n  === Correlation (current head) ===")
    print(f"  Spearman(pred_pLDDT, af2_pLDDT) = {sp_p:.4f}")
    print(f"  Spearman(pred_ipTM, af2_ipTM)   = {sp_i:.4f}")

    return {
        'pred_plddt_std': float(pred_p.std()),
        'pred_iptm_std': float(pred_i.std()),
        'true_plddt_std': float(true_p.std()),
        'true_iptm_std': float(true_i.std()),
        'spearman_plddt': float(sp_p) if not np.isnan(sp_p) else 0,
        'spearman_iptm': float(sp_i) if not np.isnan(sp_i) else 0,
        'n_designs': len(pred_p),
    }


# ─── Phase 2: Head-only fine-tune ───────────────────────────────────────────

def train_head_only(model, train_entries, val_entries, epochs=200, lr=1e-3):
    """Freeze encoder, retrain confidence heads with anti-collapse losses."""
    print("\n" + "=" * 70)
    print("PHASE 2: HEAD-ONLY FINE-TUNE (encoder frozen)")
    print("=" * 70)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze confidence heads
    head_suffixes = ['head_plddt', 'head_iptm', 'head_pae', 'conf_embed',
                     'v12_plddt', 'v12_iptm', 'v12_pae', 'v12_seq_emb',
                     'v14_iptm_bb', 'head_seq']
    trainable_params = []
    for name, param in model.named_parameters():
        if any(s in name for s in head_suffixes):
            param.requires_grad = True
            trainable_params.append(param)

    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_train:,} / {n_total:,} params ({100*n_train/n_total:.1f}%)")

    # Re-initialize head_plddt and head_iptm to break learned collapse
    print("  Re-initializing head_plddt and head_iptm (break collapse)...")
    for name, module in model.named_modules():
        if name.endswith('head_plddt') or name.endswith('head_iptm') or \
           name.endswith('v14_iptm_bb'):
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
                else:
                    nn.init.zeros_(p)

    # Losses
    l1_loss = nn.SmoothL1Loss(beta=0.05)
    corr_loss = PearsonCorrelationLoss()
    var_loss = VarianceAntiCollapse(min_ratio=0.3)
    rank_loss = GroupedMarginRanking(margin=0.01)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2)

    # Loss weights
    W_L1 = 1.0
    W_CORR = 0.8
    W_VAR = 0.5
    W_RANK = 0.3

    best_val_metric = -float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        # Shuffle training entries
        indices = np.random.permutation(len(train_entries))
        epoch_losses = []
        batch_pred_p, batch_pred_i = [], []
        batch_true_p, batch_true_i = [], []
        batch_sids = []

        # Mini-batch of 4 (process one at a time due to variable length)
        accum = 0
        optimizer.zero_grad()

        for idx in indices:
            entry = train_entries[idx]
            batch, af2_p, af2_i, sid = entry_to_model_input(entry, DEVICE)
            try:
                result = forward_with_grad(model, batch)
            except Exception:
                continue

            pred_p = result['plddt'][0].mean()
            pred_i = result['iptm'][0]
            target_p = torch.tensor(af2_p, device=DEVICE)
            target_i = torch.tensor(af2_i, device=DEVICE)

            batch_pred_p.append(pred_p)
            batch_pred_i.append(pred_i)
            batch_true_p.append(target_p)
            batch_true_i.append(target_i)
            batch_sids.append(sid)
            accum += 1

            if accum >= 4:
                # Compute batch losses
                pp = torch.stack(batch_pred_p)
                pi = torch.stack(batch_pred_i)
                tp = torch.stack(batch_true_p)
                ti = torch.stack(batch_true_i)
                sids = torch.tensor(batch_sids, device=DEVICE)

                loss_l1 = l1_loss(pp, tp) + l1_loss(pi, ti)
                loss_corr = corr_loss(pp, tp) + corr_loss(pi, ti)
                loss_var = var_loss(pp, tp) + var_loss(pi, ti)
                loss_rank = rank_loss(pi, ti, sids) + rank_loss(pp, tp, sids)

                total = W_L1 * loss_l1 + W_CORR * loss_corr + \
                        W_VAR * loss_var + W_RANK * loss_rank
                total = total / 4  # normalize by accum
                total.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_losses.append(total.item() * 4)
                batch_pred_p, batch_pred_i = [], []
                batch_true_p, batch_true_i = [], []
                batch_sids = []
                accum = 0

        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            vp, vi, vtp, vti = [], [], [], []
            with torch.no_grad():
                for entry in val_entries[:64]:
                    batch, af2_p, af2_i, sid = entry_to_model_input(entry, DEVICE)
                    try:
                        result = model.bfn.score_fixed(batch, fixed_t=0.5)
                        vp.append(result['plddt'][0].mean().item())
                        vi.append(result['iptm'][0].item())
                        vtp.append(af2_p)
                        vti.append(af2_i)
                    except Exception:
                        continue

            vp, vi = np.array(vp), np.array(vi)
            vtp, vti = np.array(vtp), np.array(vti)
            from scipy.stats import spearmanr
            sp_p = spearmanr(vp, vtp)[0] if len(vp) > 2 else 0
            sp_i = spearmanr(vi, vti)[0] if len(vi) > 2 else 0
            sp_p = 0 if np.isnan(sp_p) else sp_p
            sp_i = 0 if np.isnan(sp_i) else sp_i

            metric = sp_p + sp_i + vp.std() * 5 + vi.std() * 5  # composite
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0

            print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                  f"val: sp_p={sp_p:.3f} sp_i={sp_i:.3f} "
                  f"std_p={vp.std():.4f} std_i={vi.std():.4f}")

            history.append({
                'epoch': epoch + 1, 'loss': float(avg_loss),
                'val_spearman_plddt': float(sp_p), 'val_spearman_iptm': float(sp_i),
                'val_std_plddt': float(vp.std()), 'val_std_iptm': float(vi.std()),
            })

            if metric > best_val_metric:
                best_val_metric = metric
                save_path = OUT_DIR / "fixed_heads_best.pt"
                head_state = {k: v.cpu() for k, v in model.state_dict().items()
                              if any(s in k for s in head_suffixes)}
                torch.save({
                    'head_state': head_state,
                    'epoch': epoch + 1,
                    'metrics': history[-1],
                }, save_path)

    print(f"\n  Best composite metric: {best_val_metric:.4f}")
    print(f"  Saved → {OUT_DIR / 'fixed_heads_best.pt'}")
    return history


# ─── Phase 3: Encoder unfreeze ──────────────────────────────────────────────

def train_unfreeze(model, train_entries, val_entries, epochs=100, lr=2e-5):
    """Unfreeze last 2 encoder layers + heads, very low LR."""
    print("\n" + "=" * 70)
    print("PHASE 3: ENCODER UNFREEZE (last 2 layers, lr=2e-5)")
    print("=" * 70)

    # Load best heads from Phase 2
    best_path = OUT_DIR / "fixed_heads_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['head_state'], strict=False)
        print(f"  Loaded Phase 2 best heads (epoch {ckpt['epoch']})")

    # Freeze all, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze heads
    head_suffixes = ['head_plddt', 'head_iptm', 'head_pae', 'conf_embed',
                     'v12_plddt', 'v12_iptm', 'v12_pae', 'v12_seq_emb',
                     'v14_iptm_bb', 'head_seq']
    for name, param in model.named_parameters():
        if any(s in name for s in head_suffixes):
            param.requires_grad = True

    # Unfreeze last 2 encoder layers
    encoder_layers = []
    for name, module in model.named_modules():
        if 'encoder' in name and 'blocks' in name:
            # e.g. bfn.receiver.encoder.blocks.5
            parts = name.split('.')
            if len(parts) >= 5 and parts[-2] == 'blocks':
                try:
                    layer_idx = int(parts[-1])
                    encoder_layers.append((layer_idx, name, module))
                except ValueError:
                    pass

    if encoder_layers:
        encoder_layers.sort(key=lambda x: x[0])
        n_layers = max(l[0] for l in encoder_layers) + 1
        unfreeze_from = n_layers - 2
        for idx, name, module in encoder_layers:
            if idx >= unfreeze_from:
                for param in module.parameters():
                    param.requires_grad = True
        print(f"  Unfroze encoder layers {unfreeze_from}-{n_layers-1}")
    else:
        print("  [WARN] Could not identify encoder layers, training heads only")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,}")

    # Same losses, lower LR
    l1_loss = nn.SmoothL1Loss(beta=0.05)
    corr_loss = PearsonCorrelationLoss()
    var_loss = VarianceAntiCollapse(min_ratio=0.3)
    rank_loss = GroupedMarginRanking(margin=0.01)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_metric = -float('inf')

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_entries))
        epoch_losses = []
        bp, bi, btp, bti, bsids = [], [], [], [], []
        accum = 0
        optimizer.zero_grad()

        for idx in indices:
            entry = train_entries[idx]
            batch, af2_p, af2_i, sid = entry_to_model_input(entry, DEVICE)
            try:
                result = forward_with_grad(model, batch)
            except Exception:
                continue

            bp.append(result['plddt'][0].mean())
            bi.append(result['iptm'][0])
            btp.append(torch.tensor(af2_p, device=DEVICE))
            bti.append(torch.tensor(af2_i, device=DEVICE))
            bsids.append(sid)
            accum += 1

            if accum >= 4:
                pp, pi = torch.stack(bp), torch.stack(bi)
                tp, ti = torch.stack(btp), torch.stack(bti)
                sids = torch.tensor(bsids, device=DEVICE)

                loss = (l1_loss(pp, tp) + l1_loss(pi, ti)
                        + 0.8 * (corr_loss(pp, tp) + corr_loss(pi, ti))
                        + 0.5 * (var_loss(pp, tp) + var_loss(pi, ti))
                        + 0.3 * (rank_loss(pi, ti, sids) + rank_loss(pp, tp, sids)))
                (loss / 4).backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                optimizer.step()
                optimizer.zero_grad()
                epoch_losses.append(loss.item())
                bp, bi, btp, bti, bsids = [], [], [], [], []
                accum = 0

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            vp, vi, vtp, vti = [], [], [], []
            with torch.no_grad():
                for entry in val_entries[:64]:
                    batch, af2_p, af2_i, _ = entry_to_model_input(entry, DEVICE)
                    try:
                        result = model.bfn.score_fixed(batch, fixed_t=0.5)
                        vp.append(result['plddt'][0].mean().item())
                        vi.append(result['iptm'][0].item())
                        vtp.append(af2_p)
                        vti.append(af2_i)
                    except Exception:
                        continue
            vp, vi = np.array(vp), np.array(vi)
            vtp, vti = np.array(vtp), np.array(vti)
            from scipy.stats import spearmanr
            sp_p = spearmanr(vp, vtp)[0] if len(vp) > 2 else 0
            sp_i = spearmanr(vi, vti)[0] if len(vi) > 2 else 0
            sp_p = 0 if np.isnan(sp_p) else sp_p
            sp_i = 0 if np.isnan(sp_i) else sp_i
            metric = sp_p + sp_i + vp.std() * 5 + vi.std() * 5
            print(f"  Epoch {epoch+1:3d} | loss={np.mean(epoch_losses):.4f} | "
                  f"val: sp_p={sp_p:.3f} sp_i={sp_i:.3f} "
                  f"std_p={vp.std():.4f} std_i={vi.std():.4f}")
            if metric > best_val_metric:
                best_val_metric = metric
                save_path = OUT_DIR / "fixed_model_unfreeze_best.pt"
                torch.save({
                    'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
                    'epoch': epoch + 1,
                    'spearman_plddt': sp_p, 'spearman_iptm': sp_i,
                }, save_path)

    print(f"\n  Best metric: {best_val_metric:.4f}")
    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['diagnose', 'head-only', 'unfreeze', 'all'],
                        default='all')
    parser.add_argument('--epochs', type=int, default=0,
                        help='Override epochs (0=use defaults: 200/100)')
    parser.add_argument('--max-train', type=int, default=0,
                        help='Limit training entries (0=all)')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH} ({CKPT_PATH.stat().st_size/1024/1024:.1f} MB)")

    # Load model
    print("\nLoading model...")
    model, config = load_model(DEVICE)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load data
    print("\nLoading LMDB data...")
    train_entries = load_lmdb_batches(TRAIN_LMDB, max_entries=args.max_train)
    val_entries = load_lmdb_batches(VAL_LMDB)

    if args.phase in ('diagnose', 'all'):
        diag = diagnose(model, val_entries)
        (OUT_DIR / "diagnosis.json").write_text(
            json.dumps(diag, indent=2), encoding='utf-8')

    if args.phase in ('head-only', 'all'):
        ep = args.epochs if args.epochs > 0 else 200
        train_head_only(model, train_entries, val_entries, epochs=ep)

    if args.phase in ('unfreeze', 'all'):
        ep = args.epochs if args.epochs > 0 else 100
        train_unfreeze(model, train_entries, val_entries, epochs=ep)

    print("\n" + "=" * 70)
    print("DONE. Artifacts in calibration_artifacts/:")
    print("  - diagnosis.json")
    print("  - fixed_heads_best.pt (Phase 2)")
    print("  - fixed_model_unfreeze_best.pt (Phase 3)")
    print("=" * 70)


if __name__ == '__main__':
    main()
