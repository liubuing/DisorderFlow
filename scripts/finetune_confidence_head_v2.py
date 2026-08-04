#!/usr/bin/env python
"""方案C: Confidence head fine-tuning with anti-collapse + correlation losses.

Two modes:
  A) Full model fine-tune (requires checkpoint):
     Freeze encoder, retrain head_plddt/head_iptm with:
       - SmoothL1 regression (existing)
       - Pearson correlation loss (new: directly optimizes rank correlation)
       - Variance regularization (anti-collapse: penalize low output std)
       - Within-scaffold ranking loss (margin-based)

  B) Standalone sequence calibrator (no checkpoint needed):
     Train a lightweight MLP on sequence features extracted from LMDB
     to predict AF2 scores. Acts as a "sequence-aware prior" that can
     break the constant-output degeneracy.

Usage:
  # Mode A (with checkpoint):
  python scripts/finetune_confidence_head_v2.py --mode full \
      --checkpoint logs/bfn_v14_.../checkpoints/best.pt \
      --data data/confidence_design_variants_v14/train_grouped.lmdb \
      --val-data data/confidence_design_variants_v14/val_grouped.lmdb

  # Mode B (standalone, no checkpoint):
  python scripts/finetune_confidence_head_v2.py --mode standalone \
      --data data/confidence_design_variants_v14/train_grouped.lmdb \
      --val-data data/confidence_design_variants_v14/val_grouped.lmdb
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


# ─── Losses ──────────────────────────────────────────────────────────────────

class PearsonCorrelationLoss(nn.Module):
    """Negative Pearson correlation as a loss (minimize → maximize correlation)."""
    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        if pred.numel() < 3:
            return torch.tensor(0.0, device=pred.device)
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        cov = (pred_c * target_c).sum()
        std_p = pred_c.pow(2).sum().sqrt().clamp(min=1e-8)
        std_t = target_c.pow(2).sum().sqrt().clamp(min=1e-8)
        corr = cov / (std_p * std_t)
        return -corr  # minimize negative correlation


class VarianceRegularization(nn.Module):
    """Penalize low variance in predictions (anti-collapse).

    If the model outputs near-constant values, this loss pushes it to
    produce varied predictions proportional to the target variance.
    """
    def __init__(self, target_std_threshold=0.02):
        super().__init__()
        self.threshold = target_std_threshold

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        if pred.numel() < 4:
            return torch.tensor(0.0, device=pred.device)
        pred_std = pred.std()
        target_std = target.std()
        # Hinge: penalize when pred_std < target_std * threshold_ratio
        desired_std = target_std.clamp(min=self.threshold)
        loss = F.relu(desired_std - pred_std)
        return loss


class GroupedRankingLoss(nn.Module):
    """Within-scaffold margin ranking loss.

    For designs sharing the same scaffold, if AF2 says design A > design B,
    then BFN prediction should also rank A > B (with margin).
    """
    def __init__(self, margin=0.02):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target, scaffold_ids):
        """pred, target: (B,) scalar predictions per design.
        scaffold_ids: (B,) int tensor grouping designs by scaffold.
        """
        loss = torch.tensor(0.0, device=pred.device)
        n_pairs = 0
        unique_scaffolds = scaffold_ids.unique()
        for sid in unique_scaffolds:
            mask = scaffold_ids == sid
            if mask.sum() < 2:
                continue
            p = pred[mask]
            t = target[mask]
            # All pairs within this scaffold
            n = p.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if t[i] > t[j]:
                        loss += F.relu(self.margin - (p[i] - p[j]))
                        n_pairs += 1
                    elif t[j] > t[i]:
                        loss += F.relu(self.margin - (p[j] - p[i]))
                        n_pairs += 1
        if n_pairs > 0:
            loss = loss / n_pairs
        return loss


# ─── Standalone Sequence Calibrator (Mode B) ─────────────────────────────────

class SequenceCalibrator(nn.Module):
    """Lightweight MLP that predicts AF2 scores from sequence features.

    Input features per residue:
      - One-hot AA (20)
      - Position encoding (sin/cos, 16)
    Aggregated via mean-pool → MLP → (pLDDT, ipTM)

    This breaks the constant-output degeneracy by being sequence-sensitive.
    """
    def __init__(self, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        input_dim = 20 + 16  # one-hot + positional
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.head_plddt = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.head_iptm = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def _positional_encoding(self, length, device):
        """Sinusoidal position encoding (16-dim)."""
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(8, device=device, dtype=torch.float32).unsqueeze(0)
        freq = 1.0 / (10000 ** (dim / 8.0))
        pe = torch.zeros(length, 16, device=device)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return pe

    def forward(self, aa_indices, mask=None):
        """aa_indices: (B, L) int tensor of AA indices [0..19].
        mask: (B, L) bool tensor (True = valid residue).
        Returns: (B,) pLDDT, (B,) ipTM
        """
        B, L = aa_indices.shape
        device = aa_indices.device
        # One-hot
        onehot = F.one_hot(aa_indices.clamp(0, 19), num_classes=20).float()  # (B, L, 20)
        # Positional
        pe = self._positional_encoding(L, device).unsqueeze(0).expand(B, -1, -1)  # (B, L, 16)
        x = torch.cat([onehot, pe], dim=-1)  # (B, L, 36)
        # Encode
        h = self.encoder(x)  # (B, L, hidden)
        # Masked mean pool
        if mask is None:
            mask = torch.ones(B, L, device=device, dtype=torch.bool)
        mask_f = mask.unsqueeze(-1).float()
        h_pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, hidden)
        plddt = self.head_plddt(h_pooled).squeeze(-1)  # (B,)
        iptm = self.head_iptm(h_pooled).squeeze(-1)  # (B,)
        return plddt, iptm


# ─── LMDB Dataset for standalone mode ────────────────────────────────────────

class LMDBSequenceDataset(Dataset):
    """Extract (sequence, af2_plddt, af2_iptm, scaffold_id) from LMDB."""
    def __init__(self, lmdb_path, max_len=512):
        import lmdb
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
        self.max_len = max_len
        # Pre-filter by length
        self.valid_indices = []
        with self.env.begin() as txn:
            for i in range(self.length):
                entry = pickle.loads(txn.get(f'{i:08d}'.encode()))
                seq = entry.get('sequence', '')
                if 0 < len(seq) <= max_len:
                    self.valid_indices.append(i)
        print(f"[LMDBSequenceDataset] {len(self.valid_indices)}/{self.length} entries "
              f"(max_len={max_len})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(f'{real_idx:08d}'.encode()))
        seq = entry['sequence'][:self.max_len]
        aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
        aa_indices = torch.tensor([aa_letters.index(c) if c in aa_letters else 0
                                   for c in seq], dtype=torch.long)
        af2_plddt = entry['af2_plddt']
        if af2_plddt.dim() > 0:
            af2_plddt_mean = af2_plddt[:len(seq)].mean().item()
        else:
            af2_plddt_mean = af2_plddt.item()
        af2_iptm = entry['af2_iptm']
        if af2_iptm.dim() > 0:
            af2_iptm_val = af2_iptm.mean().item()
        else:
            af2_iptm_val = af2_iptm.item()
        scaffold_id = int(entry.get('scaffold_id', real_idx))
        return {
            'aa': aa_indices,
            'length': len(seq),
            'af2_plddt': torch.tensor(af2_plddt_mean, dtype=torch.float32),
            'af2_iptm': torch.tensor(af2_iptm_val, dtype=torch.float32),
            'scaffold_id': torch.tensor(scaffold_id, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad sequences to max length in batch."""
    max_len = max(item['length'] for item in batch)
    B = len(batch)
    aa_padded = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    af2_plddt = torch.zeros(B)
    af2_iptm = torch.zeros(B)
    scaffold_ids = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        L = item['length']
        aa_padded[i, :L] = item['aa']
        mask[i, :L] = True
        af2_plddt[i] = item['af2_plddt']
        af2_iptm[i] = item['af2_iptm']
        scaffold_ids[i] = item['scaffold_id']
    return {
        'aa': aa_padded,
        'mask': mask,
        'af2_plddt': af2_plddt,
        'af2_iptm': af2_iptm,
        'scaffold_id': scaffold_ids,
    }


# ─── Training loop (Mode B: standalone) ──────────────────────────────────────

def train_standalone(args):
    """Train SequenceCalibrator on LMDB data."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Mode B] Standalone sequence calibrator | device={device}")

    # Data
    train_ds = LMDBSequenceDataset(args.data, max_len=args.max_len)
    val_ds = LMDBSequenceDataset(args.val_data, max_len=args.max_len) if args.val_data else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0) if val_ds else None

    # Model
    model = SequenceCalibrator(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                               dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Losses
    l1_loss = nn.SmoothL1Loss(beta=0.05)
    corr_loss = PearsonCorrelationLoss()
    var_reg = VarianceRegularization(target_std_threshold=0.02)
    rank_loss = GroupedRankingLoss(margin=0.02)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss weights
    w_l1 = 1.0
    w_corr = 0.5
    w_var = 0.3
    w_rank = 0.2

    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {'total': 0, 'l1_p': 0, 'l1_i': 0, 'corr_p': 0,
                        'corr_i': 0, 'var_p': 0, 'var_i': 0, 'rank_p': 0, 'rank_i': 0}
        n_batches = 0

        for batch in train_loader:
            aa = batch['aa'].to(device)
            mask = batch['mask'].to(device)
            target_p = batch['af2_plddt'].to(device)
            target_i = batch['af2_iptm'].to(device)
            scaffold_ids = batch['scaffold_id'].to(device)

            pred_p, pred_i = model(aa, mask)

            # L1 regression
            loss_l1_p = l1_loss(pred_p, target_p)
            loss_l1_i = l1_loss(pred_i, target_i)

            # Correlation
            loss_corr_p = corr_loss(pred_p, target_p)
            loss_corr_i = corr_loss(pred_i, target_i)

            # Variance anti-collapse
            loss_var_p = var_reg(pred_p, target_p)
            loss_var_i = var_reg(pred_i, target_i)

            # Grouped ranking
            loss_rank_p = rank_loss(pred_p, target_p, scaffold_ids)
            loss_rank_i = rank_loss(pred_i, target_i, scaffold_ids)

            total = (w_l1 * (loss_l1_p + loss_l1_i)
                     + w_corr * (loss_corr_p + loss_corr_i)
                     + w_var * (loss_var_p + loss_var_i)
                     + w_rank * (loss_rank_p + loss_rank_i))

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['total'] += total.item()
            epoch_losses['l1_p'] += loss_l1_p.item()
            epoch_losses['l1_i'] += loss_l1_i.item()
            epoch_losses['corr_p'] += loss_corr_p.item()
            epoch_losses['corr_i'] += loss_corr_i.item()
            epoch_losses['var_p'] += loss_var_p.item()
            epoch_losses['var_i'] += loss_var_i.item()
            epoch_losses['rank_p'] += loss_rank_p.item()
            epoch_losses['rank_i'] += loss_rank_i.item()
            n_batches += 1

        scheduler.step()

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validation
        val_metrics = {}
        if val_loader:
            model.eval()
            val_preds_p, val_preds_i, val_targets_p, val_targets_i = [], [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    aa = batch['aa'].to(device)
                    mask = batch['mask'].to(device)
                    pred_p, pred_i = model(aa, mask)
                    val_preds_p.append(pred_p.cpu())
                    val_preds_i.append(pred_i.cpu())
                    val_targets_p.append(batch['af2_plddt'])
                    val_targets_i.append(batch['af2_iptm'])
            vp = torch.cat(val_preds_p).numpy()
            vi = torch.cat(val_preds_i).numpy()
            tp = torch.cat(val_targets_p).numpy()
            ti = torch.cat(val_targets_i).numpy()
            from scipy.stats import spearmanr, pearsonr
            sp_p = spearmanr(vp, tp)[0] if len(vp) > 2 else 0
            sp_i = spearmanr(vi, ti)[0] if len(vi) > 2 else 0
            val_metrics = {
                'spearman_plddt': float(sp_p) if not np.isnan(sp_p) else 0,
                'spearman_iptm': float(sp_i) if not np.isnan(sp_i) else 0,
                'pred_std_plddt': float(vp.std()),
                'pred_std_iptm': float(vi.std()),
                'mae_plddt': float(np.abs(vp - tp).mean()),
                'mae_iptm': float(np.abs(vi - ti).mean()),
            }
            val_loss = val_metrics['mae_plddt'] + val_metrics['mae_iptm']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = PROJECT / "calibration_artifacts" / "sequence_calibrator_best.pt"
                save_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {'hidden_dim': args.hidden_dim,
                               'num_layers': args.num_layers,
                               'max_len': args.max_len},
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                }, save_path)

        entry = {'epoch': epoch, **epoch_losses, **val_metrics}
        history.append(entry)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = (f"  Epoch {epoch+1:3d}/{args.epochs} | "
                   f"loss={epoch_losses['total']:.4f} "
                   f"(l1_p={epoch_losses['l1_p']:.4f} corr_p={epoch_losses['corr_p']:.3f} "
                   f"var_p={epoch_losses['var_p']:.4f})")
            if val_metrics:
                msg += (f" | val: sp_p={val_metrics['spearman_plddt']:.3f} "
                        f"sp_i={val_metrics['spearman_iptm']:.3f} "
                        f"std_p={val_metrics['pred_std_plddt']:.4f}")
            print(msg)

    # Save final
    final_path = PROJECT / "calibration_artifacts" / "sequence_calibrator_final.pt"
    torch.save({
        'model_state': model.state_dict(),
        'config': {'hidden_dim': args.hidden_dim,
                   'num_layers': args.num_layers,
                   'max_len': args.max_len},
        'history': history,
    }, final_path)
    print(f"\n  Saved final model → {final_path}")
    if best_val_loss < float('inf'):
        print(f"  Best val loss: {best_val_loss:.4f}")
    return history


# ─── Training loop (Mode A: full model head fine-tune) ───────────────────────

def train_full_model(args):
    """Fine-tune confidence heads on the full BFN model (requires checkpoint)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Mode A] Full model head fine-tune | device={device}")

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print("ERROR: --checkpoint required and must exist for Mode A.")
        print("  The checkpoint .pt files appear to have been cleaned from logs/.")
        print("  Options:")
        print("    1. Re-download from HuggingFace (if uploaded)")
        print("    2. Re-train from scratch: python train.py --config configs/train/bfn_v14_grouped_conf_xpu.yml")
        print("    3. Use Mode B (standalone) which doesn't need a checkpoint")
        sys.exit(1)

    from disorderflow.models import get_model
    from disorderflow.utils.misc import load_config
    from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset

    # Load model
    config, _ = load_config(PROJECT / 'configs' / 'demo_design.yml')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(mc).to(device)
    model.load_state_dict(ckpt['model'], strict=False)

    # Freeze everything except confidence heads
    trainable_suffixes = ['head_plddt', 'head_iptm', 'head_pae',
                          'v12_plddt', 'v12_iptm', 'v12_pae', 'v12_seq_emb',
                          'v14_iptm_bb', 'head_plddt_seq']
    for name, param in model.named_parameters():
        param.requires_grad = any(name.endswith(s) or f'.{s}.' in name
                                  for s in trainable_suffixes)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} params")

    # Dataset
    train_cfg = {'db_path': args.data, 'max_residues': args.max_len}
    train_ds = ConfidenceRegressionDataset(train_cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)

    # Losses
    l1_loss = nn.SmoothL1Loss(beta=0.05)
    corr_loss = PearsonCorrelationLoss()
    var_reg = VarianceRegularization()
    rank_loss = GroupedRankingLoss(margin=0.02)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            # Forward through encoder + heads
            # (This requires the model's forward method to return confidence outputs)
            # Use score_fixed pathway
            with torch.set_grad_enabled(True):
                result = model.bfn.score_fixed(batch, fixed_t=0.5)
            pred_plddt = result['plddt']  # (N, L)
            pred_iptm = result['iptm']    # (N,)

            # Targets
            target_plddt = batch.get('af2_plddt')
            target_iptm = batch.get('af2_iptm')
            if target_plddt is None:
                continue
            mask = batch['mask'].bool()

            # Per-residue pLDDT loss
            pred_p_mean = (pred_plddt * mask.float()).sum(1) / mask.float().sum(1).clamp(1)
            target_p_mean = target_plddt.mean(1) if target_plddt.dim() > 1 else target_plddt

            loss_l1 = l1_loss(pred_p_mean, target_p_mean) + l1_loss(pred_iptm, target_iptm)
            loss_corr = corr_loss(pred_p_mean, target_p_mean) + corr_loss(pred_iptm, target_iptm)
            loss_var = var_reg(pred_p_mean, target_p_mean) + var_reg(pred_iptm, target_iptm)

            scaffold_ids = batch.get('scaffold_id', torch.arange(len(pred_iptm), device=device))
            loss_rank = rank_loss(pred_iptm, target_iptm, scaffold_ids)

            total = loss_l1 + 0.5 * loss_corr + 0.3 * loss_var + 0.2 * loss_rank
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} | loss={epoch_loss/max(n_batches,1):.4f}")

    # Save fine-tuned heads only
    save_path = PROJECT / "calibration_artifacts" / "confidence_heads_finetuned.pt"
    save_path.parent.mkdir(exist_ok=True)
    head_state = {k: v for k, v in model.state_dict().items()
                  if any(s in k for s in trainable_suffixes)}
    torch.save(head_state, save_path)
    print(f"  Saved fine-tuned heads → {save_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Confidence head calibration/fine-tuning")
    parser.add_argument('--mode', choices=['full', 'standalone'], default='standalone',
                        help='full=fine-tune model heads (needs checkpoint), '
                             'standalone=train sequence calibrator (no checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to BFN checkpoint .pt (Mode A only)')
    parser.add_argument('--data', type=str,
                        default=str(PROJECT / 'data/confidence_design_variants_v14/train_grouped.lmdb'),
                        help='Training LMDB path')
    parser.add_argument('--val-data', type=str,
                        default=str(PROJECT / 'data/confidence_design_variants_v14/val_grouped.lmdb'),
                        help='Validation LMDB path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--max-len', type=int, default=512)
    args = parser.parse_args()

    if args.mode == 'standalone':
        train_standalone(args)
    else:
        train_full_model(args)


if __name__ == '__main__':
    main()
