#!/usr/bin/env python
"""CDR Transformer Calibrator: sequence-context-aware confidence predictor.

Root cause of the Spearman ceiling: the geometric encoder reads COORDINATES,
which are identical across CDR variants of the same scaffold. The confidence
heads therefore see constant input. Fix: a Transformer on the CDR sequence
itself, capturing inter-residue epistasis that determines binding quality.

Architecture:
  CDR one-hot (L, 20) → positional encoding → 4-layer Transformer encoder
  → CLS-token pooling → MLP heads → (pLDDT, ipTM)

Trains on data/confidence_design_variants_v14/ (704 train, 128 val).
CDR positions identified by cdr_flag field in LMDB batches.

Usage:
  python scripts/cdr_transformer_calibrator.py --epochs 300
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
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

TRAIN_LMDB = str(PROJECT / "data" / "confidence_design_variants_v14" / "train_grouped.lmdb")
VAL_LMDB = str(PROJECT / "data" / "confidence_design_variants_v14" / "val_grouped.lmdb")
OUT_DIR = PROJECT / "calibration_artifacts"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AA = 'ACDEFGHIKLMNPQRSTVWY'


# ─── Model ───────────────────────────────────────────────────────────────────

class CDRTransformerCalibrator(nn.Module):
    """Small Transformer encoder on CDR sequence for confidence prediction.

    Unlike the mean-pooled MLP (SequenceCalibrator), this captures:
    - Inter-residue epistasis via self-attention
    - Position-specific effects via positional encoding
    - CDR-loop context (which CDR a residue belongs to)
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, dropout=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Input embedding: one-hot AA (20) + CDR type (7) + position
        self.aa_embed = nn.Linear(20, d_model)
        self.cdr_embed = nn.Embedding(8, d_model)  # 0=framework, 1-6=CDR types, 7=unknown
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Prediction heads
        self.head_plddt = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.head_iptm = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, aa_onehot, cdr_types, mask):
        """
        aa_onehot: (B, L, 20) one-hot amino acid
        cdr_types: (B, L) int, CDR assignment per residue
        mask: (B, L) bool, True = valid position
        Returns: (B,) pLDDT, (B,) ipTM
        """
        B, L, _ = aa_onehot.shape
        device = aa_onehot.device

        # Embed
        x = self.aa_embed(aa_onehot) + self.cdr_embed(cdr_types)
        x = x + self.pos_embed[:, :L, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+L, d_model)

        # Attention mask: CLS attends to all valid positions
        # TransformerEncoder expects: True = ignore (src_key_padding_mask)
        pad_mask = ~F.pad(mask, (1, 0), value=True)  # CLS never padded

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # CLS pooling
        cls_out = x[:, 0, :]  # (B, d_model)

        plddt = self.head_plddt(cls_out).squeeze(-1)  # (B,)
        iptm = self.head_iptm(cls_out).squeeze(-1)    # (B,)
        return plddt, iptm


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CDRSequenceDataset(Dataset):
    """Extract full antibody sequences + AF2 labels from LMDB.

    Since cdr_flag is not set in this dataset, we use the full sequence
    and assign CDR types based on canonical VHH CDR positions:
      CDR1: 26-33, CDR2: 51-58, CDR3: 97-113 (1-indexed, IMGT-like)
    """
    # VHH CDR ranges (0-indexed): CDR1=25:33, CDR2=50:58, CDR3=96:113
    CDR_RANGES = [(25, 33, 1), (50, 58, 2), (96, 113, 3)]

    def __init__(self, lmdb_path, max_seq_len=200):
        import lmdb
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
        self.max_seq_len = max_seq_len

        # Pre-scan for valid entries
        self.valid = []
        with self.env.begin() as txn:
            for i in range(self.length):
                entry = pickle.loads(txn.get(f'{i:08d}'.encode()))
                seq_len = len(entry.get('sequence', ''))
                if 0 < seq_len <= max_seq_len:
                    self.valid.append(i)
        print(f"[CDRSequenceDataset] {len(self.valid)}/{self.length} entries "
              f"(max_len={max_seq_len})")

    def _assign_cdr_types(self, length):
        """Assign CDR type per position: 0=framework, 1=CDR1, 2=CDR2, 3=CDR3."""
        types = torch.zeros(length, dtype=torch.long)
        for start, end, cdr_id in self.CDR_RANGES:
            if start < length:
                types[start:min(end, length)] = cdr_id
        return types

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        real_idx = self.valid[idx]
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(f'{real_idx:08d}'.encode()))

        batch = entry['batch']
        aa = batch['aa']  # (L,) int
        L = min(len(aa), self.max_seq_len)

        # Pad
        aa_padded = torch.zeros(self.max_seq_len, dtype=torch.long)
        aa_padded[:L] = aa[:L]
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:L] = True

        # CDR type assignment
        cdr_types = torch.zeros(self.max_seq_len, dtype=torch.long)
        cdr_types[:L] = self._assign_cdr_types(L)

        # One-hot
        aa_onehot = F.one_hot(aa_padded.clamp(0, 19), num_classes=20).float()

        # Targets
        af2_plddt = entry['af2_plddt']
        af2_plddt_mean = af2_plddt.mean().item() if af2_plddt.dim() > 0 else af2_plddt.item()
        af2_iptm = entry['af2_iptm']
        af2_iptm_val = af2_iptm.item() if isinstance(af2_iptm, torch.Tensor) else float(af2_iptm)
        scaffold_id = int(entry.get('scaffold_id', real_idx))

        return {
            'aa_onehot': aa_onehot,       # (max_seq_len, 20)
            'cdr_types': cdr_types,        # (max_seq_len,)
            'mask': mask,                  # (max_seq_len,)
            'af2_plddt': torch.tensor(af2_plddt_mean, dtype=torch.float32),
            'af2_iptm': torch.tensor(af2_iptm_val, dtype=torch.float32),
            'scaffold_id': torch.tensor(scaffold_id, dtype=torch.long),
        }


# ─── Losses ──────────────────────────────────────────────────────────────────

class PearsonLoss(nn.Module):
    def forward(self, pred, target):
        if pred.numel() < 3:
            return torch.tensor(0.0, device=pred.device)
        pc, tc = pred - pred.mean(), target - target.mean()
        corr = (pc * tc).sum() / (pc.pow(2).sum().sqrt().clamp(1e-8) *
                                   tc.pow(2).sum().sqrt().clamp(1e-8))
        return -corr

class VarianceLoss(nn.Module):
    def forward(self, pred, target):
        if pred.numel() < 4:
            return torch.tensor(0.0, device=pred.device)
        return F.relu(target.std() * 0.3 - pred.std())

class GroupedRankLoss(nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin
    def forward(self, pred, target, groups):
        loss = torch.tensor(0.0, device=pred.device)
        n = 0
        for gid in groups.unique():
            m = groups == gid
            if m.sum() < 2:
                continue
            p, t = pred[m], target[m]
            dp = p.unsqueeze(0) - p.unsqueeze(1)
            dt = t.unsqueeze(0) - t.unsqueeze(1)
            sign = dt.sign()
            viol = F.relu(self.margin - sign * dp)
            valid = (dt.abs() > 1e-6).float()
            loss = loss + (viol * valid).sum()
            n += valid.sum().item()
        return loss / max(n, 1)


# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    print(f"Device: {DEVICE}")
    print(f"CDR Transformer Calibrator | d_model={args.d_model} layers={args.layers} "
          f"nhead={args.nhead}")

    # Data
    train_ds = CDRSequenceDataset(TRAIN_LMDB, max_seq_len=args.max_seq_len)
    val_ds = CDRSequenceDataset(VAL_LMDB, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # Model
    model = CDRTransformerCalibrator(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.layers, dropout=0.15,
        max_len=args.max_seq_len,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Losses
    l1 = nn.SmoothL1Loss(beta=0.05)
    corr = PearsonLoss()
    var = VarianceLoss()
    rank = GroupedRankLoss(margin=0.01)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2)

    best_metric = -float('inf')
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            aa_oh = batch['aa_onehot'].to(DEVICE)
            cdr_t = batch['cdr_types'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            tp = batch['af2_plddt'].to(DEVICE)
            ti = batch['af2_iptm'].to(DEVICE)
            sids = batch['scaffold_id'].to(DEVICE)

            pp, pi = model(aa_oh, cdr_t, mask)

            loss = (l1(pp, tp) + l1(pi, ti)
                    + 0.8 * (corr(pp, tp) + corr(pi, ti))
                    + 0.5 * (var(pp, tp) + var(pi, ti))
                    + 0.3 * (rank(pp, tp, sids) + rank(pi, ti, sids)))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            vp, vi, vtp, vti = [], [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    aa_oh = batch['aa_onehot'].to(DEVICE)
                    cdr_t = batch['cdr_types'].to(DEVICE)
                    mask = batch['mask'].to(DEVICE)
                    pp, pi = model(aa_oh, cdr_t, mask)
                    vp.extend(pp.cpu().numpy())
                    vi.extend(pi.cpu().numpy())
                    vtp.extend(batch['af2_plddt'].numpy())
                    vti.extend(batch['af2_iptm'].numpy())

            vp, vi = np.array(vp), np.array(vi)
            vtp, vti = np.array(vtp), np.array(vti)
            from scipy.stats import spearmanr
            sp_p = spearmanr(vp, vtp)[0] if len(vp) > 2 else 0
            sp_i = spearmanr(vi, vti)[0] if len(vi) > 2 else 0
            sp_p = 0 if np.isnan(sp_p) else sp_p
            sp_i = 0 if np.isnan(sp_i) else sp_i

            metric = sp_p + sp_i + vp.std() * 3 + vi.std() * 3
            avg_loss = epoch_loss / max(n_batches, 1)

            print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                  f"sp_p={sp_p:.3f} sp_i={sp_i:.3f} "
                  f"std_p={vp.std():.4f} std_i={vi.std():.4f}")

            history.append({
                'epoch': epoch + 1, 'loss': float(avg_loss),
                'sp_plddt': float(sp_p), 'sp_iptm': float(sp_i),
                'std_plddt': float(vp.std()), 'std_iptm': float(vi.std()),
            })

            if metric > best_metric:
                best_metric = metric
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {'d_model': args.d_model, 'nhead': args.nhead,
                               'num_layers': args.layers, 'max_len': args.max_seq_len},
                    'epoch': epoch + 1,
                    'metrics': history[-1],
                }, OUT_DIR / "cdr_transformer_best.pt")

    # Final save
    torch.save({
        'model_state': model.state_dict(),
        'config': {'d_model': args.d_model, 'nhead': args.nhead,
                   'num_layers': args.layers, 'max_len': args.max_seq_len},
        'history': history,
    }, OUT_DIR / "cdr_transformer_final.pt")

    print(f"\n  Best metric: {best_metric:.4f}")
    print(f"  Saved → {OUT_DIR / 'cdr_transformer_best.pt'}")

    # Summary
    if history:
        best_sp_p = max(h['sp_plddt'] for h in history)
        best_sp_i = max(h['sp_iptm'] for h in history)
        print(f"\n  === RESULTS ===")
        print(f"  Best val Spearman pLDDT: {best_sp_p:.3f}")
        print(f"  Best val Spearman ipTM:  {best_sp_i:.3f}")
        print(f"  (Compare: fixed head = 0.481/0.463, seq MLP = 0.652/N/A)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--max-seq-len', type=int, default=200)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
