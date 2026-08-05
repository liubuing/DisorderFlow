"""Preprocess IDP LMDB entries to BFN training format.
Fixes batch dimension, adds mask key, normalizes disorder labels.
Output: data/confidence_idp_bfn_ready/ — training-ready LMDB.
"""
import sys, os, pickle, lmdb
import numpy as np
import torch
from pathlib import Path

SRC = 'data/confidence_idp_unified'
DST = 'data/confidence_idp_bfn_ready'
os.makedirs(DST, exist_ok=True)

src_env = lmdb.open(SRC, readonly=True, lock=False, readahead=False, subdir=True)
dst_env = lmdb.open(DST, map_size=20 * 1024**3, subdir=True)

entries = []
with src_env.begin() as txn:
    for key, value in txn.cursor():
        if key == b'__len__':
            continue
        try:
            rec = pickle.loads(value)
        except:
            continue
        if 'batch' not in rec or 'disorder_mask' not in rec:
            continue
        entries.append((key, rec))

print(f"IDP entries: {len(entries)}")

converted = 0
with dst_env.begin(write=True) as txn:
    for idx, (key, rec) in enumerate(entries):
        batch = rec['batch']
        disorder = rec['disorder_mask']
        if isinstance(disorder, torch.Tensor):
            disorder = disorder.numpy()

        # Build training-ready batch with batch dim
        train_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                train_batch[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                train_batch[k] = torch.tensor(v).unsqueeze(0)
            else:
                train_batch[k] = v
        
        # Remove only the multi-chain key that causes PaddingCollate issues
        train_batch.pop('chain_nb', None)
        train_batch.pop('chain_id', None)

        # Add disorder label (clamped to [0,1])
        L = train_batch['aa'].shape[1]
        disorder_np = np.float32(np.clip(disorder[:L], 0, 1))
        train_batch['disorder_label'] = torch.tensor(disorder_np).unsqueeze(0)

        # Add mask key (required by BFN encoder)
        if 'mask' not in train_batch and 'mask_heavyatom' in train_batch:
            train_batch['mask'] = train_batch['mask_heavyatom'][:, :, :4].any(dim=2)
        
        # Filter by sequence length (BFN limit)
        if L > 500:
            continue

        # Filter entries with meaningful disorder signal
        if disorder_np.mean() < 0.05:
            continue

        txn.put(key, pickle.dumps(train_batch))
        converted += 1

        if converted <= 3:
            print(f"  [{converted}] {key.decode()}: L={L}, "
                  f"disordered={(disorder_np>=0.5).sum()}/{L} "
                  f"({disorder_np.mean():.1%})")

    txn.put(b'__len__', pickle.dumps(converted))
dst_env.close()
src_env.close()

print(f"\nConverted: {converted} entries → {DST}")
print(f"Ready for BFN training with batch['disorder_label']")
