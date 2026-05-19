#!/usr/bin/env python
"""Merge v3 training data + IDP v2 data → v4 training dataset.

Usage:
  python merge_v4_dataset.py
"""

import sys, os, json, pickle, shutil, time
from pathlib import Path
import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent


def load_lmdb(db_path):
    """Load all entries from an LMDB database."""
    import lmdb
    entries = []
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    with env.begin() as txn:
        n = pickle.loads(txn.get(b'__len__'))
        for i in range(n):
            entry = pickle.loads(txn.get(f'{i:08d}'.encode()))
            # Normalize entry format
            if 'af2_pae_matrix' in entry:
                if isinstance(entry['af2_pae_matrix'], np.ndarray):
                    entry['af2_pae_matrix'] = torch.from_numpy(entry['af2_pae_matrix']).float()
            if 'af2_plddt' in entry:
                if isinstance(entry['af2_plddt'], list):
                    entry['af2_plddt'] = torch.tensor(entry['af2_plddt'], dtype=torch.float32)
            # Ensure entry has the essential keys for ConfidenceRegressionDataset
            if 'batch' not in entry:
                continue  # Skip entries without batch
            entries.append(entry)
    env.close()
    return entries


def save_lmdb(db_path, entries):
    """Save entries to LMDB."""
    import lmdb
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    sample = pickle.dumps(entries[0])
    est_size = max(len(sample) * len(entries) * 3 + 10 * 1024 * 1024, 500 * 1024 * 1024)
    # LMDB requires map_size to be a multiple of page size (4096)
    est_size = ((est_size + 4095) // 4096) * 4096
    print(f'  Estimated map_size: {est_size / 1024 / 1024:.0f} MB ({len(entries)} entries)')
    env = lmdb.open(str(db_path), map_size=est_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()


def main():
    output_dir = PROJECT_DIR / 'data' / 'confidence_merged_v4'
    seed = 2026

    all_entries = []
    sources = {}

    # Load v3 train + val
    for name, path in [
        ('v3_train', PROJECT_DIR / 'data' / 'confidence_merged_v3' / 'confidence_train.lmdb'),
        ('v3_val', PROJECT_DIR / 'data' / 'confidence_merged_v3' / 'confidence_val.lmdb'),
    ]:
        if path.exists():
            entries = load_lmdb(path)
            all_entries.extend(entries)
            sources[name] = len(entries)
            print(f'Loaded {len(entries)} from {name}')

    # Load IDP v2
    idp_path = PROJECT_DIR / 'data' / 'confidence_idp_v2'
    if idp_path.exists():
        entries = load_lmdb(idp_path)
        all_entries.extend(entries)
        sources['idp_v2'] = len(entries)
        print(f'Loaded {len(entries)} from idp_v2')
        # Count is_idp
        n_idp = sum(1 for e in entries if e.get('is_idp'))
        print(f'  (of which {n_idp} are tagged is_idp=True)')
    else:
        print('WARNING: IDP v2 dataset not found, merging only v3')

    print(f'\nTotal entries before dedup: {len(all_entries)}')

    # Deduplicate by sequence
    seqs = set()
    unique_entries = []
    dupes = 0
    for entry in all_entries:
        seq = entry.get('sequence', '')
        if seq and seq not in seqs:
            seqs.add(seq)
            unique_entries.append(entry)
        elif not seq:
            unique_entries.append(entry)
        else:
            dupes += 1

    if dupes > 0:
        print(f'Removed {dupes} duplicate entries (same sequence)')
    print(f'Unique entries: {len(unique_entries)}')

    # Count IDP tagged entries in final set
    n_idp_final = sum(1 for e in unique_entries if e.get('is_idp'))
    print(f'IDP-tagged entries: {n_idp_final}')

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(unique_entries))
    split_idx = int(len(unique_entries) * 0.8)
    train = [unique_entries[i] for i in indices[:split_idx]]
    val = [unique_entries[i] for i in indices[split_idx:]]

    # Count IDPs in each split
    n_idp_train = sum(1 for e in train if e.get('is_idp'))
    n_idp_val = sum(1 for e in val if e.get('is_idp'))
    print(f'Train: {len(train)} ({n_idp_train} IDP), Val: {len(val)} ({n_idp_val} IDP)')

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_lmdb(output_dir / 'confidence_train.lmdb', train)
    save_lmdb(output_dir / 'confidence_val.lmdb', val)

    # Summary
    summary = {
        'n_train': len(train),
        'n_val': len(val),
        'n_idp_train': n_idp_train,
        'n_idp_val': n_idp_val,
        'sources': sources,
        'duplicates_removed': dupes,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved to {output_dir}')
    print(f'  confidence_train.lmdb: {len(train)} entries')
    print(f'  confidence_val.lmdb: {len(val)} entries')
    print('Done!')


if __name__ == '__main__':
    main()
