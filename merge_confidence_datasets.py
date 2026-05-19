#!/usr/bin/env python
"""Merge multiple confidence dataset LMDBs into a single combined dataset."""
import sys, os, json, pickle, shutil, time
from pathlib import Path
import numpy as np

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
                import torch
                if isinstance(entry['af2_pae_matrix'], np.ndarray):
                    entry['af2_pae_matrix'] = torch.from_numpy(entry['af2_pae_matrix']).float()
            if 'af2_plddt' in entry:
                import torch
                if isinstance(entry['af2_plddt'], list):
                    entry['af2_plddt'] = torch.tensor(entry['af2_plddt'], dtype=torch.float32)
            entries.append(entry)
    env.close()
    return entries


def save_lmdb(db_path, entries):
    """Save entries to LMDB."""
    import lmdb
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    sample = pickle.dumps(entries[0])
    est_size = max(len(sample) * len(entries) * 2 + 1024 * 1024, 10 * 1024 * 1024)
    env = lmdb.open(str(db_path), map_size=est_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description='Merge confidence datasets')
    p.add_argument('--datasets', nargs='+', required=True,
                   help='LMDB directories to merge (specify train/val pairs as dir1_train,dir1_val dir2_train,dir2_val ...)')
    p.add_argument('--output_dir', required=True, help='Output directory for merged LMDBs')
    p.add_argument('--train_split', type=float, default=0.8, help='Train/val split ratio')
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    sources = {}

    for ds_path in args.datasets:
        if not os.path.exists(ds_path):
            print(f'WARNING: {ds_path} not found, skipping')
            continue

        if os.path.isdir(ds_path):
            # Check if it's a direct LMDB
            if os.path.exists(os.path.join(ds_path, 'data.mdb')) or os.path.exists(os.path.join(ds_path, 'lock.mdb')):
                entries = load_lmdb(ds_path)
                print(f'Loaded {len(entries)} from {ds_path}')
                all_entries.extend(entries)
                sources[ds_path] = len(entries)
            else:
                print(f'WARNING: {ds_path} is not an LMDB, skipping')

    if not all_entries:
        print('No entries loaded!')
        sys.exit(1)

    print(f'\nTotal entries: {len(all_entries)}')

    # Deduplicate by sequence
    seqs = set()
    unique_entries = []
    dupes = 0
    for entry in all_entries:
        seq = entry.get('sequence', '')
        if seq not in seqs:
            seqs.add(seq)
            unique_entries.append(entry)
        else:
            dupes += 1

    if dupes > 0:
        print(f'Removed {dupes} duplicate entries')
    print(f'Unique entries: {len(unique_entries)}')

    # Shuffle and split
    np.random.seed(args.seed)
    indices = np.random.permutation(len(unique_entries))
    split_idx = int(len(unique_entries) * args.train_split)
    train = [unique_entries[i] for i in indices[:split_idx]]
    val = [unique_entries[i] for i in indices[split_idx:]]

    # Save
    save_lmdb(output_dir / 'confidence_train.lmdb', train)
    save_lmdb(output_dir / 'confidence_val.lmdb', val)
    print(f'\nSaved {len(train)} train + {len(val)} val entries to {output_dir}')

    # Summary
    summary = {
        'n_train': len(train),
        'n_val': len(val),
        'sources': sources,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to {output_dir / "dataset_summary.json"}')


if __name__ == '__main__':
    main()
