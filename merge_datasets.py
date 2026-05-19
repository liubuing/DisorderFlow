#!/usr/bin/env python
"""Merge confidence dataset LMDBs across versions (v3-v6).

Usage:
  python merge_datasets.py --version v4    # Merge v3 + IDP v2 -> v4
  python merge_datasets.py --version v5    # Merge v4 + Disease v2 -> v5
  python merge_datasets.py --version v6    # Merge v5 + Brain Disease -> v6
  python merge_datasets.py --datasets DIR1 DIR2 --output_dir DIR  # Custom merge
"""
import sys, os, json, pickle, shutil, time
from pathlib import Path
import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent


def load_lmdb(db_path):
    import lmdb
    entries = []
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    with env.begin() as txn:
        n = pickle.loads(txn.get(b'__len__'))
        for i in range(n):
            entry = pickle.loads(txn.get(f'{i:08d}'.encode()))
            if 'af2_pae_matrix' in entry and isinstance(entry['af2_pae_matrix'], np.ndarray):
                entry['af2_pae_matrix'] = torch.from_numpy(entry['af2_pae_matrix']).float()
            if 'af2_plddt' in entry and isinstance(entry['af2_plddt'], list):
                entry['af2_plddt'] = torch.tensor(entry['af2_plddt'], dtype=torch.float32)
            if 'batch' in entry:
                entries.append(entry)
    env.close()
    return entries


def save_lmdb(db_path, entries):
    import lmdb
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    sample = pickle.dumps(entries[0])
    est_size = max(len(sample) * len(entries) * 3 + 10 * 1024 * 1024, 500 * 1024 * 1024)
    est_size = ((est_size + 4095) // 4096) * 4096
    print(f'  map_size: {est_size / 1024 / 1024:.0f} MB ({len(entries)} entries)')
    env = lmdb.open(str(db_path), map_size=est_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()


def merge_and_split(entry_groups, output_dir, train_split=0.8, seed=2026, label=None):
    """Merge entry groups, deduplicate by sequence, shuffle, split, save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    sources = {}
    for name, entries in entry_groups.items():
        all_entries.extend(entries)
        sources[name] = len(entries)
        print(f'  {name}: {len(entries)} entries')

    print(f'  Total before dedup: {len(all_entries)}')

    # Deduplicate by sequence
    seqs = set()
    unique = []
    dupes = 0
    for entry in all_entries:
        seq = entry.get('sequence', '')
        if seq and seq not in seqs:
            seqs.add(seq)
            unique.append(entry)
        elif not seq:
            unique.append(entry)
        else:
            dupes += 1

    if dupes:
        print(f'  Removed {dupes} duplicates')
    print(f'  Unique: {len(unique)}')

    if label:
        n_tagged = sum(1 for e in unique if e.get(label))
        print(f'  {label}={n_tagged}')

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(unique))
    split_idx = int(len(unique) * train_split)
    train = [unique[i] for i in indices[:split_idx]]
    val = [unique[i] for i in indices[split_idx:]]

    save_lmdb(output_dir / 'confidence_train.lmdb', train)
    save_lmdb(output_dir / 'confidence_val.lmdb', val)

    summary = {
        'n_train': len(train), 'n_val': len(val),
        'sources': sources, 'duplicates_removed': dupes,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'  Saved: {len(train)} train + {len(val)} val -> {output_dir}')
    return summary


def merge_v4():
    """v3 + IDP v2 -> v4"""
    print("=== Merge V4: v3 + IDP v2 ===")
    entry_groups = {}
    for name, path in [
        ('v3_train', PROJECT_DIR / 'data/confidence_merged_v3/confidence_train.lmdb'),
        ('v3_val', PROJECT_DIR / 'data/confidence_merged_v3/confidence_val.lmdb'),
    ]:
        if path.exists():
            entry_groups[name] = load_lmdb(path)
    idp_path = PROJECT_DIR / 'data/confidence_idp_v2'
    if idp_path.exists():
        entry_groups['idp_v2'] = load_lmdb(idp_path)
    merge_and_split(entry_groups, PROJECT_DIR / 'data/confidence_merged_v4', label='is_idp')


def merge_v5(disease_dir=None):
    """v4 + Disease v2 -> v5"""
    print("=== Merge V5: v4 + Disease v2 ===")
    entry_groups = {}
    for name, path in [
        ('v4_train', PROJECT_DIR / 'data/confidence_merged_v4/confidence_train.lmdb'),
        ('v4_val', PROJECT_DIR / 'data/confidence_merged_v4/confidence_val.lmdb'),
    ]:
        if path.exists():
            entry_groups[name] = load_lmdb(path)
    disease_path = Path(disease_dir) if disease_dir else PROJECT_DIR / 'data/confidence_dataset_disease_v2'
    if disease_path.exists():
        entry_groups['disease_v2'] = load_lmdb(disease_path)
    merge_and_split(entry_groups, PROJECT_DIR / 'data/confidence_merged_v5')


def merge_v6(brain_dir=None):
    """v5 + Brain Disease -> v6"""
    print("=== Merge V6: v5 + Brain Disease ===")
    entry_groups = {}
    for name, path in [
        ('v5_train', PROJECT_DIR / 'data/confidence_merged_v5/confidence_train.lmdb'),
        ('v5_val', PROJECT_DIR / 'data/confidence_merged_v5/confidence_val.lmdb'),
    ]:
        if path.exists():
            entry_groups[name] = load_lmdb(path)
    brain_path = Path(brain_dir) if brain_dir else PROJECT_DIR / 'data/confidence_dataset_brain_disease'
    if brain_path.exists():
        entry_groups['brain_disease'] = load_lmdb(brain_path)
    merge_and_split(entry_groups, PROJECT_DIR / 'data/confidence_merged_v6')


def main():
    import argparse
    p = argparse.ArgumentParser(description='Merge confidence datasets')
    p.add_argument('--version', choices=['v4', 'v5', 'v6'], help='Merge to specific version')
    p.add_argument('--datasets', nargs='+', help='Custom: LMDB directories to merge')
    p.add_argument('--output_dir', help='Custom: output directory')
    p.add_argument('--disease_dir', help='v5 only: disease v2 dataset directory')
    p.add_argument('--brain_dir', help='v6 only: brain disease dataset directory')
    p.add_argument('--train_split', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    if args.version == 'v4':
        merge_v4()
    elif args.version == 'v5':
        merge_v5(args.disease_dir)
    elif args.version == 'v6':
        merge_v6(args.brain_dir)
    elif args.datasets and args.output_dir:
        entry_groups = {}
        for ds in args.datasets:
            if os.path.isdir(ds) and (os.path.exists(os.path.join(ds, 'data.mdb')) or
                                       os.path.exists(os.path.join(ds, 'lock.mdb'))):
                entry_groups[os.path.basename(ds)] = load_lmdb(ds)
        if entry_groups:
            merge_and_split(entry_groups, args.output_dir, args.train_split, args.seed)
        else:
            print('No valid LMDB datasets found.')
    else:
        p.print_help()


if __name__ == '__main__':
    main()
