#!/usr/bin/env python
"""Merge V5 training data + Brain Disease → V6 training dataset.

Adds 1,123 brain/neurological disease proteins to the existing V5 dataset.
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
            if 'af2_pae_matrix' in entry:
                if isinstance(entry['af2_pae_matrix'], np.ndarray):
                    entry['af2_pae_matrix'] = torch.from_numpy(entry['af2_pae_matrix']).float()
            if 'af2_plddt' in entry:
                if isinstance(entry['af2_plddt'], list):
                    entry['af2_plddt'] = torch.tensor(entry['af2_plddt'], dtype=torch.float32)
            if 'batch' not in entry:
                continue
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
    print(f'  Estimated map_size: {est_size / 1024 / 1024:.0f} MB ({len(entries)} entries)')
    env = lmdb.open(str(db_path), map_size=est_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--brain_disease_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset_brain_disease'))
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_merged_v6'))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    seed = 2026

    all_entries = []
    sources = {}

    # 1. Load V5 train + val
    for name, path in [
        ('v5_train', PROJECT_DIR / 'data' / 'confidence_merged_v5' / 'confidence_train.lmdb'),
        ('v5_val', PROJECT_DIR / 'data' / 'confidence_merged_v5' / 'confidence_val.lmdb'),
    ]:
        if path.exists():
            entries = load_lmdb(path)
            all_entries.extend(entries)
            sources[name] = len(entries)
            print(f'Loaded {len(entries)} from {name}')
        else:
            print(f'WARNING: {name} not found at {path}')

    # 2. Load brain disease
    brain_path = Path(args.brain_disease_dir) / 'brain_disease.lmdb'
    if brain_path.exists():
        entries = load_lmdb(brain_path)
        all_entries.extend(entries)
        sources['brain_disease'] = len(entries)
        print(f'Loaded {len(entries)} from brain_disease')

        from collections import Counter
        cat_counts = Counter(e.get('category', 'unknown') for e in entries)
        print(f'  Brain disease categories ({len(cat_counts)}): {dict(cat_counts.most_common(10))}...')
    else:
        print(f'WARNING: Brain disease not found at {brain_path}')
        sys.exit(1)

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

    # Tag brain disease entries
    n_brain = 0
    for e in unique_entries:
        if 'category' in e and e.get('source', '') == '':
            e['source'] = f'BrainDisease_{e["category"]}'
        if 'BrainDisease' in (e.get('source', '') or ''):
            n_brain += 1

    from collections import Counter
    source_counts = Counter(e.get('source', 'unknown') for e in unique_entries)
    n_idp = sum(1 for e in unique_entries if e.get('is_idp'))
    n_disease = sum(1 for e in unique_entries if 'Disease' in (e.get('source', '') or ''))

    print(f'IDP-tagged: {n_idp}')
    print(f'Disease-tagged (all types): {n_disease}')
    print(f'Brain disease: {n_brain}')

    # Shuffle and split 80/20
    np.random.seed(seed)
    indices = np.random.permutation(len(unique_entries))
    split_idx = int(len(unique_entries) * 0.8)
    train = [unique_entries[i] for i in indices[:split_idx]]
    val = [unique_entries[i] for i in indices[split_idx:]]

    n_brain_train = sum(1 for e in train if 'BrainDisease' in (e.get('source', '') or ''))
    n_brain_val = sum(1 for e in val if 'BrainDisease' in (e.get('source', '') or ''))
    n_idp_train = sum(1 for e in train if e.get('is_idp'))
    n_idp_val = sum(1 for e in val if e.get('is_idp'))

    print(f'\nTrain: {len(train)} ({n_idp_train} IDP, {n_brain_train} brain disease)')
    print(f'Val:   {len(val)} ({n_idp_val} IDP, {n_brain_val} brain disease)')

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_lmdb(output_dir / 'confidence_train.lmdb', train)
    save_lmdb(output_dir / 'confidence_val.lmdb', val)

    # Summary
    summary = {
        'n_train': len(train),
        'n_val': len(val),
        'n_brain_train': n_brain_train,
        'n_brain_val': n_brain_val,
        'n_idp_train': n_idp_train,
        'n_idp_val': n_idp_val,
        'sources': sources,
        'duplicates_removed': dupes,
        'total_unique': len(unique_entries),
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
