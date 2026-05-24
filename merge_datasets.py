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


def merge_v7(liver_dir=None):
    """v6 + Liver Disease -> v7"""
    print("=== Merge V7: v6 + Liver Disease ===")
    entry_groups = {}
    for name, path in [
        ('v6_train', PROJECT_DIR / 'data/confidence_merged_v6/confidence_train.lmdb'),
        ('v6_val', PROJECT_DIR / 'data/confidence_merged_v6/confidence_val.lmdb'),
    ]:
        if path.exists():
            entry_groups[name] = load_lmdb(path)
    # Liver disease LMDB is inside the output directory
    liver_base = Path(liver_dir) if liver_dir else PROJECT_DIR / 'data/confidence_dataset_liver_disease'
    liver_lmdb = liver_base / 'liver_disease.lmdb'
    if liver_lmdb.exists():
        entry_groups['liver_disease'] = load_lmdb(liver_lmdb)
    elif liver_base.exists():
        # Fallback: try loading the base directory (legacy format)
        entry_groups['liver_disease'] = load_lmdb(liver_base)
    merge_and_split(entry_groups, PROJECT_DIR / 'data/confidence_merged_v7')


def merge_v8(brain_dir=None, v8_expansion_dir=None, liver_dir=None, iptm_threshold=0.6):
    """V8: quality-filtered V6 brain + V8 brain expansion + quality liver subset -> V8

    Strategy (depth > breadth):
      - V6 brain disease: keep only ipTM > 0.6 entries (removes ~50% low-quality)
      - V8 brain expansion: new brain disease categories, already ipTM > 0.6
      - V7 liver disease: keep only top 150 by ipTM (quality over quantity)
      - V6 general (V5 base): keep only ipTM > 0.6
    """
    print("=== Merge V8: Quality-filtered brain-focused dataset ===")
    entry_groups = {}

    # V6 brain disease with quality filter
    brain_path = Path(brain_dir) if brain_dir else PROJECT_DIR / 'data/confidence_dataset_brain_disease'
    brain_lmdb = brain_path / 'brain_disease.lmdb'
    if brain_lmdb.exists():
        all_brain = load_lmdb(brain_lmdb)
        brain_quality = []
        brain_low = 0
        for e in all_brain:
            iptm = e.get('af2_iptm', 0)
            if hasattr(iptm, 'item'):
                iptm = iptm.item()
            if iptm >= iptm_threshold:
                brain_quality.append(e)
            else:
                brain_low += 1
        entry_groups['v6_brain_quality'] = brain_quality
        print(f'  V6 brain: {len(all_brain)} total, {len(brain_quality)} pass ipTM>{iptm_threshold} ({brain_low} filtered)')
    elif brain_path.exists():
        all_brain = load_lmdb(brain_path)
        brain_quality = []
        brain_low = 0
        for e in all_brain:
            iptm = e.get('af2_iptm', 0)
            if hasattr(iptm, 'item'):
                iptm = iptm.item()
            if iptm >= iptm_threshold:
                brain_quality.append(e)
            else:
                brain_low += 1
        entry_groups['v6_brain_quality'] = brain_quality
        print(f'  V6 brain: {len(all_brain)} total, {len(brain_quality)} pass ipTM>{iptm_threshold} ({brain_low} filtered)')

    # V8 brain expansion (already quality-filtered during build)
    v8_path = Path(v8_expansion_dir) if v8_expansion_dir else PROJECT_DIR / 'data/confidence_v8_brain_expansion'
    v8_lmdb = v8_path / 'v8_brain_expansion.lmdb'
    if v8_lmdb.exists():
        entry_groups['v8_brain_expansion'] = load_lmdb(v8_lmdb)
        print(f'  V8 brain expansion: {len(entry_groups["v8_brain_expansion"])} entries')
    elif v8_path.exists():
        entry_groups['v8_brain_expansion'] = load_lmdb(v8_path)

    # V6 general proteins (from v5 merged) with quality filter
    v6_train = PROJECT_DIR / 'data/confidence_merged_v6/confidence_train.lmdb'
    v6_val = PROJECT_DIR / 'data/confidence_merged_v6/confidence_val.lmdb'
    general_entries = []
    for path in [v6_train, v6_val]:
        if path.exists():
            entries = load_lmdb(path)
            quality = []
            for e in entries:
                iptm = e.get('af2_iptm', 0)
                if hasattr(iptm, 'item'):
                    iptm = iptm.item()
                if iptm >= iptm_threshold:
                    quality.append(e)
            general_entries.extend(quality)
    if general_entries:
        entry_groups['v6_general_quality'] = general_entries
        print(f'  V6 general (quality-filtered): {len(general_entries)} entries')

    # V7 liver disease: top 200 by ipTM only
    liver_base = Path(liver_dir) if liver_dir else PROJECT_DIR / 'data/confidence_dataset_liver_disease'
    liver_lmdb = liver_base / 'liver_disease.lmdb'
    if liver_lmdb.exists():
        all_liver = load_lmdb(liver_lmdb)
        liver_sorted = sorted(all_liver, key=lambda e: (
            e.get('af2_iptm').item() if hasattr(e.get('af2_iptm', 0), 'item') else e.get('af2_iptm', 0)
        ), reverse=True)
        top_liver = liver_sorted[:200]
        entry_groups['liver_top200'] = top_liver
        print(f'  Liver disease: {len(all_liver)} total, keeping top 200 by ipTM')
    elif liver_base.exists():
        all_liver = load_lmdb(liver_base)
        liver_sorted = sorted(all_liver, key=lambda e: (
            e.get('af2_iptm').item() if hasattr(e.get('af2_iptm', 0), 'item') else e.get('af2_iptm', 0)
        ), reverse=True)
        top_liver = liver_sorted[:200]
        entry_groups['liver_top200'] = top_liver
        print(f'  Liver disease: {len(all_liver)} total, keeping top 200 by ipTM')

    output_dir = PROJECT_DIR / 'data/confidence_merged_v8'
    return merge_and_split(entry_groups, output_dir, label='is_idp')


def merge_v9(idp_dir=None):
    """V8 + IDP v3 -> V9 (IDP-aware, no ipTM filter on IDPs)

    V8 already has ipTM>0.6 quality filter on folded proteins.
    IDP v3 entries are added AS-IS — they naturally have low ipTM
    and would be incorrectly filtered out by the ipTM threshold.
    """
    print("=== Merge V9: V8 + IDP v3 (IDP-preserving) ===")
    entry_groups = {}

    # Load V8 (already quality-filtered)
    for name, path in [
        ('v8_train', PROJECT_DIR / 'data/confidence_merged_v8/confidence_train.lmdb'),
        ('v8_val', PROJECT_DIR / 'data/confidence_merged_v8/confidence_val.lmdb'),
    ]:
        if path.exists():
            entry_groups[name] = load_lmdb(path)
            n_idp = sum(1 for e in entry_groups[name] if e.get('is_idp'))
            print(f'  {name}: {len(entry_groups[name])} entries ({n_idp} IDP)')

    # Load IDP v3
    idp_path = Path(idp_dir) if idp_dir else PROJECT_DIR / 'data/confidence_idp_v3'
    if idp_path.exists():
        idp_entries = load_lmdb(idp_path)
        entry_groups['idp_v3'] = idp_entries
        n_idp = sum(1 for e in idp_entries if e.get('is_idp'))
        print(f'  idp_v3: {len(idp_entries)} entries ({n_idp} IDP)')
    else:
        print(f'  WARNING: IDP v3 not found at {idp_path}')

    output_dir = PROJECT_DIR / 'data/confidence_merged_v9'
    summary = merge_and_split(entry_groups, output_dir, label='is_idp')

    # Explicit IDP count for training_config reference
    n_idp_train = sum(1 for e in load_lmdb(output_dir / 'confidence_train.lmdb') if e.get('is_idp'))
    n_idp_val = sum(1 for e in load_lmdb(output_dir / 'confidence_val.lmdb') if e.get('is_idp'))
    summary['n_idp_train'] = n_idp_train
    summary['n_idp_val'] = n_idp_val
    summary['n_idp_total'] = n_idp_train + n_idp_val
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  IDP entries: {n_idp_train} train + {n_idp_val} val = {n_idp_train + n_idp_val} total')
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser(description='Merge confidence datasets')
    p.add_argument('--version', choices=['v4', 'v5', 'v6', 'v7', 'v8', 'v9'], help='Merge to specific version')
    p.add_argument('--datasets', nargs='+', help='Custom: LMDB directories to merge')
    p.add_argument('--output_dir', help='Custom: output directory')
    p.add_argument('--disease_dir', help='v5 only: disease v2 dataset directory')
    p.add_argument('--brain_dir', help='v6/v8: brain disease dataset directory')
    p.add_argument('--liver_dir', help='v7/v8: liver disease dataset directory')
    p.add_argument('--v8_expansion_dir', help='v8 only: brain expansion dataset directory')
    p.add_argument('--idp_dir', help='v9 only: IDP v3 dataset directory')
    p.add_argument('--iptm_threshold', type=float, default=0.6, help='v8: ipTM quality threshold')
    p.add_argument('--train_split', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    if args.version == 'v4':
        merge_v4()
    elif args.version == 'v5':
        merge_v5(args.disease_dir)
    elif args.version == 'v6':
        merge_v6(args.brain_dir)
    elif args.version == 'v7':
        merge_v7(args.liver_dir)
    elif args.version == 'v8':
        merge_v8(args.brain_dir, args.v8_expansion_dir, args.liver_dir, args.iptm_threshold)
    elif args.version == 'v9':
        merge_v9(args.idp_dir)
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
