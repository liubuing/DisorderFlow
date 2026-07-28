#!/usr/bin/env python
"""
Merge SAbDab Phase 3 data with existing Phase 3 data.

Combines the original antibody_complexes dataset with the SAbDab-preprocessed
complexes, deduplicates by PDB code, and produces a unified train/val LMDB.

Usage:
    python scripts/merge_phase3_datasets.py \
        --inputs data/phase3_processed data/sabdab_phase3_processed \
        --output data/phase3_expanded
"""

import os
import sys
import json
import pickle
import logging
import argparse

import lmdb
from tqdm import tqdm
from disorderflow.utils.homology_split import (
    annotate_homology_clusters,
    grouped_train_val_split,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_lmdb_entries(lmdb_path):
    """Load all entries from an LMDB database."""
    ids_path = lmdb_path + '-ids'
    entries = {}
    if not os.path.exists(lmdb_path) or not os.path.exists(ids_path):
        logger.warning(f"LMDB not found: {lmdb_path}")
        return entries

    with open(ids_path, 'rb') as f:
        ids = pickle.load(f)

    env = lmdb.open(lmdb_path, readonly=True, subdir=False)
    with env.begin() as txn:
        for id_ in ids:
            data = txn.get(id_.encode('utf-8'))
            if data is not None:
                entries[id_] = pickle.loads(data)
    env.close()
    logger.info(f"Loaded {len(entries)} entries from {lmdb_path}")
    return entries


def build_lmdb(data_list, lmdb_path, map_size=16 * 1024 * 1024 * 1024):
    """Build LMDB from list of processed dicts."""
    db_conn = lmdb.open(lmdb_path, map_size=map_size, create=True, subdir=False, readonly=False)
    ids = []
    with db_conn.begin(write=True, buffers=True) as txn:
        for data in data_list:
            ids.append(data['id'])
            txn.put(data['id'].encode('utf-8'), pickle.dumps(data))
    with open(lmdb_path + '-ids', 'wb') as f:
        pickle.dump(ids, f)
    db_conn.close()
    logger.info(f"Wrote {len(ids)} entries to {lmdb_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge Phase 3 datasets')
    parser.add_argument('--inputs', nargs='+', required=True, help='Input LMDB directories')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--mmseqs', default='mmseqs', help='MMseqs2 executable')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load all datasets
    all_entries = {}
    for input_dir in args.inputs:
        for split in ['train', 'val']:
            lmdb_path = os.path.join(input_dir, f'{split}.lmdb')
            if os.path.exists(lmdb_path):
                entries = load_lmdb_entries(lmdb_path)
                for id_, data in entries.items():
                    normalized_id = str(id_).casefold()
                    if normalized_id not in all_entries:
                        all_entries[normalized_id] = data
                    else:
                        # Favor SAbDab version (more complete metadata)
                        if 'sabdab' in input_dir.lower():
                            all_entries[normalized_id] = data

    logger.info(f"Total unique entries after merge: {len(all_entries)}")

    # Recover sequence metadata from legacy Phase 3 entries before clustering.
    all_list = list(all_entries.values())
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    for entry in all_list:
        entry['pdb_id'] = str(entry.get('pdb_id', entry['id'])).casefold()
        for key, field in [('heavy', 'vh_sequence'), ('light', 'vl_sequence'),
                           ('antigen', 'antigen_sequence')]:
            chain = entry.get(key)
            if chain is not None and not entry.get(field):
                entry[field] = ''.join(
                    alphabet[int(value)] if int(value) < 20 else 'X'
                    for value in chain['aa'])
        heavy = entry.get('heavy')
        if heavy is not None and not entry.get('cdr_h3_sequence'):
            mask = heavy.get('cdr_flag') == 3
            entry['cdr_h3_sequence'] = ''.join(
                alphabet[int(value)] if int(value) < 20 else 'X'
                for value in heavy['aa'][mask])

    annotate_homology_clusters(
        all_list,
        {'vh_sequence': 0.5, 'vl_sequence': 0.5,
         'cdr_h3_sequence': 0.5, 'antigen_sequence': 0.3},
        executable=args.mmseqs,
    )
    cluster_fields = [
        'vh_sequence_cluster_id', 'vl_sequence_cluster_id',
        'cdr_h3_sequence_cluster_id', 'antigen_sequence_cluster_id',
    ]
    train_data, val_data = grouped_train_val_split(
        all_list, args.val_ratio, args.seed, cluster_fields)

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Build
    build_lmdb(train_data, os.path.join(args.output, 'train.lmdb'))
    build_lmdb(val_data, os.path.join(args.output, 'val.lmdb'))

    # Summary
    fab = sum(1 for d in all_list if d.get('heavy') and d.get('light'))
    nano = sum(1 for d in all_list if d.get('heavy') and not d.get('light'))
    with_ag = sum(1 for d in all_list if d.get('antigen') is not None)

    print(f"\n=== Merged Phase 3 Dataset ===")
    print(f"Total: {len(all_list)} complexes")
    print(f"  Fab:       {fab}")
    print(f"  Nanobody:  {nano}")
    print(f"  With Ag:   {with_ag}")
    print(f"  Train:     {len(train_data)}")
    print(f"  Val:       {len(val_data)}")

    meta = {
        'total': len(all_list),
        'train_count': len(train_data),
        'val_count': len(val_data),
        'fab_count': fab,
        'nanobody_count': nano,
        'with_antigen_count': with_ag,
        'sources': args.inputs,
        'train_ids': [d['id'] for d in train_data],
        'val_ids': [d['id'] for d in val_data],
        'split_strategy': 'MMseqs2 connected homology clusters',
        'cluster_fields': cluster_fields,
    }
    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
