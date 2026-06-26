#!/usr/bin/env python3
"""Split the V14 design-variant LMDB into scaffold-disjoint train/val groups.

PROBLEM (diagnosed 2026-06-26): build_design_variant_dataset.py only ever wrote
`train.lmdb` — there is no V14 val. So configs/train/bfn_v14_grouped_conf_xpu.yml
pointed val at the legacy V11 confidence_val.lmdb, which contains single
independent PDBs with NO scaffold grouping. On that val the grouped losses
(conf_variance / conf_ranking / grouped_margin / conf_anticollapse) are all
non-computable (logged as 0.0000 or saturated at the anti-collapse ceiling),
so best.pt selection was blind to the very property we trained for.

FIX: split the existing train.lmdb BY SCAFFOLD (whole groups kept intact, no
scaffold appears in both splits) into:
    data/confidence_design_variants_v14/train_grouped.lmdb
    data/confidence_design_variants_v14/val_grouped.lmdb
Keeping whole scaffold groups in val means within-scaffold variant pairs exist
on val → grouped losses compute → best.pt can reward design-discrimination.

Usage:
    python build_v14_grouped_val.py                  # 15% scaffolds -> val
    python build_v14_grouped_val.py --val_frac 0.20
    python build_v14_grouped_val.py --seed 2031
"""
import os
import sys
import pickle
import random
import argparse
import collections

import lmdb

SRC = 'data/confidence_design_variants_v14/train.lmdb'
DST_DIR = 'data/confidence_design_variants_v14'


def read_all(src):
    """Return list of (key_int, entry) from a V14 LMDB."""
    env = lmdb.open(src, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = []
        for k, _ in txn.cursor():
            if k == b'__len__':
                continue
            keys.append(int(k.decode()))
        keys.sort()
        entries = []
        for ki in keys:
            entries.append((ki, pickle.loads(txn.get(f'{ki:08d}'.encode()))))
    env.close()
    return entries


def write_split(path, entries):
    env = lmdb.open(path, map_size=1 << 35)
    key_idx = 0
    with env.begin(write=True) as txn:
        for _, entry in entries:
            txn.put(f'{key_idx:08d}'.encode(), pickle.dumps(entry))
            key_idx += 1
        txn.put(b'__len__', pickle.dumps(key_idx))
    env.close()
    return key_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--dst_dir', default=DST_DIR)
    ap.add_argument('--val_frac', type=float, default=0.15,
                    help='fraction of SCAFFOLDS (not entries) to hold out for val')
    ap.add_argument('--seed', type=int, default=2031)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f'ERROR: source not found: {args.src}', file=sys.stderr)
        sys.exit(1)

    print(f'Reading {args.src} ...')
    entries = read_all(args.src)
    print(f'  {len(entries)} entries')

    # Group entries by scaffold_id, preserving the per-scaffold variant count.
    by_scaffold = collections.OrderedDict()
    for ki, entry in entries:
        sid = entry.get('scaffold_id', ki)
        by_scaffold.setdefault(sid, []).append(entry)

    scaffolds = list(by_scaffold.keys())
    n_variants_per = {s: len(by_scaffold[s]) for s in scaffolds}
    print(f'  {len(scaffolds)} scaffolds, variants/scaffold: '
          f'min={min(n_variants_per.values())} max={max(n_variants_per.values())} '
          f'mean={sum(n_variants_per.values())/len(scaffolds):.1f}')

    rng = random.Random(args.seed)
    rng.shuffle(scaffolds)
    n_val_scaff = max(1, int(round(len(scaffolds) * args.val_frac)))
    val_scaff = set(scaffolds[:n_val_scaff])
    train_scaff = [s for s in scaffolds if s not in val_scaff]

    train_entries, val_entries = [], []
    for s in train_scaff:
        train_entries.extend((i, e) for i, e in enumerate(by_scaffold[s]))
    # re-key val independently from 0
    val_entries_keys = []
    for s in sorted(val_scaff, key=lambda x: scaffolds.index(x)):
        for e in by_scaffold[s]:
            val_entries_keys.append((len(val_entries_keys), e))

    os.makedirs(args.dst_dir, exist_ok=True)
    train_path = os.path.join(args.dst_dir, 'train_grouped.lmdb')
    val_path = os.path.join(args.dst_dir, 'val_grouped.lmdb')

    n_train = write_split(train_path, [(i, e) for i, e in train_entries])
    n_val = write_split(val_path, val_entries_keys)

    print(f'\nWrote {n_train} train -> {train_path}')
    print(f'Wrote {n_val} val   -> {val_path}')
    print(f'  val scaffolds: {len(val_scaff)}/{len(scaffolds)} ({100*len(val_scaff)/len(scaffolds):.1f}%)')
    print(f'  val entries:   {n_val}/{len(entries)} ({100*n_val/len(entries):.1f}%)')
    print(f'\nUpdate config val.db_path -> {val_path}')
    print(f'         (optional) config train.db_path -> {train_path}')


if __name__ == '__main__':
    main()
