#!/usr/bin/env python3
"""Create an antigen-centered Phase3 train/validation split."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.utils.homology_split import (  # noqa: E402
    annotate_homology_clusters,
    grouped_train_val_split,
)
from scripts.preprocess_sabdab_phase3 import build_lmdb  # noqa: E402

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'


def sequence_from_chain(chain, mask=None):
    if chain is None:
        return ''
    amino_acids = chain.get('aa')
    if amino_acids is None:
        return ''
    if mask is not None:
        amino_acids = amino_acids[mask]
    return ''.join(
        ALPHABET[int(value)] if int(value) < len(ALPHABET) else 'X'
        for value in amino_acids)


def normalize_sequences(record):
    heavy = record.get('heavy')
    light = record.get('light')
    antigen = record.get('antigen')
    record['vh_sequence'] = record.get('vh_sequence') or sequence_from_chain(heavy)
    record['vl_sequence'] = record.get('vl_sequence') or sequence_from_chain(light)
    record['antigen_sequence'] = (
        record.get('antigen_sequence') or sequence_from_chain(antigen))
    if not record.get('cdr_h3_sequence') and heavy is not None:
        cdr_flag = heavy.get('cdr_flag')
        if cdr_flag is not None:
            record['cdr_h3_sequence'] = sequence_from_chain(heavy, cdr_flag == 3)
    return record


def read_lmdb(path):
    ids_path = path + '-ids'
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=False)
    records = []
    with env.begin() as txn:
        if os.path.exists(ids_path):
            with open(ids_path, 'rb') as handle:
                keys = [str(value).encode() for value in pickle.load(handle)]
        else:
            keys = [key for key, _ in txn.cursor() if not key.startswith(b'__')]
        for key in keys:
            value = txn.get(key)
            if value is not None:
                records.append(pickle.loads(value))
    env.close()
    return records


def usable(record):
    antibody = record.get('heavy') is not None or record.get('light') is not None
    antigen = record.get('antigen') is not None and bool(record.get('antigen_sequence'))
    return antibody and antigen and bool(record.get('cdr_h3_sequence'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-source', default='data/phase3_expanded/train.lmdb')
    parser.add_argument('--val-source', default='data/phase3_expanded/val.lmdb')
    parser.add_argument('--output-dir', default='data/phase3_v5_1_pair_clustered')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2051)
    parser.add_argument('--mmseqs', default='mmseqs')
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')

    records = read_lmdb(args.train_source) + read_lmdb(args.val_source)
    deduplicated = {}
    for record in records:
        normalize_sequences(record)
        if not usable(record):
            continue
        record_id = str(record.get('id') or record.get('pdb_id')).casefold()
        record['id'] = record_id
        record['pdb_id'] = str(record.get('pdb_id') or record_id).casefold()
        record['pdb_cluster_id'] = f"pdb:{record['pdb_id']}"
        record['cdr_h3_exact_id'] = f"cdr_h3:{record['cdr_h3_sequence'].upper()}"
        deduplicated.setdefault(record_id, record)
    records = list(deduplicated.values())
    if len(records) < 500:
        raise RuntimeError(f'Too few usable antibody-antigen complexes: {len(records)}')

    annotate_homology_clusters(
        records,
        {'antigen_sequence': 0.3},
        executable=args.mmseqs,
    )
    missing_clusters = [
        record['id'] for record in records
        if not record.get('antigen_sequence_cluster_id')
    ]
    if missing_clusters:
        raise RuntimeError(
            f'MMseqs2 did not cluster {len(missing_clusters)} antigens')
    cluster_fields = [
        'pdb_cluster_id',
        'cdr_h3_exact_id',
        'antigen_sequence_cluster_id',
    ]
    train, val = grouped_train_val_split(
        records, args.val_ratio, args.seed, cluster_fields)
    if min(len(train), len(val)) < round(0.05 * len(records)):
        raise RuntimeError(
            f'Antigen-centered split is too imbalanced: {len(train)}/{len(val)}')

    hard_fields = ['pdb_id', 'cdr_h3_sequence', 'antigen_sequence_cluster_id']
    for field in hard_fields:
        train_values = {record[field] for record in train}
        val_values = {record[field] for record in val}
        overlap = train_values & val_values
        if overlap:
            raise RuntimeError(f'Hard split overlap for {field}: {len(overlap)}')

    output.mkdir(parents=True)
    build_lmdb(train, str(output / 'train.lmdb'))
    build_lmdb(val, str(output / 'val.lmdb'))
    metadata = {
        'version': 'v5.1',
        'sources': [args.train_source, args.val_source],
        'n_input': len(records),
        'n_train': len(train),
        'n_val': len(val),
        'split_strategy': 'antigen_centered_v1',
        'hard_isolation_passed': True,
        'hard_fields': hard_fields,
        'audit_only_fields': [
            'vh_sequence', 'vl_sequence', 'cdr_h3_sequence_homology'],
        'cluster_fields': cluster_fields,
        'thresholds': {
            'antigen_sequence': 0.3,
            'coverage': 0.8,
        },
    }
    with open(output / 'dataset_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
