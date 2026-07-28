#!/usr/bin/env python3
"""Audit LMDB train/validation splits for identity and homology leakage."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.utils.homology_split import mmseqs_cluster


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
SEQUENCE_FIELDS = ('sequence', 'vh_sequence', 'vl_sequence',
                   'cdr_h3_sequence', 'antigen_sequence')


def _tensor_sequence(values):
    if values is None:
        return ''
    return ''.join(ALPHABET[int(value)] if 0 <= int(value) < 20 else 'X'
                   for value in values)


def _normalize_entry(entry):
    normalized = {
        'pdb_id': str(entry.get('pdb_id', entry.get('id', ''))).casefold(),
        'sequence': str(entry.get('sequence', '')).upper(),
    }
    for chain_key, field in (('heavy', 'vh_sequence'), ('light', 'vl_sequence'),
                             ('antigen', 'antigen_sequence')):
        chain = entry.get(chain_key)
        normalized[field] = str(entry.get(field, '')).upper()
        if not normalized[field] and chain is not None:
            normalized[field] = _tensor_sequence(chain.get('aa'))
    normalized['cdr_h3_sequence'] = str(entry.get('cdr_h3_sequence', '')).upper()
    heavy = entry.get('heavy')
    if not normalized['cdr_h3_sequence'] and heavy is not None:
        cdr_flag = heavy.get('cdr_flag')
        if cdr_flag is not None:
            normalized['cdr_h3_sequence'] = _tensor_sequence(heavy['aa'][cdr_flag == 3])

    plddt = entry.get('af2_plddt')
    normalized['plddt_mean'] = (float(plddt.float().mean())
                                if hasattr(plddt, 'float') and plddt.numel() else None)
    normalized['has_disorder_label'] = bool(
        'disorder_label' in entry or 'disorder_mask' in entry
        or (isinstance(entry.get('batch'), dict)
            and ('disorder_label' in entry['batch'] or 'disorder_mask' in entry['batch'])))
    return normalized


def load_lmdb(path):
    path = Path(path)
    subdir = path.is_dir()
    env = lmdb.open(str(path), subdir=subdir, readonly=True, lock=False,
                    readahead=False, max_readers=1)
    entries = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            if key.startswith(b'__'):
                continue
            try:
                entries.append(_normalize_entry(pickle.loads(value)))
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
                entries.append({'load_error': str(exc), 'pdb_id': ''})
    env.close()
    return entries


def _overlap(train, val, field):
    train_values = {entry.get(field) for entry in train if entry.get(field)}
    val_values = {entry.get(field) for entry in val if entry.get(field)}
    overlap = train_values & val_values
    return {
        'train_unique': len(train_values),
        'val_unique': len(val_values),
        'overlap_count': len(overlap),
        'overlap_examples': sorted(overlap)[:10],
    }


def _homology_overlap(train, val, field, min_identity, executable):
    combined = ([{'sequence': entry.get(field, '')} for entry in train] +
                [{'sequence': entry.get(field, '')} for entry in val])
    mapping = mmseqs_cluster(combined, 'sequence', min_identity=min_identity,
                             coverage=0.8, executable=executable)
    boundary = len(train)
    train_clusters = {cluster for index, cluster in mapping.items() if index < boundary}
    val_clusters = {cluster for index, cluster in mapping.items() if index >= boundary}
    overlap = train_clusters & val_clusters
    return {
        'min_identity': min_identity,
        'coverage': 0.8,
        'train_clusters': len(train_clusters),
        'val_clusters': len(val_clusters),
        'cross_split_clusters': len(overlap),
        'cluster_examples': sorted(overlap)[:10],
    }


def audit_dataset(name, train_path, val_path, executable='mmseqs'):
    started = time.time()
    train = load_lmdb(train_path)
    val = load_lmdb(val_path)
    exact = {'pdb_id': _overlap(train, val, 'pdb_id')}
    homology = {}
    thresholds = {
        'sequence': 0.3,
        'vh_sequence': 0.5,
        'vl_sequence': 0.5,
        'cdr_h3_sequence': 0.5,
        'antigen_sequence': 0.3,
    }
    for field in SEQUENCE_FIELDS:
        exact[field] = _overlap(train, val, field)
        if exact[field]['train_unique'] and exact[field]['val_unique']:
            homology[field] = _homology_overlap(
                train, val, field, thresholds[field], executable)

    plddt = [entry['plddt_mean'] for entry in train + val
             if entry.get('plddt_mean') is not None]
    return {
        'name': name,
        'train_path': str(train_path),
        'val_path': str(val_path),
        'n_train': len(train),
        'n_val': len(val),
        'load_errors': sum('load_error' in entry for entry in train + val),
        'entries_with_disorder_label': sum(
            entry.get('has_disorder_label', False) for entry in train + val),
        'plddt': ({
            'count': len(plddt),
            'min': float(np.min(plddt)),
            'max': float(np.max(plddt)),
            'mean': float(np.mean(plddt)),
            'inferred_scale': '0-1' if max(plddt, default=0) <= 1.0 else '0-100',
        } if plddt else {'count': 0}),
        'exact_overlap': exact,
        'homology_overlap': homology,
        'elapsed_seconds': round(time.time() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='append', nargs=3,
                        metavar=('NAME', 'TRAIN_LMDB', 'VAL_LMDB'), required=True)
    parser.add_argument('--mmseqs', default='mmseqs')
    parser.add_argument('--output', default='dataset_split_audit.json')
    args = parser.parse_args()

    report = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'datasets': [audit_dataset(*dataset, executable=args.mmseqs)
                     for dataset in args.dataset],
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
