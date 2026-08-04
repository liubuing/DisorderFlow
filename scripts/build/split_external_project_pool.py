#!/usr/bin/env python3
"""Cluster and split a reference-disjoint external project pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import lmdb


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from disorderflow.utils.homology_split import (  # noqa: E402
    annotate_homology_clusters,
    grouped_train_val_split,
)
from scripts.preprocess_sabdab_phase3 import build_lmdb  # noqa: E402


def read_records(path: Path):
    ids = pickle.loads(Path(f'{path}-ids').read_bytes())
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    records = []
    with env.begin() as transaction:
        for sample_id in ids:
            raw = transaction.get(str(sample_id).encode())
            if raw is None:
                raise RuntimeError(f'LMDB key missing for {sample_id}')
            records.append(pickle.loads(raw))
    env.close()
    return records


def digest_ids(records):
    values = [str(record['id']) for record in records]
    return hashlib.sha256('\n'.join(values).encode('utf-8')).hexdigest()


def assert_isolated(left, right, fields):
    result = {}
    for field in fields:
        overlap = ({str(item[field]).casefold() for item in left}
                   & {str(item[field]).casefold() for item in right})
        result[field] = len(overlap)
        if overlap:
            raise RuntimeError(f'Split overlap for {field}: {len(overlap)}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--mmseqs', default='mmseqs')
    parser.add_argument('--final-ratio', type=float, default=0.2)
    parser.add_argument('--dev-ratio-of-remainder', type=float, default=0.2)
    parser.add_argument('--final-seed', type=int, default=2111)
    parser.add_argument('--dev-seed', type=int, default=2113)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite split: {args.output_dir}')

    source_records = read_records(args.source)
    records = [
        record for record in source_records
        if record.get('antigen_sequence') and record.get('cdr_h3_sequence')
    ]
    annotate_homology_clusters(
        records,
        {'antigen_sequence': 0.3, 'cdr_h3_sequence': 0.5},
        executable=args.mmseqs,
    )
    cluster_fields = ('antigen_sequence_cluster_id', 'cdr_h3_sequence_cluster_id')
    if any(not record.get(field) for record in records for field in cluster_fields):
        raise RuntimeError('MMseqs2 left records without required cluster IDs')

    remainder, final = grouped_train_val_split(
        records, args.final_ratio, args.final_seed, cluster_fields=cluster_fields)
    adaptation, dev = grouped_train_val_split(
        remainder, args.dev_ratio_of_remainder, args.dev_seed,
        cluster_fields=cluster_fields)
    if min(map(len, (adaptation, dev, final))) < 50:
        raise RuntimeError(
            f'Split too small: {len(adaptation)}/{len(dev)}/{len(final)}')

    hard_fields = ('pdb_id', *cluster_fields)
    isolation = {
        'adaptation_vs_dev': assert_isolated(adaptation, dev, hard_fields),
        'adaptation_vs_final': assert_isolated(adaptation, final, hard_fields),
        'dev_vs_final': assert_isolated(dev, final, hard_fields),
    }
    args.output_dir.mkdir(parents=True)
    for name, split in (
            ('adaptation', adaptation), ('dev', dev), ('final', final)):
        build_lmdb(split, str(args.output_dir / f'{name}.lmdb'))
    report = {
        'schema_version': 1,
        'source': str(args.source.resolve()),
        'n_source': len(source_records),
        'n_excluded_missing_sequence': len(source_records) - len(records),
        'n_adaptation': len(adaptation),
        'n_dev': len(dev),
        'n_final': len(final),
        'final_seed': args.final_seed,
        'dev_seed': args.dev_seed,
        'thresholds': {
            'antigen_sequence': {'identity': 0.3, 'coverage': 0.8},
            'cdr_h3_sequence': {'identity': 0.5, 'coverage': 0.8},
        },
        'isolation': isolation,
        'ids_sha256': {
            'adaptation': digest_ids(adaptation),
            'dev': digest_ids(dev),
            'final': digest_ids(final),
        },
        'final_status': 'sealed; no model forward before protocol freeze',
    }
    (args.output_dir / 'split_audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
