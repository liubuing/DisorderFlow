#!/usr/bin/env python3
"""Create homology-disjoint train/val LMDBs from a completed v5 build."""

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import lmdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.utils.homology_split import (  # noqa: E402
    annotate_homology_clusters,
    grouped_train_val_split,
)


def _read_metadata(env):
    metadata = []
    with env.begin() as txn:
        raw_length = txn.get(b'__len__')
        length = pickle.loads(raw_length) if raw_length else 0
        for index in range(length):
            value = txn.get(f'{index:08d}'.encode())
            if value is None:
                raise RuntimeError(f'Missing source entry {index:08d}')
            entry = pickle.loads(value)
            sequence = str(entry.get('sequence', '')).upper()
            if not sequence:
                raise RuntimeError(f'Entry {index:08d} has no sequence')
            metadata.append({
                '_source_index': index,
                'pdb_id': str(entry.get('pdb_id', index)).casefold(),
                'sequence': sequence,
            })
    return metadata


def _copy_split(source_env, entries, destination):
    destination.mkdir(parents=True, exist_ok=False)
    output = lmdb.open(str(destination), map_size=source_env.info()['map_size'])
    with source_env.begin() as source_txn:
        for start in range(0, len(entries), 100):
            chunk = entries[start:start + 100]
            with output.begin(write=True) as output_txn:
                for offset, entry in enumerate(chunk, start=start):
                    value = source_txn.get(f'{entry["_source_index"]:08d}'.encode())
                    output_txn.put(f'{offset:08d}'.encode(), value)
        with output.begin(write=True) as output_txn:
            output_txn.put(b'__len__', pickle.dumps(len(entries)))
    output.sync()
    output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=(
        'data/confidence_conformation_v5/confidence_train.lmdb'))
    parser.add_argument('--output-dir', default='data/confidence_conformation_v5_clustered')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--expected-count', type=int, default=1289)
    parser.add_argument('--mmseqs', default='mmseqs')
    parser.add_argument(
        '--calibration-report',
        default='data/confidence_conformation_v5/rmsf_calibration_1289.json')
    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')

    source_env = lmdb.open(str(source_path), readonly=True, lock=False,
                           readahead=False, max_readers=1)
    metadata = _read_metadata(source_env)
    if args.expected_count and len(metadata) != args.expected_count:
        source_env.close()
        raise RuntimeError(
            f'Build is incomplete: found {len(metadata)}, expected {args.expected_count}')

    annotate_homology_clusters(
        metadata, {'sequence': 0.3}, coverage=0.8, executable=args.mmseqs)
    calibration_path = Path(args.calibration_report)
    if not calibration_path.is_file():
        source_env.close()
        raise FileNotFoundError(
            f'Experimental RMSF calibration report is required: {calibration_path}')
    with open(calibration_path, encoding='utf-8') as handle:
        calibration = json.load(handle)
    if calibration.get('source_count') != len(metadata):
        source_env.close()
        raise RuntimeError(
            'Calibration report does not match completed source LMDB: '
            f'{calibration.get("source_count")} != {len(metadata)}')
    calibration_indices = set(calibration.get('calibration_source_indices', []))
    if not calibration_indices:
        source_env.close()
        raise RuntimeError('Calibration report contains no holdout source indices')
    calibration_clusters = {
        entry['sequence_cluster_id'] for entry in metadata
        if entry['_source_index'] in calibration_indices
    }
    holdout = [
        entry for entry in metadata
        if entry['sequence_cluster_id'] in calibration_clusters
    ]
    development = [
        entry for entry in metadata
        if entry['sequence_cluster_id'] not in calibration_clusters
    ]
    train, val = grouped_train_val_split(
        development, val_ratio=args.val_ratio, seed=args.seed,
        cluster_fields=['sequence_cluster_id'])

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        _copy_split(source_env, train, output_dir / 'confidence_train.lmdb')
        _copy_split(source_env, val, output_dir / 'confidence_val.lmdb')
        _copy_split(source_env, holdout, output_dir / 'confidence_calibration.lmdb')
        train_clusters = {entry['sequence_cluster_id'] for entry in train}
        val_clusters = {entry['sequence_cluster_id'] for entry in val}
        if not train_clusters.isdisjoint(val_clusters):
            raise RuntimeError('Homology clusters overlap after split')
        manifest = {
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source': str(source_path),
            'split_strategy': 'MMseqs2 full-sequence clusters',
            'min_identity': 0.3,
            'coverage': 0.8,
            'seed': args.seed,
            'n_total': len(metadata),
            'n_train': len(train),
            'n_val': len(val),
            'n_calibration_holdout': len(holdout),
            'n_train_clusters': len(train_clusters),
            'n_val_clusters': len(val_clusters),
            'train_source_indices': [entry['_source_index'] for entry in train],
            'val_source_indices': [entry['_source_index'] for entry in val],
            'calibration_source_indices': [entry['_source_index'] for entry in holdout],
            'calibration_sequence_clusters': sorted(calibration_clusters),
            'calibration_report': str(calibration_path),
        }
        with open(output_dir / 'split_manifest.json', 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)
    except Exception:
        source_env.close()
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    source_env.close()
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
