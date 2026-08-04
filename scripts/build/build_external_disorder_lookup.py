#!/usr/bin/env python3
"""Export antigen FASTA and build a split-scoped external disorder lookup."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np


AA = 'ACDEFGHIKLMNPQRSTVWY'


def sha256(path: Path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def ids_digest(ids):
    return hashlib.sha256('\n'.join(map(str, ids)).encode('utf-8')).hexdigest()


def load_sequences(lmdb_path: Path):
    ids_path = Path(f"{lmdb_path}-ids")
    ids = pickle.loads(ids_path.read_bytes())
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    sequences = {}
    pdb_ids = {}
    with env.begin() as transaction:
        for sample_id in ids:
            raw = transaction.get(str(sample_id).encode())
            if raw is None:
                raise RuntimeError(f"LMDB key missing for {sample_id}")
            record = pickle.loads(raw)
            antigen = record.get('antigen')
            if antigen is None or antigen.get('aa') is None:
                raise RuntimeError(f"Antigen sequence missing for {sample_id}")
            indices = antigen['aa'].tolist()
            if any(int(index) < 0 or int(index) >= 20 for index in indices):
                raise RuntimeError(f"Non-standard antigen residue in {sample_id}")
            sequences[str(sample_id)] = ''.join(AA[int(index)] for index in indices)
            pdb_ids[str(sample_id)] = str(record.get('pdb_id', sample_id))
    env.close()
    return ids, ids_path, sequences, pdb_ids


def write_fasta(path: Path, ids, sequences):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='ascii', newline='\n') as handle:
        for sample_id in ids:
            key = str(sample_id)
            handle.write(f'>{key}\n{sequences[key]}\n')


def parse_scores(path: Path, expected_sequences=None):
    profiles = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if len(row) < 2:
                raise RuntimeError(f"Malformed predictor row in {path}")
            sample_id = row[0]
            if sample_id in profiles:
                raise RuntimeError(f"Duplicate predictor ID: {sample_id}")
            score_start = 1
            if row[1].strip().isalpha():
                sequence = row[1].strip().upper()
                if expected_sequences is not None and expected_sequences.get(sample_id) != sequence:
                    raise RuntimeError(f"Predictor sequence mismatch for {sample_id}")
                score_start = 2
            profiles[sample_id] = np.asarray(
                [float(value) for value in row[score_start:]], dtype=np.float32)
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmdb', required=True, type=Path)
    parser.add_argument('--split', required=True,
                        choices=(
                            'train', 'dev', 'legacy_val', 'external_blind',
                            'adaptation', 'project_dev', 'project_final'))
    parser.add_argument('--fasta', required=True, type=Path)
    parser.add_argument('--scores-csv', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--predictor', default='metapredict')
    parser.add_argument('--predictor-version', default='3.0.2')
    parser.add_argument('--model-version', default='V3')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    ids, ids_path, sequences, pdb_ids = load_sequences(args.lmdb)
    write_fasta(args.fasta, ids, sequences)
    export = {
        'n_ids': len(ids),
        'ids_sha256': ids_digest(ids),
        'fasta': str(args.fasta),
        'fasta_sha256': sha256(args.fasta),
    }
    if args.scores_csv is None:
        print(json.dumps(export, indent=2))
        return
    if args.output is None:
        parser.error('--output is required with --scores-csv')
    profiles = parse_scores(args.scores_csv, sequences)
    expected = {str(sample_id) for sample_id in ids}
    if set(profiles) != expected:
        raise RuntimeError(
            f"Predictor ID mismatch: missing={len(expected - set(profiles))}, "
            f"extra={len(set(profiles) - expected)}")

    lengths = []
    means = []
    for sample_id, sequence in sequences.items():
        profile = profiles[sample_id]
        if len(profile) != len(sequence):
            raise RuntimeError(
                f"Profile length mismatch for {sample_id}: {len(profile)} != {len(sequence)}")
        if not np.isfinite(profile).all() or np.any(profile < 0) or np.any(profile > 1):
            raise RuntimeError(f"Invalid profile values for {sample_id}")
        lengths.append(len(profile))
        means.append(float(profile.mean()))

    source = f'{args.predictor}_{args.predictor_version}_{args.model_version}_normalized_{args.device}'
    artifact = {
        'schema_version': 2,
        'split': args.split,
        'lmdb_path': str(args.lmdb.resolve()),
        'ids_path': str(ids_path.resolve()),
        'ids_sha256': ids_digest(ids),
        'source_contract': {
            'primary_source': source,
            'predictor': args.predictor,
            'package_version': args.predictor_version,
            'model_version': args.model_version,
            'device': args.device,
            'normalized': True,
            'per_residue': True,
            'experimental_ground_truth': False,
            'fallback_used': False,
        },
        'input_fasta_sha256': sha256(args.fasta),
        'predictor_output_sha256': sha256(args.scores_csv),
        'profiles': profiles,
        'provenance': {
            sample_id: {
                'source': source,
                'pdb_id': pdb_ids[sample_id],
                'length': len(profile),
            }
            for sample_id, profile in profiles.items()
        },
        'stats': {
            'n_ids': len(ids),
            'n_profiles': len(profiles),
            'length_min': min(lengths),
            'length_max': max(lengths),
            'mean_profile_quantiles': {
                str(q): float(np.quantile(means, q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
            'residue_fraction_at_least_0_5': float(
                (np.concatenate(list(profiles.values())) >= 0.5).mean()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    audit = {key: value for key, value in artifact.items() if key not in ('profiles', 'provenance')}
    args.output.with_suffix('.audit.json').write_text(
        json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), **artifact['stats']}, indent=2))


if __name__ == '__main__':
    main()
