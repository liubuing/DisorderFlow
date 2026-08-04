#!/usr/bin/env python3
"""Create an internal routing train/dev split without touching legacy validation."""

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

from disorderflow.utils.homology_split import grouped_train_val_split  # noqa: E402
from scripts.preprocess_sabdab_phase3 import build_lmdb  # noqa: E402


def read_records(path: Path):
    ids = pickle.loads(Path(f"{path}-ids").read_bytes())
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    records = []
    with env.begin() as transaction:
        for sample_id in ids:
            raw = transaction.get(str(sample_id).encode())
            if raw is None:
                raise RuntimeError(f"LMDB key missing for {sample_id}")
            records.append(pickle.loads(raw))
    env.close()
    return ids, records


def digest(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('data/phase3_v5_1_pair_clustered/train.lmdb'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/phase3_v5_1_routing_train_dev'))
    parser.add_argument('--dev-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2091)
    parser.add_argument('--base-lookup', type=Path,
                        help='Prior-stage lookup; all represented groups stay in train')
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing split: {args.output_dir}")
    source_ids, records = read_records(args.source)
    required = ('pdb_cluster_id', 'cdr_h3_exact_id', 'antigen_sequence_cluster_id')
    missing = {
        field: sum(not record.get(field) for record in records)
        for field in required
    }
    if any(missing.values()):
        raise RuntimeError(f"Source records lack required cluster annotations: {missing}")

    prior_ids = set()
    if args.base_lookup:
        prior_artifact = pickle.loads(args.base_lookup.read_bytes())
        prior_profiles = prior_artifact.get('profiles', prior_artifact)
        prior_ids = {str(value).casefold() for value in prior_profiles}
    train, dev = grouped_train_val_split(
        records, args.dev_ratio, args.seed, cluster_fields=required,
        fixed_train_ids=prior_ids)
    minimum_dev = 50 if args.base_lookup else round(0.05 * len(records))
    if len(dev) < minimum_dev:
        raise RuntimeError(
            f"Too few eligible internal-dev records: {len(dev)} < {minimum_dev}")

    isolation = {}
    for field in ('pdb_id', 'cdr_h3_exact_id', 'antigen_sequence_cluster_id'):
        train_values = {str(record[field]).casefold() for record in train}
        dev_values = {str(record[field]).casefold() for record in dev}
        overlap = train_values & dev_values
        isolation[field] = {'overlap_count': len(overlap)}
        if overlap:
            raise RuntimeError(f"Internal split overlap for {field}: {len(overlap)}")

    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / 'train.lmdb'
    dev_path = args.output_dir / 'dev.lmdb'
    build_lmdb(train, str(train_path))
    build_lmdb(dev, str(dev_path))
    train_ids = pickle.loads(Path(f"{train_path}-ids").read_bytes())
    dev_ids = pickle.loads(Path(f"{dev_path}-ids").read_bytes())
    report = {
        'schema_version': 1,
        'purpose': 'routing model selection; legacy validation remains excluded',
        'source': str(args.source.resolve()),
        'source_ids_sha256': digest(source_ids),
        'seed': args.seed,
        'dev_ratio_requested': args.dev_ratio,
        'dev_ratio_observed': len(dev) / len(records),
        'n_source': len(records),
        'n_train': len(train),
        'n_dev': len(dev),
        'train_ids_sha256': digest(train_ids),
        'dev_ids_sha256': digest(dev_ids),
        'cluster_fields': list(required),
        'isolation': isolation,
        'legacy_validation_used': False,
        'prior_stage_lookup': str(args.base_lookup.resolve()) if args.base_lookup else None,
        'n_prior_stage_ids': len(prior_ids),
        'dev_prior_stage_id_overlap': len(
            {str(record['id']).casefold() for record in dev} & prior_ids),
    }
    (args.output_dir / 'split_audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
