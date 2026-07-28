#!/usr/bin/env python3
"""Fail-closed readiness checks for the v5.1 multi-stage training pipeline."""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import lmdb
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.misc import load_config  # noqa: E402


def lmdb_count(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ids_path = path + '-ids'
    if os.path.isfile(ids_path):
        with open(ids_path, 'rb') as handle:
            return len(pickle.load(handle))
    env = lmdb.open(
        path, readonly=True, lock=False, readahead=False,
        subdir=os.path.isdir(path))
    with env.begin() as txn:
        value = txn.get(b'__len__')
        count = pickle.loads(value) if value else sum(
            1 for key, _ in txn.cursor() if not key.startswith(b'__'))
    env.close()
    return count


def raw_manifest_report(path, manifest_path):
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    env = lmdb.open(
        path, readonly=True, lock=False, readahead=False,
        subdir=os.path.isdir(path))
    entries = []
    with env.begin() as txn:
        value = txn.get(b'__len__')
        count = pickle.loads(value) if value else 0
        for index in range(count):
            entry = pickle.loads(txn.get(f'{index:08d}'.encode()))
            entries.append((str(entry['pdb_id']), str(entry['sequence'])))
    env.close()
    payload = '\n'.join(f'{pdb_id}\t{sequence}' for pdb_id, sequence in entries)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    actual = {
        'count': len(entries),
        'unique_pdb_ids': len({pdb_id for pdb_id, _ in entries}),
        'unique_sequences': len({sequence for _, sequence in entries}),
        'canonical_fingerprint': fingerprint,
    }
    expected = {
        key: manifest[key] for key in actual
    }
    if actual != expected:
        raise RuntimeError(
            f'Frozen v5.1 manifest mismatch: actual={actual}, expected={expected}')
    return actual


def audit_is_clean(path):
    with open(path, encoding='utf-8') as handle:
        report = json.load(handle)
    datasets = report.get('datasets', [report])
    failures = []
    for dataset in datasets:
        if dataset.get('load_errors', 0):
            failures.append(f"load_errors={dataset['load_errors']}")
        for field, result in dataset.get('exact_overlap', {}).items():
            if result.get('overlap_count', 0):
                failures.append(f'exact:{field}={result["overlap_count"]}')
        for field, result in dataset.get('homology_overlap', {}).items():
            if result.get('cross_split_clusters', 0):
                failures.append(
                    f'homology:{field}={result["cross_split_clusters"]}')
    return failures


def phase3_audit_is_clean(path, metadata_path):
    with open(metadata_path, encoding='utf-8') as handle:
        metadata = json.load(handle)
    required_hard = {'pdb_id', 'cdr_h3_sequence', 'antigen_sequence_cluster_id'}
    failures = []
    if metadata.get('split_strategy') != 'antigen_centered_v1':
        failures.append('unexpected split strategy')
    if set(metadata.get('hard_fields', [])) != required_hard:
        failures.append('missing hard isolation fields')
    if not metadata.get('hard_isolation_passed'):
        failures.append('hard isolation assertion failed')

    with open(path, encoding='utf-8') as handle:
        report = json.load(handle)
    for dataset in report.get('datasets', [report]):
        if dataset.get('load_errors', 0):
            failures.append(f"load_errors={dataset['load_errors']}")
        exact = dataset.get('exact_overlap', {})
        homology = dataset.get('homology_overlap', {})
        for field in ('pdb_id', 'cdr_h3_sequence', 'antigen_sequence'):
            if field not in exact:
                failures.append(f'missing exact audit:{field}')
            elif exact[field].get('overlap_count', 0):
                failures.append(f'exact:{field}={exact[field]["overlap_count"]}')
        antigen_homology = homology.get('antigen_sequence')
        if antigen_homology is None:
            failures.append('missing homology audit:antigen_sequence')
        elif antigen_homology.get('cross_split_clusters', 0):
            failures.append(
                'homology:antigen_sequence='
                f'{antigen_homology["cross_split_clusters"]}')
        for field in ('vh_sequence', 'vl_sequence', 'cdr_h3_sequence'):
            if field not in exact or field not in homology:
                failures.append(f'missing audit-only field:{field}')
    return failures


def checkpoint_report(path, config_path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model')
    if not isinstance(state, dict) or not state:
        raise ValueError(f'Checkpoint has no model state: {path}')
    nonfinite = [
        key for key, value in state.items()
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise ValueError(f'Checkpoint contains non-finite tensors: {nonfinite[:5]}')
    config, _ = load_config(config_path)
    model = get_model(config.model)
    target = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    loaded_numel = sum(value.numel() for value in compatible.values())
    total_numel = sum(value.numel() for value in target.values())
    coverage = loaded_numel / total_numel
    if coverage < 0.6:
        raise ValueError(f'Checkpoint compatibility coverage is only {coverage:.1%}')
    min_val = checkpoint.get('min_val_loss')
    if isinstance(min_val, torch.Tensor):
        min_val = min_val.item()
    if min_val is not None and not math.isfinite(float(min_val)):
        raise ValueError(f'Checkpoint min_val_loss is not finite: {min_val}')
    return {
        'path': path,
        'compatible_tensors': len(compatible),
        'target_tensors': len(target),
        'parameter_coverage': round(coverage, 6),
        'iteration': checkpoint.get('iteration'),
        'min_val_loss': min_val,
    }


def disorder_supervision_contract(config_path):
    config, _ = load_config(config_path)
    supervision = config.train.get('disorder_supervision')
    model_weights = config.model.get('loss_weight', {})
    train_weights = config.train.get('loss_weights', {})
    continuous_enabled = (
        model_weights.get('disorder', 0) > 0
        or train_weights.get('disorder', 0) > 0)
    rank_enabled = (
        model_weights.get('disorder_rank', 0) > 0
        and train_weights.get('disorder_rank', 0) > 0)
    if supervision == 'rank_only' and rank_enabled and not continuous_enabled:
        return supervision
    if (supervision == 'physical_continuous' and continuous_enabled
            and not rank_enabled):
        return supervision
    raise RuntimeError(
        f'Disorder supervision configuration is inconsistent: {supervision}')


def calibration_report(path, expected_count, supervision='physical_continuous'):
    with open(path, encoding='utf-8') as handle:
        report = json.load(handle)
    if report.get('source_count') != expected_count:
        raise RuntimeError(
            'RMSF calibration source count does not match v5: '
            f'{report.get("source_count")} != {expected_count}')
    summary = report.get('summary', {})
    expected_verdict = {
        'physical_continuous': 'pass_physical_label',
        'rank_only': 'rank_only',
    }.get(supervision)
    if expected_verdict is None:
        raise RuntimeError(f'Unsupported disorder supervision: {supervision}')
    if summary.get('verdict') != expected_verdict:
        raise RuntimeError(
            f'AF2-RMSF calibration does not support {supervision}: '
            f'{summary.get("verdict", "missing verdict")}')
    if summary.get('n_proteins', 0) < 10:
        raise RuntimeError('RMSF calibration has fewer than 10 independent proteins')
    idp_summary = report.get('subgroups', {}).get('annotated_idp', {})
    if idp_summary.get('verdict') != expected_verdict:
        raise RuntimeError(
            'AF2-RMSF calibration did not pass on the annotated-IDP subset: '
            f'{idp_summary.get("verdict", "missing verdict")}')
    return {'all_proteins': summary, 'annotated_idp': idp_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--config', default='configs/train/bfn_v5_1_stage1_disorder.yml')
    parser.add_argument('--raw-only', action='store_true')
    parser.add_argument('--stage1-only', action='store_true')
    parser.add_argument('--expected-count', type=int, required=True)
    parser.add_argument(
        '--raw-manifest',
        default='data/confidence_conformation_v5/frozen_manifest_v5_1.json')
    parser.add_argument('--output')
    parser.add_argument(
        '--calibration-report',
        default='data/confidence_conformation_v5/rmsf_calibration_1289.json')
    parser.add_argument(
        '--stage1-config',
        default='configs/train/bfn_v5_1_stage1_disorder.yml')
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error('--expected-count must be positive')
    if args.raw_only and args.stage1_only:
        parser.error('--raw-only and --stage1-only are mutually exclusive')

    report = {
        'raw_v5': lmdb_count(
            'data/confidence_conformation_v5/confidence_train.lmdb'),
    }
    if report['raw_v5'] != args.expected_count:
        raise RuntimeError(
            f"Raw v5 count mismatch: {report['raw_v5']}/{args.expected_count}")
    report['raw_manifest'] = raw_manifest_report(
        'data/confidence_conformation_v5/confidence_train.lmdb',
        args.raw_manifest)

    if not args.raw_only:
        conformation_paths = [
            'data/confidence_conformation_v5_clustered/confidence_train.lmdb',
            'data/confidence_conformation_v5_clustered/confidence_val.lmdb',
            'data/confidence_conformation_v5_clustered/confidence_calibration.lmdb',
        ]
        report['conformation_counts'] = [lmdb_count(path) for path in conformation_paths]
        if sum(report['conformation_counts']) != args.expected_count:
            raise RuntimeError(
                f'Clustered v5.1 count mismatch: {report["conformation_counts"]}')
        audit_paths = ['data/confidence_conformation_v5_clustered/split_audit.json']
        if not args.stage1_only:
            phase3_paths = [
                'data/phase3_v5_1_pair_clustered/train.lmdb',
                'data/phase3_v5_1_pair_clustered/val.lmdb',
            ]
            report['phase3_counts'] = [lmdb_count(path) for path in phase3_paths]
            if min(report['phase3_counts']) < 20 or sum(report['phase3_counts']) < 500:
                raise RuntimeError(
                    f'Phase3 v5.1 split is unusable: {report["phase3_counts"]}')
            phase3_audit = 'data/phase3_v5_1_pair_clustered/split_audit.json'
            phase3_meta = 'data/phase3_v5_1_pair_clustered/dataset_meta.json'
        for path in audit_paths:
            failures = audit_is_clean(path)
            if failures:
                raise RuntimeError(f'Leakage audit failed for {path}: {failures}')
        if not args.stage1_only:
            failures = phase3_audit_is_clean(phase3_audit, phase3_meta)
            if failures:
                raise RuntimeError(
                    f'Phase3 antigen-centered audit failed: {failures}')
        if not args.stage1_only:
            lookup = Path('data/sabdab_disorder_lookup.pkl')
            if not lookup.is_file():
                raise FileNotFoundError(lookup)
            with open(lookup, 'rb') as handle:
                profiles = pickle.load(handle)
            profile_ids = {
                str(key).casefold() for key, value in profiles.items() if value is not None
            }
            profile_coverage = []
            for path in phase3_paths:
                with open(path + '-ids', 'rb') as handle:
                    ids = [str(value).casefold() for value in pickle.load(handle)]
                profile_coverage.append(sum(value in profile_ids for value in ids))
            if min(profile_coverage) < 20 or sum(profile_coverage) < 500:
                raise RuntimeError(
                    f'Insufficient Phase3 disorder-profile coverage: {profile_coverage}')
            report['disorder_profile_coverage'] = profile_coverage
            audit_paths.append(phase3_audit)
        report['audits'] = audit_paths
        supervision = disorder_supervision_contract(args.stage1_config)
        report['disorder_supervision'] = supervision
        report['rmsf_calibration'] = calibration_report(
            args.calibration_report, args.expected_count, supervision)

    report['checkpoint'] = checkpoint_report(args.checkpoint, args.config)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
