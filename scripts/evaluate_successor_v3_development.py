#!/usr/bin/env python
"""Evaluate successor-v3 specificity on the frozen exposed development panel."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from disorderflow.datasets import get_dataset
from disorderflow.models import get_model
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.misc import load_config, seed_all
from disorderflow.utils.train import recursive_to


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3203, 3217, 3229)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(checkpoint_path, model_config, device):
    model = get_model(copy.deepcopy(model_config)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}')
    model.eval()
    return model


def contact_labels(batch):
    mask_gen = batch['generate_flag'].bool() & batch['mask'].bool()
    antigen = batch['mask_antigen'].bool()
    cb = batch['pos_heavyatom'][:, :, 4]
    cb_mask = batch['mask_heavyatom'][:, :, 4].bool()
    ca = batch['pos_heavyatom'][:, :, 1]
    positions = torch.where(cb_mask.unsqueeze(-1), cb, ca)
    labels = torch.zeros_like(mask_gen, dtype=torch.float32)
    for row in range(mask_gen.shape[0]):
        if mask_gen[row].any() and antigen[row].any():
            distances = torch.cdist(
                positions[row][mask_gen[row]], positions[row][antigen[row]])
            labels[row][mask_gen[row]] = (
                distances.min(dim=1).values < 8.0).float()
    return labels, mask_gen


def component_bootstrap(values, seed=3251, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(model, pairs, fetch, device):
    rows = []
    contact_scores, contact_targets = [], []
    for index, (target, donor) in enumerate(pairs):
        source_batch = PaddingCollate()([fetch(target), fetch(donor)])
        gaps, factual_nll = [], []
        for seed in SEEDS:
            seed_all(seed + index)
            batch = recursive_to(copy.deepcopy(source_batch), device)
            batch['fixed_t'] = 0.5
            batch['return_per_sample_metrics'] = True
            with torch.inference_mode():
                losses = model(batch)
            if not bool(losses['antigen_mismatch_valid_per_sample'][0]):
                continue
            gaps.append(float(losses['antigen_mismatch_gap_per_sample'][0]))
            factual_nll.append(float(losses['factual_nll_per_sample'][0]))

        score_batch = recursive_to(copy.deepcopy(source_batch), device)
        with torch.inference_mode():
            scores = model.score(score_batch, fixed_t=0.5)
        labels, cdr_mask = contact_labels(score_batch)
        contact_scores.extend(torch.sigmoid(scores['contact'][0][cdr_mask[0]]).cpu().tolist())
        contact_targets.extend(labels[0][cdr_mask[0]].cpu().int().tolist())
        rows.append({
            'id': target['id'],
            'component': target['component'],
            'donor_id': donor['id'],
            'valid_seeds': len(gaps),
            'factual_minus_mismatched_native_h3_nll': (
                float(np.mean(gaps)) if gaps else None),
            'factual_native_h3_nll': (
                float(np.mean(factual_nll)) if factual_nll else None),
        })

    auroc = None
    if len(set(contact_targets)) == 2:
        auroc = float(roc_auc_score(contact_targets, contact_scores))
    return rows, {
        'auroc': auroc,
        'n_residues': len(contact_targets),
        'positive_fraction': float(np.mean(contact_targets)),
    }


def summarize(rows):
    valid = [row for row in rows if row['valid_seeds'] == len(SEEDS)]
    gaps = [row['factual_minus_mismatched_native_h3_nll'] for row in valid]
    nll = [row['factual_native_h3_nll'] for row in valid]
    return {
        'valid_components': len(valid),
        'valid_fraction': len(valid) / len(rows),
        'mean_factual_minus_mismatched_native_h3_nll': float(np.mean(gaps)),
        'component_bootstrap_ci95': component_bootstrap(gaps),
        'mean_factual_native_h3_nll': float(np.mean(nll)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--successor-checkpoint', type=Path, required=True)
    parser.add_argument('--stage-a-checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--manifest', type=Path,
        default=ROOT / 'data/successor_v3_development/manifest.json')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results/successor_v3_development/analysis.json')
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    config, _ = load_config('configs/train/bfn_successor_v3_h3_specificity.yml')
    dataset_config = copy.deepcopy(config.dataset.val)
    datasets = {}
    for source_class, path in {
        'adaptation': './data/sabdab2_abag_project_split_v2/adaptation.lmdb',
        'development': './data/sabdab2_abag_project_split_v2/dev.lmdb',
    }.items():
        dataset_config.lmdb_path = path
        datasets[source_class] = get_dataset(copy.deepcopy(dataset_config))

    records = {record['id']: record for record in manifest['records']}
    representatives = [records[item] for item in manifest['representative_ids']]
    indices = {
        source: {item: index for index, item in enumerate(dataset.all_ids)}
        for source, dataset in datasets.items()
    }

    def fetch(record):
        dataset = datasets[record['source_class']]
        return dataset[indices[record['source_class']][record['id']]]

    pairs = []
    for target in representatives:
        donors = [item for item in representatives if item['component'] != target['component']]
        donor = min(donors, key=lambda item: (
            abs(item['antigen_length'] - target['antigen_length']), item['id']))
        pairs.append((target, donor))

    successor = load_model(args.successor_checkpoint, config.model, args.device)
    successor_rows, successor_contact = evaluate(
        successor, pairs, fetch, args.device)
    del successor
    torch.cuda.empty_cache()
    stage_a = load_model(args.stage_a_checkpoint, config.model, args.device)
    stage_a_rows, stage_a_contact = evaluate(stage_a, pairs, fetch, args.device)

    successor_summary = summarize(successor_rows)
    stage_a_summary = summarize(stage_a_rows)
    successor_by_id = {row['id']: row for row in successor_rows}
    nll_deltas = [
        successor_by_id[row['id']]['factual_native_h3_nll']
        - row['factual_native_h3_nll']
        for row in stage_a_rows
        if row['valid_seeds'] == len(SEEDS)
        and successor_by_id[row['id']]['valid_seeds'] == len(SEEDS)
    ]
    mean_nll_delta = float(np.mean(nll_deltas))
    gates = {
        'minimum_12_components': successor_summary['valid_components'] >= 12,
        'valid_component_fraction_at_least_0_80': (
            successor_summary['valid_fraction'] >= 0.80),
        'specificity_ci95_upper_below_zero': (
            successor_summary['component_bootstrap_ci95'][1] < 0),
        'factual_nll_not_worse_than_stage_a_by_0_05': mean_nll_delta <= 0.05,
        'contact_auroc_at_least_0_60': (
            successor_contact['auroc'] is not None
            and successor_contact['auroc'] >= 0.60),
    }
    report = {
        'schema_version': 1,
        'classification': 'exposed development only; not confirmatory',
        'fixed_t': 0.5,
        'seeds': list(SEEDS),
        'manifest_sha256': sha256(args.manifest),
        'checkpoints': {
            'successor': {
                'path': str(args.successor_checkpoint),
                'sha256': sha256(args.successor_checkpoint),
            },
            'stage_a': {
                'path': str(args.stage_a_checkpoint),
                'sha256': sha256(args.stage_a_checkpoint),
            },
        },
        'successor': {**successor_summary, 'contact': successor_contact},
        'stage_a': {**stage_a_summary, 'contact': stage_a_contact},
        'successor_minus_stage_a_factual_native_h3_nll': {
            'mean': mean_nll_delta,
            'component_bootstrap_ci95': component_bootstrap(nll_deltas, seed=3253),
        },
        'gates': gates,
        'all_development_gates_passed': all(gates.values()),
        'rows': successor_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'classification', 'successor', 'stage_a',
        'successor_minus_stage_a_factual_native_h3_nll', 'gates',
        'all_development_gates_passed')}, indent=2))


if __name__ == '__main__':
    main()
