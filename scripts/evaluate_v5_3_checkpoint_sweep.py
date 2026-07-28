#!/usr/bin/env python3
"""Select a flexibility checkpoint on the independent NMR ranking holdout."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_stage1_nmr_ranking import load_model, passes_gate, predict, score
from disorderflow.datasets import get_dataset
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.misc import load_config, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--calibration-report',
        default='data/confidence_conformation_v5/rmsf_calibration_1289.json')
    parser.add_argument(
        '--lmdb',
        default='data/confidence_conformation_v5_clustered/confidence_calibration.lmdb')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2071)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config, _ = load_config(args.config)
    dataset_config = copy.deepcopy(config.dataset.val)
    dataset_config.db_path = args.lmdb
    dataset_config.max_residues = 500
    dataset_config.fix_backbone = True
    loader = DataLoader(
        get_dataset(dataset_config), batch_size=1, shuffle=False,
        collate_fn=PaddingCollate())
    references = {
        item['accession']: item
        for item in json.loads(Path(args.calibration_report).read_text(
            encoding='utf-8'))['proteins']
    }

    seed_all(args.seed)
    baseline_model = load_model(config, args.baseline_checkpoint, args.device, strict=False)
    baseline = score(predict(
        baseline_model, loader, references, args.device, args.seed), references)
    del baseline_model
    torch.cuda.empty_cache()

    checkpoints = sorted(
        (path for path in Path(args.checkpoint_dir).glob('*.pt') if path.stem.isdigit()),
        key=lambda path: int(path.stem))
    rows = []
    for checkpoint in checkpoints:
        seed_all(args.seed)
        model = load_model(config, checkpoint, args.device, strict=True)
        trained = score(predict(model, loader, references, args.device, args.seed), references)
        gate = passes_gate(trained, baseline)
        rows.append({
            'checkpoint': str(checkpoint),
            'iteration': int(checkpoint.stem),
            'summary': trained['summary'],
            'annotated_idp': trained['annotated_idp'],
            'gate': gate,
        })
        del model
        torch.cuda.empty_cache()

    accepted = [row for row in rows if row['gate']['passed']]
    selected = max(
        accepted,
        key=lambda row: (
            row['summary']['median_spearman'],
            row['annotated_idp']['median_spearman']),
        default=None)
    report = {
        'status': 'pass' if selected else 'fail',
        'selection_rule': 'highest overall median Spearman among dual-gate checkpoints',
        'baseline_checkpoint': args.baseline_checkpoint,
        'baseline': {
            'summary': baseline['summary'],
            'annotated_idp': baseline['annotated_idp'],
        },
        'selected': selected,
        'checkpoints': rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'status': report['status'],
        'selected': selected,
        'accepted_iterations': [row['iteration'] for row in accepted],
    }, indent=2))
    if not selected:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
