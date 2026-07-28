#!/usr/bin/env python3
"""Check that flexibility adaptation preserves held-out StateContrast ranking."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disorderflow.datasets import get_dataset
from disorderflow.models import get_model
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate
from disorderflow.utils.misc import load_config, seed_all
from disorderflow.utils.train import recursive_to


def evaluate(checkpoint_path, config, loader, device, seed):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_model(config.model).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    losses = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            seed_all(seed + index)
            batch = recursive_to(batch, device)
            batch['fixed_t'] = 0.5
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device == 'cuda'):
                output = model(batch)
            loss = output['grouped_contrastive']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite grouped loss in validation group {index}')
            losses.append(float(loss))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return {
        'checkpoint': str(checkpoint_path),
        'groups': len(losses),
        'mean_grouped_contrastive_loss': sum(losses) / len(losses),
        'min_grouped_contrastive_loss': min(losses),
        'max_grouped_contrastive_loss': max(losses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument(
        '--config', default='configs/train/bfn_v5_2_grouped_structural.yml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2083)
    parser.add_argument('--max-relative-regression', type=float, default=0.05)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config, _ = load_config(args.config)
    dataset = get_dataset(config.dataset.val)
    loader = DataLoader(
        dataset,
        batch_sampler=CompleteGroupBatchSampler(
            dataset, config.train.batch_size, shuffle=False, seed=args.seed),
        collate_fn=PaddingCollate(),
        num_workers=0,
    )
    baseline = evaluate(args.baseline, config, loader, args.device, args.seed)
    candidate = evaluate(args.candidate, config, loader, args.device, args.seed)
    relative_change = (
        candidate['mean_grouped_contrastive_loss']
        / baseline['mean_grouped_contrastive_loss'] - 1.0)
    report = {
        'status': 'pass' if relative_change <= args.max_relative_regression else 'fail',
        'criterion': {
            'metric': 'heldout_mean_grouped_contrastive_loss',
            'maximum_relative_regression': args.max_relative_regression,
        },
        'relative_change': relative_change,
        'baseline': baseline,
        'candidate': candidate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if report['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
