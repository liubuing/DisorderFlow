#!/usr/bin/env python3
"""Evaluate a Stage 1 disorder head against held-out solution-NMR ranks."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.datasets import get_dataset
from disorderflow.models import get_model
from disorderflow.utils.conformation_calibration import (
    calibration_metrics,
    calibration_verdict,
)
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.misc import load_config, seed_all
from disorderflow.utils.train import recursive_to


def load_model(config, checkpoint_path, device, strict):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['config'].model if strict else config.model
    model = get_model(model_config).to(device)
    state = checkpoint['model']
    if strict:
        model.load_state_dict(state, strict=True)
    else:
        compatible = {
            key: value for key, value in state.items()
            if key in model.state_dict()
            and model.state_dict()[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    model.eval()
    return model


def predict(model, loader, references, device, seed):
    predictions = {}
    captured = {}

    def capture_disorder(_module, _inputs, output):
        captured['logits'] = output.squeeze(-1).detach()

    handle = model.bfn.receiver.head_disorder.register_forward_hook(capture_disorder)
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                accession = str(batch['pdb_id'][0]).split('-')[0]
                if accession not in references:
                    continue
                seed_all(seed + batch_index)
                batch = recursive_to(batch, device)
                batch['fixed_t'] = 0.5
                captured.clear()
                model(batch)
                logits = captured.get('logits')
                if logits is None:
                    raise RuntimeError('Disorder head did not produce logits')
                predictions[accession] = logits[0].float().cpu().numpy()
    finally:
        handle.remove()
    return predictions


def score(predictions, references, minimum_proteins=10):
    results = []
    for accession, prediction in predictions.items():
        reference = references[accession]
        indices = np.asarray(reference['query_indices'], dtype=int)
        experimental = np.asarray(reference['experimental_rmsf'], dtype=np.float64)
        valid = indices < len(prediction)
        if valid.sum() < 20:
            continue
        metrics = calibration_metrics(prediction[indices[valid]], experimental[valid])
        results.append({
            'accession': accession,
            'disorder_fraction': reference.get('disorder_fraction'),
            **metrics,
        })
    idp = [
        result for result in results
        if result.get('disorder_fraction') is not None
        and float(result['disorder_fraction']) >= 0.20
    ]
    return {
        'summary': calibration_verdict(results, minimum_proteins),
        'annotated_idp': calibration_verdict(idp, minimum_proteins),
        'proteins': results,
    }


def passes_gate(trained, baseline):
    accepted = {'rank_only', 'pass_physical_label'}
    reasons = []
    for subgroup in ('summary', 'annotated_idp'):
        current = trained[subgroup]
        previous = baseline[subgroup]
        if current.get('verdict') not in accepted:
            reasons.append(f'{subgroup} verdict={current.get("verdict")}')
        if current.get('median_spearman') is None or previous.get('median_spearman') is None:
            reasons.append(f'{subgroup} missing median Spearman')
        elif current['median_spearman'] < previous['median_spearman']:
            reasons.append(f'{subgroup} Spearman below baseline')
    return {'passed': not reasons, 'reasons': reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--config', default='configs/train/bfn_v5_1_stage1_disorder.yml')
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
    dataset = get_dataset(dataset_config)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate())

    with open(args.calibration_report, encoding='utf-8') as handle:
        calibration = json.load(handle)
    references = {item['accession']: item for item in calibration['proteins']}

    seed_all(args.seed)
    trained_model = load_model(config, args.checkpoint, args.device, strict=True)
    trained_predictions = predict(
        trained_model, loader, references, args.device, args.seed)
    del trained_model
    torch.cuda.empty_cache()

    seed_all(args.seed)
    baseline_model = load_model(
        config, args.baseline_checkpoint, args.device, strict=False)
    baseline_predictions = predict(
        baseline_model, loader, references, args.device, args.seed)

    trained = score(trained_predictions, references)
    baseline = score(baseline_predictions, references)
    report = {
        'checkpoint': args.checkpoint,
        'baseline_checkpoint': args.baseline_checkpoint,
        'fixed_t': 0.5,
        'trained': trained,
        'baseline': baseline,
        'gate': passes_gate(trained, baseline),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'trained': trained['summary'],
        'trained_idp': trained['annotated_idp'],
        'baseline': baseline['summary'],
        'baseline_idp': baseline['annotated_idp'],
        'gate': report['gate'],
    }, indent=2))
    if not report['gate']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
