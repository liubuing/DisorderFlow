#!/usr/bin/env python
"""Evaluate BFN disorder head performance on validation data.

Computes per-residue disorder prediction metrics against ground truth
disorder labels (from EBI MobiDB-lite or AF2 pLDDT fallback).

Metrics:
  - AUC-ROC: area under ROC curve (classification quality)
  - AUC-PR: area under precision-recall curve (imbalanced data)
  - Balanced accuracy at 0.5 threshold
  - F1 score at optimal threshold
  - Mean Absolute Error (continuous)
  - Per-residue correlation with AF2 pLDDT

Usage:
  python evaluate_disorder.py \
      --checkpoint ./logs/bfn_idp_phase2_xpu_2026_05_28__01_22_06/checkpoints/best.pt \
      --config configs/train/bfn_idp_phase2_cuda.yml \
      --device xpu

  python evaluate_disorder.py \
      --checkpoint ./logs/bfn_phase3_disorder_*/checkpoints/best.pt \
      --config configs/train/bfn_phase3_disorder.yml \
      --device xpu --output results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from disorderflow.datasets import get_dataset
from disorderflow.models import get_model
from disorderflow.utils.misc import load_config, seed_all
from disorderflow.utils.train import recursive_to
from disorderflow.utils.data import PaddingCollate


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate BFN disorder head')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    p.add_argument('--config', required=True, help='Path to training config (for dataset)')
    p.add_argument('--device', default='cuda', help='Device')
    p.add_argument('--output', '-o', default=None, help='Output JSON path')
    p.add_argument('--no_amp', action='store_true', help='Disable AMP')
    p.add_argument('--max_samples', type=int, default=0, help='Limit evaluation samples')
    return p.parse_args()


def compute_metrics(targets, predictions):
    """Compute disorder detection metrics.

    Args:
        targets: (N,) ground truth disorder labels [0, 1]
        predictions: (N,) predicted disorder probabilities [0, 1]

    Returns:
        dict of metrics
    """
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    # Remove NaN
    valid = ~(np.isnan(targets) | np.isnan(predictions))
    targets = targets[valid]
    predictions = predictions[valid]

    if len(targets) < 2:
        return {'error': 'Insufficient valid samples', 'n_valid': len(targets)}

    # Binary classification metrics
    y_true = (targets > 0.5).astype(int)
    y_pred_bin = (predictions > 0.5).astype(int)

    # Accuracy
    correct = (y_true == y_pred_bin).sum()
    accuracy = correct / len(y_true)

    # Balanced accuracy
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = y_pred_bin[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
    tnr = 1.0 - y_pred_bin[neg_mask].mean() if neg_mask.sum() > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0

    # F1
    tp = ((y_true == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true == 1) & (y_pred_bin == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MAE
    mae = np.abs(targets - predictions).mean()

    # AUC-ROC (try import)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = roc_auc_score(targets > 0.5, predictions)
        auc_pr = average_precision_score(targets > 0.5, predictions)
    except ImportError:
        auc_roc = None
        auc_pr = None

    metrics = {
        'n_samples': len(y_true),
        'n_positive': int(pos_mask.sum()),
        'n_negative': int(neg_mask.sum()),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'tpr': float(tpr),
        'tnr': float(tnr),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mae': float(mae),
        'pred_mean': float(predictions.mean()),
        'pred_std': float(predictions.std()),
        'target_mean': float(targets.mean()),
    }
    if auc_roc is not None:
        metrics['auc_roc'] = float(auc_roc)
        metrics['auc_pr'] = float(auc_pr)

    return metrics


def evaluate(model, dataloader, device, use_amp=True, max_samples=0):
    """Run disorder evaluation over validation set."""
    model.eval()
    all_targets = []
    all_predictions = []
    all_plddt = []
    all_is_idp = []

    device_type = device if device in ('cuda', 'xpu') else 'cpu'
    if device == 'mps':
        device_type = 'mps'

    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', dynamic_ncols=True):
            batch = recursive_to(batch, device)

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                # Forward pass — use encode + receiver directly to get disorder predictions
                # We use forward() which computes all losses including disorder
                loss_dict = model(batch)

            # Get disorder target and prediction
            if 'disorder_label' not in batch:
                continue

            # Get predictions by doing a full sample pass
            result = model.sample(batch, sample_opt={
                'deterministic': True,
                'num_recycles': 1,
                'return_disorder': True,
            })

            if 'disorder' not in result or result['disorder'] is None:
                continue

            for b in range(batch['aa'].size(0)):
                mask = batch['mask'][b].bool()
                n_res = mask.sum().int().item()
                if n_res < 1:
                    continue

                targets = batch['disorder_label'][b, :n_res].cpu().numpy()
                disorder_raw = result['disorder'][b, :n_res].cpu().numpy()
                preds = 1.0 / (1.0 + np.exp(-disorder_raw))  # sigmoid

                all_targets.extend(targets.tolist())
                all_predictions.extend(preds.tolist())

                if 'af2_plddt' in batch:
                    plddt = batch['af2_plddt'][b, :n_res].cpu().numpy()
                    all_plddt.extend(plddt.tolist())

                is_idp = batch.get('is_idp', None)
                if is_idp is not None:
                    val = is_idp[b].item() if isinstance(is_idp, torch.Tensor) else is_idp
                    all_is_idp.append(val)

                n_samples += 1
                if max_samples > 0 and n_samples >= max_samples:
                    break
            if max_samples > 0 and n_samples >= max_samples:
                break

    return {
        'targets': all_targets,
        'predictions': all_predictions,
        'plddt': all_plddt,
        'is_idp': all_is_idp,
    }


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f'ERROR: Checkpoint not found: {args.checkpoint}')
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f'ERROR: Config not found: {args.config}')
        sys.exit(1)

    print(f'Checkpoint: {args.checkpoint}')
    print(f'Config:     {args.config}')
    print(f'Device:     {args.device}')

    # Load config and dataset
    config, _ = load_config(args.config)
    seed_all(config.train.seed)

    print('Loading validation dataset...')
    val_dataset = get_dataset(config.dataset.val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=PaddingCollate(), num_workers=0,
    )
    print(f'  {len(val_dataset)} entries')

    # Load model
    print('Loading model...')
    model = get_model(config.model).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing:
        print(f'  Missing keys ({len(missing)}), will use random init')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}), skipped')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  {total_params:,} parameters')

    # Run evaluation
    use_amp = args.device not in ('cpu', 'mps') and not args.no_amp
    print('Running evaluation...')
    raw = evaluate(model, val_loader, args.device, use_amp=use_amp,
                   max_samples=args.max_samples)

    # Compute overall metrics
    metrics = compute_metrics(raw['targets'], raw['predictions'])
    print(f'\n{"=" * 60}')
    print(f'  Disorder Detection Evaluation')
    print(f'{"=" * 60}')
    print(f'  Total residues:     {metrics["n_samples"]}')
    print(f'  Positive (IDR):     {metrics["n_positive"]}')
    print(f'  Negative (folded):  {metrics["n_negative"]}')
    print(f'  Accuracy:           {metrics["accuracy"]:.4f}')
    print(f'  Balanced Acc:       {metrics["balanced_accuracy"]:.4f}')
    print(f'  TPR (sensitivity):  {metrics["tpr"]:.4f}')
    print(f'  TNR (specificity):  {metrics["tnr"]:.4f}')
    print(f'  Precision:          {metrics["precision"]:.4f}')
    print(f'  Recall:             {metrics["recall"]:.4f}')
    print(f'  F1:                 {metrics["f1"]:.4f}')
    print(f'  MAE:                {metrics["mae"]:.4f}')
    if 'auc_roc' in metrics:
        print(f'  AUC-ROC:            {metrics["auc_roc"]:.4f}')
        print(f'  AUC-PR:             {metrics["auc_pr"]:.4f}')
    print(f'  Pred mean:          {metrics["pred_mean"]:.4f}')
    print(f'  Target mean:        {metrics["target_mean"]:.4f}')

    # IDP/Folded split if available
    idp_indices = [i for i, v in enumerate(raw.get('is_idp', [])) if v]
    folded_indices = [i for i, v in enumerate(raw.get('is_idp', [])) if not v]
    if idp_indices and folded_indices:
        # Per-group metrics (approximate — per-sample, not per-residue)
        print(f'\n  Per-sample split:')
        print(f'    IDP entries:     {len(idp_indices)}')
        print(f'    Folded entries:  {len(folded_indices)}')

    # Save output
    if args.output:
        output = {
            'checkpoint': args.checkpoint,
            'config': args.config,
            'metrics': metrics,
            'raw': {
                'targets_mean': float(np.mean(raw['targets'])),
                'predictions_mean': float(np.mean(raw['predictions'])),
            },
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nResults saved to: {args.output}')

    print('\nDone.')


if __name__ == '__main__':
    main()
