#!/usr/bin/env python3
"""Deterministic held-out Phase3 evaluation and disorder-condition ablation."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.datasets.disorder_augmented import (  # noqa: E402
    ContrastiveNegativeDataset,
    DisorderAugmentedDataset,
    load_disorder_lookup,
)
from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import seed_all  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402


def binary_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if not n_pos or not n_neg:
        return {'auroc': None, 'auprc': None, 'n': len(labels), 'positive_rate': n_pos / max(len(labels), 1)}

    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    descending = np.argsort(-scores, kind='stable')
    sorted_labels = labels[descending]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    auprc = float(precision[sorted_labels == 1].mean())
    return {
        'auroc': float(auroc),
        'auprc': auprc,
        'n': len(labels),
        'positive_rate': n_pos / len(labels),
    }


def rank_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def paired_summary(values, seed=2081, n_bootstrap=2000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'n': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return {
        'n': len(values),
        'mean': float(values.mean()),
        'ci95': [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def load_checkpoint(path, device, allow_raw=False):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('weights_kind') != 'ema' and not allow_raw:
        raise RuntimeError('Authoritative evaluation requires an EMA best checkpoint')
    model = get_model(checkpoint['config'].model).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint['config']


def build_dataset(config):
    base = get_dataset(config.dataset.val)
    lookup = load_disorder_lookup(config.dataset.disorder_lookup)
    valid = {str(key).casefold() for key, value in lookup.items() if value is not None}
    ids = [sample_id for sample_id in base.ids if str(sample_id).casefold() in valid]
    base.ids = ids
    dataset = DisorderAugmentedDataset(base, lookup, ids)
    if config.dataset.get('contrastive_cross_sample_negatives', False):
        dataset = ContrastiveNegativeDataset(dataset)
    return dataset, lookup, ids


def contact_targets(batch):
    mask_gen = batch['generate_flag'].bool()
    mask_antigen = batch['mask_antigen'].bool()
    cb = batch['pos_heavyatom'][:, :, 3]
    cb_mask = batch['mask_heavyatom'][:, :, 3]
    ca = batch['pos_heavyatom'][:, :, 1]
    positions = torch.where(cb_mask.unsqueeze(-1), cb, ca)
    labels = torch.zeros_like(mask_gen, dtype=torch.float32)
    for batch_index in range(mask_gen.shape[0]):
        cdr = mask_gen[batch_index]
        antigen = mask_antigen[batch_index]
        if cdr.any() and antigen.any():
            distances = torch.cdist(positions[batch_index][cdr], positions[batch_index][antigen])
            labels[batch_index][cdr] = (distances.min(dim=1).values < 8.0).float()
    return labels


def condition_batch(batch, arm, donor_mean=None):
    if arm == 'factual':
        return
    profile = batch['epitope_disorder_profile']
    antigen = batch['fragment_type'] == 3
    profile.zero_()
    value = 0.0 if arm == 'zero' else float(donor_mean)
    profile[antigen] = value
    batch['epitope_disorder'] = torch.full_like(batch['epitope_disorder'], value)


def evaluate_arm(model, source_batch, device, seed, arm='factual', donor_mean=None, shuffle_cdr=False):
    batch = recursive_to(copy.deepcopy(source_batch), device)
    condition_batch(batch, arm, donor_mean)
    if shuffle_cdr:
        negative_aa = batch.get('contrastive_negative_aa')
        for row in range(batch['aa'].shape[0]):
            indices = torch.where(batch['generate_flag'][row].bool())[0]
            if len(indices) > 1:
                if negative_aa is not None:
                    batch['aa'][row, indices] = negative_aa[row, indices]
                else:
                    batch['aa'][row, indices] = batch['aa'][row, indices.roll(1)]
    batch['fixed_t'] = 0.5
    captured = {}

    def hook(_module, _inputs, output):
        captured['receiver'] = output

    handle = model.bfn.receiver.register_forward_hook(hook)
    try:
        seed_all(seed)
        with torch.inference_mode():
            model(batch)
    finally:
        handle.remove()
    output = captured['receiver']
    logits = output[0][..., :20].float()
    pred_contact = output[8].float()
    pred_contrastive = output[9].float()
    cdr = batch['generate_flag'].bool() & batch['mask'].bool()
    if not cdr.any():
        raise RuntimeError('Evaluation sample contains no generated CDR residues')
    targets = batch['aa'][cdr].long()
    cdr_logits = logits[cdr]
    probabilities = F.softmax(cdr_logits, dim=-1)
    native_nll = F.cross_entropy(cdr_logits, targets).item()
    entropy = (-(probabilities * torch.log(probabilities.clamp(min=1e-8))).sum(dim=-1)).mean().item()
    recovery = (cdr_logits.argmax(dim=-1) == targets).float().mean().item()
    profile = batch['epitope_disorder_profile'].float()
    antigen = batch['mask_antigen'].bool()
    disorder_mean = ((profile * antigen).sum() / antigen.sum().clamp(min=1)).item()
    labels = contact_targets(batch)
    return {
        'native_nll': native_nll,
        'native_ppl': math.exp(min(native_nll, 20.0)),
        'recovery': recovery,
        'entropy': entropy,
        'disorder_mean': disorder_mean,
        'cdr_logits': cdr_logits.cpu(),
        'contact_scores': torch.sigmoid(pred_contact[cdr]).cpu().numpy(),
        'contact_labels': labels[cdr].cpu().numpy(),
        'contrastive_logit': float(pred_contrastive[0].item()),
    }


def mean_metrics(records):
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in ('native_nll', 'native_ppl', 'recovery', 'entropy')
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2081)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--require-gates', action='store_true')
    parser.add_argument('--allow-raw', action='store_true',
                        help='Diagnostic only; authoritative gates require EMA weights')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    model, config = load_checkpoint(args.checkpoint, args.device, args.allow_raw)
    dataset, lookup, ids = build_dataset(config)
    if args.max_samples:
        ids = ids[:args.max_samples]
        dataset._base.ids = ids
        dataset._ids = ids
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=PaddingCollate())
    lookup_casefold = {str(key).casefold(): value for key, value in lookup.items() if value is not None}
    donor_means = [float(np.asarray(lookup_casefold[str(sample_id).casefold()]).mean()) for sample_id in ids]

    arms = {'factual': [], 'zero': [], 'mismatched': []}
    contact_scores, contact_labels = [], []
    contrastive_scores, contrastive_labels = [], []
    logit_changes = []
    for index, batch in enumerate(loader):
        seed = args.seed + index
        factual = evaluate_arm(model, batch, args.device, seed, 'factual')
        zero = evaluate_arm(model, batch, args.device, seed, 'zero')
        donor_mean = donor_means[(index + max(1, len(donor_means) // 2)) % len(donor_means)]
        mismatched = evaluate_arm(model, batch, args.device, seed, 'mismatched', donor_mean)
        negative = evaluate_arm(model, batch, args.device, seed, 'factual', shuffle_cdr=True)
        for name, result in (('factual', factual), ('zero', zero), ('mismatched', mismatched)):
            arms[name].append({key: value for key, value in result.items() if key != 'cdr_logits'
                               and not key.startswith('contact_') and key != 'contrastive_logit'})
        logit_changes.append(float((factual['cdr_logits'] - zero['cdr_logits']).abs().mean()))
        contact_scores.extend(factual['contact_scores'].tolist())
        contact_labels.extend(factual['contact_labels'].astype(int).tolist())
        contrastive_scores.extend([factual['contrastive_logit'], negative['contrastive_logit']])
        contrastive_labels.extend([1, 0])
        if (index + 1) % 10 == 0:
            print(f'Evaluated {index + 1}/{len(dataset)} complexes', flush=True)

    factual_nll = np.asarray([item['native_nll'] for item in arms['factual']])
    zero_nll = np.asarray([item['native_nll'] for item in arms['zero']])
    mismatch_nll = np.asarray([item['native_nll'] for item in arms['mismatched']])
    disorder = [item['disorder_mean'] for item in arms['factual']]
    factual_entropy = [item['entropy'] for item in arms['factual']]
    zero_entropy = [item['entropy'] for item in arms['zero']]
    contact = binary_metrics(contact_scores, contact_labels)
    contrastive = binary_metrics(contrastive_scores, contrastive_labels)
    condition_change = paired_summary(logit_changes, args.seed)
    report = {
        'checkpoint': args.checkpoint,
        'weights_kind': 'raw_diagnostic' if args.allow_raw else 'ema',
        'fixed_t': 0.5,
        'n_complexes': len(arms['factual']),
        'independence': {
            'antigen_homology_isolated': True,
            'antibody_homology_isolated': False,
            'classification': 'held-out validation, not external blind test',
        },
        'arms': {name: mean_metrics(records) for name, records in arms.items()},
        'paired_effects': {
            'factual_minus_zero_native_nll': paired_summary(factual_nll - zero_nll, args.seed),
            'factual_minus_mismatched_native_nll': paired_summary(factual_nll - mismatch_nll, args.seed),
            'factual_vs_zero_mean_abs_logit_change': condition_change,
            'disorder_entropy_spearman_factual': rank_correlation(disorder, factual_entropy),
            'disorder_entropy_spearman_zero': rank_correlation(disorder, zero_entropy),
        },
        'contact': contact,
        'contrastive': contrastive,
    }
    report['gates'] = {
        'contact_auroc_at_least_0_60': contact['auroc'] is not None and contact['auroc'] >= 0.60,
        'contrastive_auroc_at_least_0_60': contrastive['auroc'] is not None and contrastive['auroc'] >= 0.60,
        'condition_changes_logits': condition_change['mean'] is not None and condition_change['mean'] > 1e-4,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if args.require_gates and not all(report['gates'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
