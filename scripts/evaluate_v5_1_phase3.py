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
    # Counterfactual evaluation only needs the factual receiver pass. Disable
    # auxiliary training forwards so hooks cannot observe negative-loss arms.
    for key in ('disorder_position_rank', 'disorder_mismatch_rank'):
        model.bfn.loss_weight[key] = 0.0
    model.eval()
    return model, checkpoint['config']


def build_dataset(config, lmdb_path=None, lookup_path=None, lookup_split=None):
    dataset_config = copy.deepcopy(config.dataset.val)
    if lmdb_path:
        dataset_config.lmdb_path = lmdb_path
    base = get_dataset(dataset_config)
    lookup_path = lookup_path or config.dataset.get('disorder_lookup_val') or config.dataset.get('disorder_lookup')
    expected_split = lookup_split or config.dataset.get('disorder_lookup_val_split', 'val')
    lookup = load_disorder_lookup(
        lookup_path,
        expected_split=expected_split if lookup_path else None,
        expected_ids=base.ids if lookup_path else None,
        require_envelope=bool(lookup_path),
    )
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
    cb = batch['pos_heavyatom'][:, :, 4]
    cb_mask = batch['mask_heavyatom'][:, :, 4]
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


def condition_batch(batch, arm, donor_profile=None, seed=0):
    if arm == 'factual':
        return
    profile = batch['epitope_disorder_profile']
    antigen = batch['fragment_type'] == 3
    if arm == 'shuffled':
        generator = torch.Generator(device=profile.device)
        generator.manual_seed(seed)
        for row in range(profile.shape[0]):
            values = profile[row][antigen[row]].clone()
            order = torch.randperm(len(values), generator=generator, device=profile.device)
            profile[row][antigen[row]] = values[order]
        return
    if arm == 'zero':
        profile.zero_()
        value = 0.0
    elif arm == 'one':
        profile.zero_()
        value = 1.0
    elif arm == 'mismatched':
        donor = np.asarray(donor_profile, dtype=np.float32)
        for row in range(profile.shape[0]):
            indices = torch.where(antigen[row])[0]
            if not len(indices):
                continue
            mapped = np.interp(
                np.linspace(0.0, 1.0, len(indices)),
                np.linspace(0.0, 1.0, len(donor)), donor)
            factual_values = profile[row, indices].detach().cpu().numpy()
            donor_order = np.argsort(mapped, kind='stable')
            matched = np.empty_like(mapped, dtype=np.float32)
            matched[donor_order] = np.sort(factual_values)
            profile[row, indices] = torch.as_tensor(
                matched, device=profile.device, dtype=profile.dtype)
        value = float(profile[antigen].mean()) if antigen.any() else 0.0
    else:
        raise ValueError(f'Unknown counterfactual arm: {arm}')
    if arm in ('zero', 'one'):
        profile[antigen] = value
    batch['epitope_disorder'] = torch.full_like(batch['epitope_disorder'], value)


def evaluate_arm(model, source_batch, device, seed, arm='factual', donor_profile=None, shuffle_cdr=False):
    batch = recursive_to(copy.deepcopy(source_batch), device)
    condition_batch(batch, arm, donor_profile, seed)
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
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output)

    handle = model.bfn.receiver.register_forward_hook(hook)
    try:
        seed_all(seed)
        with torch.inference_mode():
            model(batch)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError('Receiver hook did not capture an output')
    output = captured[0]
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


def paired_logit_effect(reference, alternate):
    ref_logits = reference['cdr_logits']
    alt_logits = alternate['cdr_logits']
    ref_probs = F.softmax(ref_logits, dim=-1)
    alt_probs = F.softmax(alt_logits, dim=-1)
    midpoint = 0.5 * (ref_probs + alt_probs)
    js = 0.5 * (
        F.kl_div(midpoint.log(), ref_probs, reduction='batchmean')
        + F.kl_div(midpoint.log(), alt_probs, reduction='batchmean')
    )
    hamming = (ref_logits.argmax(dim=-1) != alt_logits.argmax(dim=-1)).float().mean()
    return {
        'mean_abs_logit_change': float((ref_logits - alt_logits).abs().mean()),
        'js_divergence': float(js),
        'argmax_hamming': float(hamming),
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
    parser.add_argument('--disorder-lookup', help='Override the checkpoint dataset lookup for counterfactual evaluation')
    parser.add_argument('--lmdb', help='Override the checkpoint validation LMDB')
    parser.add_argument('--lookup-split', help='Required split label in the lookup artifact')
    parser.add_argument('--blind', action='store_true',
                        help='Write-once full external evaluation; forbids subsampling')
    parser.add_argument('--independence-audit', type=Path)
    args = parser.parse_args()

    model, config = load_checkpoint(args.checkpoint, args.device, args.allow_raw)
    output = Path(args.output)
    if args.blind and (args.max_samples or output.exists()):
        raise RuntimeError('Blind evaluation must be full-set and write to a new output path')
    dataset, lookup, ids = build_dataset(
        config, args.lmdb, args.disorder_lookup, args.lookup_split)
    if args.max_samples:
        ids = ids[:args.max_samples]
        dataset._base.ids = ids
        dataset._ids = ids
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=PaddingCollate())
    lookup_casefold = {str(key).casefold(): value for key, value in lookup.items() if value is not None}
    donor_profiles = [np.asarray(lookup_casefold[str(sample_id).casefold()]) for sample_id in ids]

    arms = {'factual': [], 'zero': [], 'one': [], 'shuffled': [], 'mismatched': []}
    contact_scores, contact_labels = [], []
    contrastive_scores, contrastive_labels = [], []
    logit_changes = []
    counterfactual_effects = {name: [] for name in ('zero', 'one', 'shuffled', 'mismatched')}
    for index, batch in enumerate(loader):
        seed = args.seed + index
        factual = evaluate_arm(model, batch, args.device, seed, 'factual')
        zero = evaluate_arm(model, batch, args.device, seed, 'zero')
        one = evaluate_arm(model, batch, args.device, seed, 'one')
        shuffled = evaluate_arm(model, batch, args.device, seed, 'shuffled')
        donor_profile = donor_profiles[(index + max(1, len(donor_profiles) // 2)) % len(donor_profiles)]
        mismatched = evaluate_arm(model, batch, args.device, seed, 'mismatched', donor_profile)
        negative = evaluate_arm(model, batch, args.device, seed, 'factual', shuffle_cdr=True)
        for name, result in (
                ('factual', factual), ('zero', zero), ('one', one),
                ('shuffled', shuffled), ('mismatched', mismatched)):
            arms[name].append({key: value for key, value in result.items() if key != 'cdr_logits'
                               and not key.startswith('contact_') and key != 'contrastive_logit'})
        for name, result in (
                ('zero', zero), ('one', one), ('shuffled', shuffled), ('mismatched', mismatched)):
            counterfactual_effects[name].append(paired_logit_effect(factual, result))
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
    shuffled_nll = np.asarray([item['native_nll'] for item in arms['shuffled']])
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
            'classification': 'external blind routing test' if args.blind
            else 'development or legacy validation; not external blind test',
        },
        'arms': {name: mean_metrics(records) for name, records in arms.items()},
        'paired_effects': {
            'factual_minus_zero_native_nll': paired_summary(factual_nll - zero_nll, args.seed),
            'factual_minus_mismatched_native_nll': paired_summary(factual_nll - mismatch_nll, args.seed),
            'factual_minus_shuffled_native_nll': paired_summary(factual_nll - shuffled_nll, args.seed),
            'factual_vs_zero_mean_abs_logit_change': condition_change,
            'disorder_entropy_spearman_factual': rank_correlation(disorder, factual_entropy),
            'disorder_entropy_spearman_zero': rank_correlation(disorder, zero_entropy),
            'deterministic_counterfactuals': {
                name: {
                    metric: paired_summary([item[metric] for item in records], args.seed)
                    for metric in ('mean_abs_logit_change', 'js_divergence', 'argmax_hamming')
                }
                for name, records in counterfactual_effects.items()
            },
        },
        'contact': contact,
        'contrastive': contrastive,
    }
    mismatch_effect = report['paired_effects']['factual_minus_mismatched_native_nll']
    shuffled_effect = report['paired_effects']['factual_minus_shuffled_native_nll']
    shuffled_change = report['paired_effects']['deterministic_counterfactuals']['shuffled']['mean_abs_logit_change']
    report['gates'] = {
        'contact_auroc_at_least_0_60': contact['auroc'] is not None and contact['auroc'] >= 0.60,
        'contrastive_auroc_at_least_0_60': contrastive['auroc'] is not None and contrastive['auroc'] >= 0.60,
        'condition_changes_logits': condition_change['mean'] is not None and condition_change['mean'] > 1e-4,
        'factual_better_than_mismatched_ci95': mismatch_effect['ci95'][1] < 0,
        'factual_better_than_shuffled_ci95': shuffled_effect['ci95'][1] < 0,
        'position_order_changes_logits': shuffled_change['mean'] is not None and shuffled_change['mean'] > 1e-4,
    }
    routing_gate_names = (
        'condition_changes_logits',
        'factual_better_than_mismatched_ci95',
        'factual_better_than_shuffled_ci95',
        'position_order_changes_logits',
    )
    report['routing_gates_passed'] = all(
        report['gates'][name] for name in routing_gate_names)
    report['auxiliary_diagnostics'] = {
        'contact_head_gate': report['gates']['contact_auroc_at_least_0_60'],
        'contrastive_head_gate': report['gates']['contrastive_auroc_at_least_0_60'],
        'scope': 'frozen auxiliary heads; not routing-only acceptance criteria',
    }
    report['evaluation_protocol'] = {
        'blind': args.blind,
        'checkpoint_selection_allowed': not args.blind,
        'subsampling_allowed': not args.blind,
        'write_once_output': args.blind,
    }
    if args.independence_audit:
        audit = json.loads(args.independence_audit.read_text(encoding='utf-8'))
        report['independence'] = audit.get('interface_eligibility', report['independence'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if args.require_gates and not report['routing_gates_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
