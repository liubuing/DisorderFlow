#!/usr/bin/env python3
"""Accept the corrected v5.1 checkpoint only on supported scientific gates."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--phase3-report', required=True)
    parser.add_argument('--nmr-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--state-file', required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('weights_kind') != 'ema':
        raise RuntimeError('Final corrected checkpoint must contain EMA weights')
    if any(not torch.isfinite(value).all() for value in checkpoint['model'].values()
           if isinstance(value, torch.Tensor)):
        raise RuntimeError('Final corrected checkpoint contains non-finite tensors')

    phase3 = json.loads(Path(args.phase3_report).read_text(encoding='utf-8'))
    nmr = json.loads(Path(args.nmr_report).read_text(encoding='utf-8'))
    nll_effect = phase3['paired_effects']['factual_minus_zero_native_nll']
    supported_gates = {
        'contact_auroc_at_least_0_60': phase3['contact']['auroc'] >= 0.60,
        'condition_changes_logits': (
            phase3['paired_effects']['factual_vs_zero_mean_abs_logit_change']['mean'] > 1e-4),
        'factual_condition_improves_native_nll': nll_effect['ci95'][1] < 0.0,
        'condition_tracks_disorder_entropy': (
            phase3['paired_effects']['disorder_entropy_spearman_factual'] > 0.30),
        'nmr_rank_retained': bool(nmr['gate']['passed']),
    }
    if not all(supported_gates.values()):
        raise RuntimeError(f'Corrected v5.1 supported gates failed: {supported_gates}')

    report = {
        'status': 'accepted_with_scope_limits',
        'checkpoint': args.checkpoint,
        'iteration': checkpoint.get('iteration'),
        'weights_kind': checkpoint.get('weights_kind'),
        'supported_gates': supported_gates,
        'metrics': {
            'nmr_median_spearman': nmr['trained']['summary']['median_spearman'],
            'nmr_idp_median_spearman': nmr['trained']['annotated_idp']['median_spearman'],
            'contact_auroc': phase3['contact']['auroc'],
            'contact_auprc': phase3['contact']['auprc'],
            'condition_logit_change': phase3['paired_effects'][
                'factual_vs_zero_mean_abs_logit_change'],
            'factual_minus_zero_native_nll': nll_effect,
            'disorder_entropy_spearman': phase3['paired_effects'][
                'disorder_entropy_spearman_factual'],
        },
        'unsupported_objectives': {
            'cdr_antigen_contrastive_compatibility': {
                'status': 'disabled_pending_valid_negative_pair_data',
                'heldout_auroc': phase3['contrastive']['auroc'],
                'reason': (
                    'Single-positive-per-antigen Phase3 data did not support '
                    'generalizable compatibility discrimination.'),
            },
        },
        'evidence': {
            'phase3_report': args.phase3_report,
            'nmr_report': args.nmr_report,
        },
    }
    output = Path(args.output)
    state = Path(args.state_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    state.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    state.write_text(args.checkpoint + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
