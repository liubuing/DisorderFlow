#!/usr/bin/env python3
"""Apply a frozen external-routing protocol to baseline and candidate reports."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--require-gates', action='store_true')
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    candidate = json.loads(args.candidate.read_text(encoding='utf-8'))
    if not baseline.get('evaluation_protocol', {}).get('blind'):
        raise RuntimeError('Baseline report is not marked as blind')
    if not candidate.get('evaluation_protocol', {}).get('blind'):
        raise RuntimeError('Candidate report is not marked as blind')
    if baseline['n_complexes'] != candidate['n_complexes']:
        raise RuntimeError('Baseline and candidate evaluated different sample counts')

    gates = protocol['external_blind']['candidate_gates']
    paired = candidate['paired_effects']
    mismatch = paired['factual_minus_mismatched_native_nll']
    shuffled = paired['factual_minus_shuffled_native_nll']
    shuffled_effects = paired['deterministic_counterfactuals']['shuffled']
    factual_degradation = (
        candidate['arms']['factual']['native_nll']
        - baseline['arms']['factual']['native_nll'])
    checks = {
        'mismatch_mean': mismatch['mean']
        <= gates['factual_minus_mismatched_nll_mean_at_most'],
        'mismatch_ci95': mismatch['ci95'][1]
        < gates['factual_minus_mismatched_nll_ci95_upper_below'],
        'shuffled_mean': shuffled['mean']
        <= gates['factual_minus_shuffled_nll_mean_at_most'],
        'shuffled_ci95': shuffled['ci95'][1]
        < gates['factual_minus_shuffled_nll_ci95_upper_below'],
        'shuffled_logit_change': shuffled_effects['mean_abs_logit_change']['mean']
        >= gates['shuffled_mean_abs_logit_change_at_least'],
        'shuffled_argmax_hamming': shuffled_effects['argmax_hamming']['mean']
        >= gates['shuffled_argmax_hamming_at_least'],
        'factual_nll_preserved': factual_degradation
        <= gates['factual_nll_degradation_vs_baseline_at_most'],
    }
    report = {
        'protocol': str(args.protocol),
        'baseline': str(args.baseline),
        'candidate': str(args.candidate),
        'n_complexes': candidate['n_complexes'],
        'measurements': {
            'mismatch': mismatch,
            'shuffled': shuffled,
            'shuffled_mean_abs_logit_change':
                shuffled_effects['mean_abs_logit_change']['mean'],
            'shuffled_argmax_hamming': shuffled_effects['argmax_hamming']['mean'],
            'factual_nll_degradation_vs_baseline': factual_degradation,
        },
        'checks': checks,
        'accepted': all(checks.values()),
        'failure_policy': protocol['failure_policy'],
    }
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite assessment: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if args.require_gates and not report['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
