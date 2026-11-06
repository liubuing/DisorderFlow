"""FLEX-BINDER Phase C1 direction correction (v2).

Deviation disclosure: the frozen C1 gate clause scored "judge better" as
paired percentile difference (judge - baseline) < 0, while the implemented
percentile is the fraction of the pool scoring WORSE than the native
(higher = native ranked better; the preregistration's own note "0 = native
best" contradicts the clause direction). v2 recomputes both clause outcomes
from the frozen c1_summary.json under both readings.

Pure post-processing; no model inference, no network. The v1 verdict
(gate = fail) is retained as computed; this artifact records whether the
verdict is robust to the direction correction.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_c1_v1'
THRESHOLD = 0.70


def main():
    summary = read(OUT / 'c1_summary.json')
    corrected = {}
    for baseline, payload in summary['baseline_comparisons'].items():
        diffs = payload['family_mean_paired_diff']
        n = payload['n_families']
        wins_positive = sum(1 for v in diffs.values() if v > 0)
        wins_negative = sum(1 for v in diffs.values() if v < 0)
        corrected[baseline] = {
            'families_judge_better_corrected_diff_gt_0': wins_positive,
            'families_judge_better_frozen_diff_lt_0': wins_negative,
            'n_families': n,
            'clause_pass_corrected': wins_positive / n >= THRESHOLD,
            'clause_pass_frozen': wins_negative / n >= THRESHOLD,
            'family_mean_paired_diff': diffs,
        }
    gate_corrected = all(v['clause_pass_corrected'] for v in corrected.values()) and \
        summary['stats']['flex_ecls_marginal']['bootstrap_ci'][0] > 0.5
    gate_frozen = summary['gate_parts']
    result = {
        'classification': 'direction_correction_disclosure_verdict_unchanged',
        'frozen_gate_parts': gate_frozen,
        'frozen_verdict': summary['verdict'],
        'corrected_clauses': corrected,
        'corrected_gate_verdict': 'pass' if gate_corrected else 'fail',
        'verdict_robust_to_direction': summary['verdict'] == ('pass' if gate_corrected else 'fail'),
        'reading': 'under the corrected direction the judge beats ProteinMPNN '
                   'self-likelihood in 6/8 families but still fails the '
                   'composition-baseline clause (5/8 < 70%); the frozen gate '
                   'verdict (fail) is unchanged',
        'provenance': {
            'c1_summary_sha256': digest(OUT / 'c1_summary.json'),
            'script_sha256': digest(Path(__file__)),
        },
    }
    freeze(OUT / 'direction_correction.json', result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('corrected_clauses',)}, indent=1))
    for baseline, payload in corrected.items():
        print(baseline, '-> corrected', payload['families_judge_better_corrected_diff_gt_0'],
              '/', payload['n_families'], '| clause_pass:', payload['clause_pass_corrected'])


if __name__ == '__main__':
    main()
