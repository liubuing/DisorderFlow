"""FLEX-BINDER Phase D termination artifact (decision B, 2026-11-02).

Terminates the Phase D binder-vs-nonbinder line. Records the decision, the
measured rationale (A' falsified), and what survives (the boundary-study
manuscript). Read-only consolidation of frozen artifacts; no new data.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_phase_d_v1'


def main():
    evidence = {
        'feasibility_report_v1': digest(OUT / 'feasibility_report_v1.json'),
        'iedb_feasibility_audit': digest(OUT / 'iedb_feasibility_audit.json'),
        'pilot_v1_summary': digest(OUT / 'pilot_summary.json'),
        'pilot_v2_summary': digest(OUT / 'pilot_v2_summary.json'),
        'pilot_v3_summary': digest(OUT / 'pilot_v3_summary.json'),
        'pilot_v3b_summary': digest(OUT / 'pilot_v3b_summary.json'),
        'negative_pool_summary': digest(OUT / 'negative_pool_summary.json'),
        'power_table': digest(OUT / 'power_table.json'),
    }
    result = {
        'decision': 'B_terminate_phase_d',
        'date': '2026-11-02',
        'classification': 'line_termination_decision_with_measured_rationale',
        'rationale': [
            'The preregistration freeze condition requires >=24 curated families '
            'with composition-matched experimentally-confirmed non-binders.',
            'Exact multiset-composition matching is measurably infeasible at any '
            'reachable pool scale: the full IEDB negative universe (1,234,987 '
            'records -> 709,890 distinct sequences, 672,336 distinct '
            'compositions, mean 1.056 sequences per composition) raises the '
            'matched yield only from 0.96% to 1.63% (1/200 families pass; '
            'pilot_v3b_summary.json).',
            'Loosening to nearest-neighbour composition matching would weaken '
            'the control below the level the C1 ranking result already showed '
            'is decisive (a composition baseline beat the judge in 3/8 '
            'families), so a Phase D positive result under relaxed matching '
            'would not survive its own baseline.',
            'Volume is not the constraint: 30,388 distinct antigens carry '
            'negative measurements, and the top-volume antigens are dense '
            'overlapping-window scans whose negatives are position-confounded '
            '(pilot v1/v2); per-antibody families do not exist in public '
            'B-cell data (12 receptor-annotated negatives vs 4,160 positives).',
        ],
        'what_survives': [
            'The boundary-study manuscript '
            '(publication/flex_boundary_manuscript_draft_v1.md), which now also '
            'carries the measured data-infeasibility of the binder-vs-nonbinder '
            'extension.',
            'The frozen Phase D audit chain (this directory), including the '
            'power table any future re-opening must meet (>=24 families; '
            'results/flex_phase_d_v1/power_table.json).',
        ],
        'reopen_condition': 'a new experimental negative dataset (e.g. curated '
                            'DMS or Proteome-scans with per-mutant binding '
                            'readouts) that changes the composition-collision '
                            'mathematics; re-opening requires a fresh '
                            'preregistration frozen before any new data contact.',
        'evidence_sha256': evidence,
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'termination.json', result)
    print(json.dumps({'decision': result['decision'],
                      'reopen_condition': result['reopen_condition']}, indent=1))


if __name__ == '__main__':
    main()
