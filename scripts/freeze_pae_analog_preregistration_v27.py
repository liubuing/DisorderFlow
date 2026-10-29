"""Freeze the machine-readable v27 analog preregistration.

Must run after docs/PAE_ANALOG_PREREGISTRATION_V27.md is final and before any
candidate generation or AF2 teacher run. The frozen numbers come from the
completed v23-v27 gates; the document is the authoritative statement.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/pae_analog_preregistration_v27'
DOC = ROOT / 'docs/PAE_ANALOG_PREREGISTRATION_V27.md'
V26 = ROOT / 'results/pae_analog_structures_v26'
V27 = ROOT / 'results/pae_analog_family_split_v27'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v26 = read(V26 / 'summary.json')
    v27 = read(V27 / 'summary.json')
    split = read(V27 / 'split.json')
    payload = {
        'classification': 'preregistration_freeze_before_generation_and_labels',
        'frozen_utc': '2026-09-30',
        'document': 'docs/PAE_ANALOG_PREREGISTRATION_V27.md',
        'document_sha256': digest(DOC),
        'gate_chain_sha256': {
            'v23_audit': digest(ROOT / 'results/pae_analog_audit_v23/audit.json'),
            'v24_audit': digest(ROOT / 'results/pae_analog_chain_identity_v24/audit.json'),
            'v25_audit': digest(ROOT / 'results/pae_analog_overlap_v25/audit.json'),
            'v26_manifest': digest(V26 / 'manifest.json'),
            'v27_split': digest(V27 / 'split.json'),
        },
        'cohort': {
            'v26_components_passed': v26['components_passed'],
            'v26_by_cohort': v26['by_cohort'],
            'confirmatory': v27['split']['tcr_confirmatory_components'],
            'vhh_pilot': v27['split']['vhh_pilot_components'],
            'eval_components': sorted(split['eval_components']),
            'eval_families': split['eval_families'],
            'train_components': sorted(split['train_components']),
            'train_families': v27['split']['train_families'],
        },
        'design': {
            'eval_candidates_per_component': 14, 'eval_controls_per_component': 2,
            'train_candidates_per_component': 7,
            'replicates_per_entity': 6,
            'af2': 'AF2-Multimer v3 models 1+2, seeds 7103/7111/7121, recycles 3, single-sequence MSA, no templates',
            'estimand': 'mean normalized directional candidate-region-to-peptide PAE /31',
            'generation': 'ProteinMPNN v_48_020 soluble, temperature 0.3, seed 20260919, design chain only',
            'primary_endpoint': 'component-equal recall of teacher top-20% at 20% budget (3/14)',
            'model': 'StandardScaler + Ridge(alpha=10) on train components only',
            'total_af2_slots': 6 * 16 * 6 + 12 * 7 * 6,
            'claim_boundary': 'analog transfer; no binding, affinity, speedup or architecture-superiority claim',
        },
        'deviation_log': 'results/pae_analog_deviation_log/',
    }
    freeze(OUT / 'preregistration.json', payload)
    print(json.dumps({'frozen': str(OUT / 'preregistration.json'),
                      'document_sha256': payload['document_sha256']}, indent=1))


if __name__ == '__main__':
    main()
