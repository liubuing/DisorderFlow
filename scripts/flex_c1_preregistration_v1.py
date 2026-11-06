"""FLEX-BINDER Phase C1 preregistration: native-recognition ranker comparison.

Frozen BEFORE any candidate generation or scoring. Task: for each of the 26
J0b-valid components, rank a candidate pool of 101 H3 sequences
{native, 50 ProteinMPNN designs, 50 preregistered composition shuffles} and
measure where the deposited native lands. This extends the J0 question
("native vs composition shuffles") to "native vs real outsourced competitors"
using the J0b-validated ensemble-marginalized judge.

Main comparison table (M8 discipline): the judge must beat the composition
baseline and ProteinMPNN NLL, not just random, to keep Phase C standing.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_c1_v1'

METHODS = {
    'flex_ecls_marginal': {
        'definition': 'ECLS_marginal from the J0b judge (4-conformer complex NLL '
                      'minus stripped NLL, 3 scoring seeds averaged); rank ascending '
                      '(lower ECLS = more native-like)',
        'status': 'primary judge under test',
    },
    'composition_logistic': {
        'definition': 'logistic regression on 20-dim amino-acid composition of the '
                      'candidate region (+ no length feature; pools have equal length '
                      'within a component), trained native=1 vs shuffles=0 with '
                      'leave-one-family-out CV, C=1.0, lbfgs, seed 20261029; rank by '
                      'predicted native probability descending',
        'status': 'M8 simple-baseline control',
    },
    'proteinmpnn_nll': {
        'definition': 'ProteinMPNN conditional NLL of each candidate in the complex '
                      'state (peptide-present backbone, H3 positions only), from one '
                      'conditional_probs_only pass per component; rank ascending',
        'status': 'outsourced-generator self-likelihood; known self-preference bias '
                  'is expected to favor its own designs over the native and is '
                  'disclosed rather than corrected',
    },
    'random': {
        'definition': 'expected percentile 0.5 / expected top-1 rate 1/(pool size)',
        'status': 'floor',
    },
}

CANDIDATE_GENERATION = {
    'generator': 'ProteinMPNN vanilla v_48_020 (ProteinMPNN/protein_mpnn_run.py)',
    'mode': 'sequence sampling, fixed backbone',
    'fixed': 'all heavy-chain residues except the CDR-H3 region; all peptide residues',
    'context': 'H and P chains parsed (peptide-present complex state)',
    'n_designs_per_component': 50,
    'temperature': 0.2,
    'seed_rule': '8161 + component alphabetical index (deterministic)',
    'dedup': 'none; all 50 sampled designs enter the pool, duplicates allowed',
    'pool': 'native + 50 designs + the 50 preregistered J0 shuffles (101 candidates)',
}


def main():
    build = read(ROOT / 'results/flex_j0b_v1/build_summary.json')
    cohort = read(ROOT / 'results/flex_j0_v1/cohort.json')
    prereg_j0 = read(ROOT / 'results/flex_j0_v1/preregistration.json')
    j0b_results = read(ROOT / 'results/flex_j0b_v1/j0b_results.json')['components']
    family_of = {row['instance']: row['family'] for row in j0b_results}
    entries = {e['instance']: e for e in cohort['components']}
    components = []
    for index, row in enumerate(sorted(
            (r for r in build['rows'] if r['valid']),
            key=lambda r: r['instance'])):
        entry = entries[row['instance']]
        components.append({
            'instance': entry['instance'],
            'pdb_id': entry['pdb_id'],
            'component_pdb': entry['component_pdb'],
            'region_spec': entry['region_spec'],
            'region_sequence': entry['region_sequence'],
            'peptide_sequence': entry['peptide_sequence'],
            'family': family_of[entry['instance']],
            'mpnn_seed': 8161 + index,
            'shuffles': entry['shuffles'],
        })

    protocol = {
        'classification': 'development_ranker_comparison_not_confirmatory_claim',
        'task': 'native-recognition ranking: locate the deposited native H3 within '
                'a 101-candidate pool of outsourced designs and composition shuffles',
        'n_components': len(components),
        'n_families': len({c['family'] for c in components}),
        'components': components,
        'candidate_generation': CANDIDATE_GENERATION,
        'methods': METHODS,
        'endpoints': {
            'primary': 'native mean percentile within pool (0 = native best; '
                       'ties counted 0.5), per component, family means, '
                       'family-clustered bootstrap 95% CI (10,000 draws, seed 20261029)',
            'secondary_descriptive': 'top-1 and top-5 native rates per method',
        },
        'gates': {
            'primary': 'judge mean percentile beats composition_logistic AND '
                       'proteinmpnn_nll: per family, mean paired component '
                       'percentile difference (judge minus baseline) < 0 in >= 70% '
                       'of families for BOTH baselines, AND the judge pooled '
                       'percentile family-bootstrap CI upper bound < 0.5 (better '
                       'than random)',
            'on_fail': 'M8 clause: reposition as benchmark/boundary study; no '
                       'further Phase C development claims',
        },
        'power_statement': {
            'reading': '26 components / 8 families; all comparisons are '
                       'development-level descriptive with paired component '
                       'differences; no confirmatory or multiple-comparison '
                       'significance claims',
        },
        'reuse_disclosure': 'native and shuffle scores are reused bit-identically '
                            'from the J0a/J0b frozen caches (complex per-conformer '
                            'and stripped); only the 50 MPNN designs per component '
                            'are newly scored',
        'provenance': {
            'j0b_build_sha256': digest(ROOT / 'results/flex_j0b_v1/build_summary.json'),
            'j0_cohort_sha256': digest(ROOT / 'results/flex_j0_v1/cohort.json'),
            'j0b_results_sha256': digest(ROOT / 'results/flex_j0b_v1/j0b_results.json'),
            'script_sha256': digest(Path(__file__)),
        },
    }
    freeze(OUT / 'preregistration.json', protocol)
    print(json.dumps({'n_components': len(components),
                      'n_families': protocol['n_families'],
                      'families': sorted({c['family'] for c in components})}, indent=1))


if __name__ == '__main__':
    main()
