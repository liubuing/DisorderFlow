"""FLEX-BINDER Phase C1 boundary decomposition (post-processing only).

No new model inference: everything is derived from frozen caches
(flex_j0a/j0b scoring, flex_c1 scoring, MPNN NLL cache, composition cache,
A1b identity review). Three questions:

1. Where does the native actually rank? Percentile of the native among
   (i) the 50 shuffles only, (ii) the sampled designs only, (iii) the full
   pool - per method. This turns "recognizes the native but never top-1"
   into separate numbers per competitor class.
2. What does the judge's score correlate with? Per-component Spearman between
   the judge ECLS and ProteinMPNN NLL / composition probability across the
   unique pool candidates.
3. Who are the antigens? Family -> member PDB -> deposited title mapping,
   with the seq_4 / seq_1 anomalies called out.

Descriptive characterization for the boundary repositioning; no gates.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_a2_atlas_nmr_audit_v1 import average_ranks

C1 = ROOT / 'results/flex_c1_v1'
J0 = ROOT / 'results/flex_j0_v1'
J0B = ROOT / 'results/flex_j0b_v1'


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def percentile(native, reference, scores, better):
    ref = [scores[c] for c in reference if c in scores]
    if not ref:
        return None
    native_score = scores[native]
    if better == 'lower':
        worse = sum(1 for s in ref if s > native_score)
        tied = sum(1 for s in ref if s == native_score)
    else:
        worse = sum(1 for s in ref if s < native_score)
        tied = sum(1 for s in ref if s == native_score)
    return (worse + 0.5 * tied) / len(ref)


def judge_marginal_scores(entry, c1_scoring, j0a_row, j0b_row):
    """Judge marginal ECLS for the full 101-candidate pool."""
    stripped = dict(j0a_row['nll']['stripped'])
    stripped.update(c1_scoring['stripped_for_designs'])
    complex_dicts = list(j0b_row['nll_complex_per_conformer'])
    designs_complex = list(c1_scoring['complex_per_conformer'])
    merged = []
    for base, extra in zip(complex_dicts, designs_complex):
        merged_dict = dict(base)
        merged_dict.update(extra)
        merged.append(merged_dict)
    pool = set(stripped)
    for d in merged:
        pool &= set(d)
    return {c: float(np.mean([d[c] for d in merged])) - stripped[c]
            for c in pool}


def main():
    prereg = read(C1 / 'preregistration.json')
    mpnn_cache = read(C1 / 'mpnn_nll_cache.json')
    composition_cache = read(C1 / 'composition_prob_cache.json')
    identity = read(ROOT / 'results/flex_a1b_isolation_families_v1/identity_review.json')

    rows = []
    for component in sorted(prereg['components'], key=lambda c: c['instance']):
        entry = component
        instance = entry['instance']
        native = entry['region_sequence']
        designs = read(C1 / 'generation' / instance / 'candidates.json')['designs']
        unique_designs = sorted(set(designs))
        j0a_row = read(J0 / 'scoring' / f'{instance}.json')
        j0b_row = read(J0B / 'scoring' / f'{instance}.json')
        c1_scoring = read(C1 / 'scoring' / f'{instance}.json')
        judge = judge_marginal_scores(entry, c1_scoring, j0a_row, j0b_row)
        mpnn_scores = mpnn_cache[instance]
        comp_scores = composition_cache.get(instance) or {}

        row = {'instance': instance, 'family': entry['family'],
               'pdb': entry['pdb_id'], 'title': identity.get(entry['pdb_id'], {}).get('title'),
               'native': native,
               'judge_percentile': {}, 'mpnn_percentile': {},
               'composition_percentile': {}}
        for label, reference in (('shuffles', entry['shuffles']),
                                 ('designs', unique_designs),
                                 ('full_pool', list(judge.keys()))):
            row['judge_percentile'][label] = percentile(native, reference, judge, 'lower')
            row['mpnn_percentile'][label] = percentile(native, reference, mpnn_scores, 'lower')
            row['composition_percentile'][label] = percentile(
                native, reference, comp_scores, 'higher')
        pool_candidates = sorted(judge)
        row['spearman_judge_vs_mpnn'] = spearman(
            [judge[c] for c in pool_candidates],
            [mpnn_scores[c] for c in pool_candidates])
        row['spearman_judge_vs_composition'] = spearman(
            [judge[c] for c in pool_candidates],
            [comp_scores[c] for c in pool_candidates])
        rows.append(row)
        print(instance, 'judge shuffles %.2f designs %.2f pool %.2f' % (
            row['judge_percentile']['shuffles'],
            row['judge_percentile']['designs'],
            row['judge_percentile']['full_pool']), flush=True)

    def family_means(key):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['family']].append(row[key])
        return {k: float(np.mean(v)) for k, v in grouped.items()}

    # Family means computed per reference set and per method, explicitly.
    family_tables = {}
    for method in ('judge_percentile', 'mpnn_percentile', 'composition_percentile'):
        for label in ('shuffles', 'designs', 'full_pool'):
            key = (method, label)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row['family']].append(row[method][label])
            family_tables['/'.join(key)] = {
                k: float(np.mean(v)) for k, v in sorted(grouped.items())}
    family_spearman = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['family']].append(row['spearman_judge_vs_mpnn'])
    for k, v in grouped.items():
        family_spearman[k] = float(np.mean([x for x in v if x is not None]))

    summary = {
        'classification': 'descriptive_boundary_characterization_no_gates',
        'n_components': len(rows),
        'n_families': len({r['family'] for r in rows}),
        'judge_native_percentile_family_means': family_tables['judge_percentile/shuffles'],
        'note_shuffles_vs_designs': family_tables,
        'judge_vs_mpnn_spearman_family_means': family_spearman,
        'provenance': {
            'c1_prereg_sha256': digest(C1 / 'preregistration.json'),
            'script_sha256': digest(Path(__file__)),
        },
    }
    freeze(C1 / 'boundary_decomposition.json', {'components': rows})
    freeze(C1 / 'boundary_summary.json', summary)
    print(json.dumps({
        'judge/shuffles family means': summary['judge_native_percentile_family_means'],
        'judge/designs family means': family_tables['judge_percentile/designs'],
    }, indent=1), flush=True)


if __name__ == '__main__':
    main()
