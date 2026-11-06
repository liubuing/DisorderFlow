"""FLEX-BINDER Phase D assembly pilot v2: window-deconfounded matching.

v1 failure mode (frozen in pilot_summary.json / pilot_families/): the top-8
both-label antigens are coronavirus replicase polyproteins covered by dense
overlapping-window scans; exact-composition matching pairs binders with their
own shifted windows (e.g. AAINSVPWSKIL <-> AINSVPWSKILA), which are
position-confounded controls, not composition-matched non-binders.

v2 rule (declared before rerun): a negative may only pair with a binder when
its Hamming distance to EVERY distinct binder in the family is
>= max(3, 40% of the sequence length). Composition identity is retained
(same multiset => same length), so the pair still controls for composition;
the distance constraint removes window-shift artifacts.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_phase_d_assembly_pilot_v1 import (
    AA, LENGTH_FILTER, NEG_PAGE_LIMIT, POS_PAGE_LIMIT, MAX_PERSISTED_PAIRS,
    YIELD_GATE, assay_tally, fetch_all, standard_sequence, composition)

OUT = ROOT / 'results/flex_phase_d_v1'


def min_hamming_to_binders(candidate, binders):
    """Minimum Hamming distance from candidate to any binder (same length)."""
    minimum = None
    for binder in binders:
        if len(binder) != len(candidate):
            continue
        distance = sum(1 for a, b in zip(binder, candidate) if a != b)
        if minimum is None or distance < minimum:
            minimum = distance
    return minimum


def assemble_family_v2(accession, negative_count):
    accession_filter = f'curated_source_antigen->>accession=eq.{accession}'
    label_filter = 'linear_sequence=not.is.null'
    positive_rows = fetch_all(
        f'qualitative_measure=eq.Positive&{LENGTH_FILTER}&{label_filter}&{accession_filter}',
        POS_PAGE_LIMIT, 40000)
    negative_rows = fetch_all(
        f'qualitative_measure=eq.Negative&{LENGTH_FILTER}&{label_filter}&{accession_filter}',
        NEG_PAGE_LIMIT, negative_count + NEG_PAGE_LIMIT)

    identity = None
    for row in positive_rows + negative_rows:
        antigen = row.get('curated_source_antigen') or {}
        if isinstance(antigen, dict) and antigen.get('accession') == accession:
            identity = {'accession': accession,
                        'name': antigen.get('name'),
                        'organism': antigen.get('source_organism_name')}
            break

    binders = {standard_sequence(r.get('linear_sequence'))
               for r in positive_rows}
    negatives = {standard_sequence(r.get('linear_sequence'))
                 for r in negative_rows}
    binders.discard(None)
    negatives.discard(None)
    negatives.difference_update(binders)

    by_composition = {}
    for sequence in negatives:
        by_composition.setdefault(composition(sequence), []).append(sequence)

    matched_pairs = []
    unmatched = []
    for binder in sorted(binders):
        floor = max(3, int(0.4 * len(binder)))
        bucket = by_composition.get(composition(binder))
        if not bucket:
            unmatched.append(binder)
            continue
        partner = None
        for candidate in bucket:
            distance = min_hamming_to_binders(candidate, binders)
            if distance is not None and distance >= floor:
                partner = candidate
                break
        if partner is None:
            unmatched.append(binder)
        else:
            bucket.remove(partner)
            matched_pairs.append({'binder': binder, 'negative': partner,
                                  'min_hamming_to_binders': distance})
    yield_ratio = len(matched_pairs) / len(binders) if binders else 0.0
    distances = [p['min_hamming_to_binders'] for p in matched_pairs]

    return {
        'accession': accession,
        'identity': identity,
        'rule': 'composition-exact AND min Hamming to every binder >= max(3, 0.4*len)',
        'n_positive_records': len(positive_rows),
        'n_negative_records': len(negative_rows),
        'n_distinct_binders': len(binders),
        'n_distinct_negatives': len(negatives),
        'composition_matched_pairs': len(matched_pairs),
        'composition_match_yield': round(yield_ratio, 4),
        'matched_pair_distance_median': (float(Counter(distances).most_common(1)[0][0])
                                         if distances else None),
        'assay_tally_positive': assay_tally(positive_rows),
        'matched_pairs_preview': matched_pairs[:MAX_PERSISTED_PAIRS],
        'passes_pilot': (len(binders) >= 2 and negatives
                         and yield_ratio >= YIELD_GATE),
    }


def main():
    report = read(OUT / 'feasibility_report_v1.json')
    targets = report['top_antigens_with_both_labels']
    freeze(OUT / 'pilot_v2_protocol.json', {
        'classification': 'assembly_pilot_v2_window_deconfounded_no_model_scoring',
        'v1_failure_mode': 'overlapping-window scans make exact-composition '
                           'negatives position-confounded (shifted windows of '
                           'the binder itself)',
        'v2_rule': 'negative must be composition-exact AND Hamming '
                   '>= max(3, 0.4*len) from every distinct binder',
        'targets': [t['accession'] for t in targets],
        'pass_rule': {'min_distinct_binders': 2,
                      'composition_match_yield': YIELD_GATE},
        'boundary': 'assembly only; family packages inert until prereg freeze',
        'script_sha256': digest(Path(__file__))})

    families = []
    for target in targets:
        accession = target['accession']
        family = assemble_family_v2(accession, target['negative_records'])
        families.append(family)
        freeze(OUT / 'pilot_families_v2' / f'{accession}.json', family)
        print(accession, 'binders', family['n_distinct_binders'],
              'yield', family['composition_match_yield'],
              'PASS' if family['passes_pilot'] else 'FAIL', flush=True)
        time.sleep(1.5)

    passing = [f for f in families if f['passes_pilot']]
    summary = {
        'families_attempted': len(families),
        'families_passed': len(passing),
        'mean_yield': (round(sum(f['composition_match_yield'] for f in families)
                             / len(families), 4) if families else None),
        'total_matched_pairs': sum(f['composition_matched_pairs'] for f in families),
        'verdict': ('assembly_chain_works_proceed_to_full_curated_assembly'
                    if len(passing) >= 6
                    else 'top_volume_antigens_are_window_scans_select_diverse_antigens'),
        'reading': 'the v1->v2 delta measures how much of the apparent both-label '
                   'volume is window-scan artifact; family selection for the full '
                   'assembly must not be ordered by record volume',
        'classification': 'pilot_v2_summary_no_gate_claim',
    }
    freeze(OUT / 'pilot_v2_summary.json', summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
