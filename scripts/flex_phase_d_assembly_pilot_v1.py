"""FLEX-BINDER Phase D assembly pilot (record-level, 8 both-label antigens).

For each of the top-8 IEDB antigens (frozen in feasibility_report_v1): pull
positive and negative linear-epitope B-cell records, deduplicate sequences,
review antigen identity from the curated metadata, and attempt
composition-matched binder/non-binder pairing WITHIN the same source antigen
(the C1 lesson applied at data level: any discrimination claim must survive a
composition-matched control).

Boundary: assembly only. No model scoring, no candidate generation; the
family packages are inert data until the Phase D preregistration is frozen.

Pilot pass rule (declared before running): a family passes when it has
>= 2 distinct binder sequences, >= 1 distinct negative sequence, and
composition-match yield >= 0.5 over its distinct binders. All 8 passing =>
freeze-readiness signal; any failure mode => fix the rule, not the data.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_phase_d_v1'
API = 'https://query-api.iedb.org/bcell_search'
AA = set('ACDEFGHIKLMNPQRSTVWY')
LENGTH_FILTER = 'linear_sequence_length=gte.5&linear_sequence_length=lte.50'
NEG_PAGE_LIMIT = 10000
POS_PAGE_LIMIT = 10000
MAX_PERSISTED_PAIRS = 50
YIELD_GATE = 0.5
STANDARD_AA_ONLY = True


def fetch_all(base_query, page_limit, max_rows, order='bcell_id.asc'):
    rows = []
    offset = 0
    while offset < max_rows:
        url = f'{API}?{base_query}&order={order}&limit={page_limit}&offset={offset}'
        request = urllib.request.Request(
            url, headers={'User-Agent': 'DisorderFlow-phaseD-pilot/1',
                          'Accept': 'application/json'})
        last = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    page = json.loads(response.read())
                break
            except Exception as error:
                last = error
                time.sleep(min(2 ** attempt, 60))
        else:
            raise RuntimeError(f'{base_query} offset={offset}: {last}')
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        time.sleep(0.8)
        if len(page) < page_limit:
            break
    return rows


def standard_sequence(value):
    if not value:
        return None
    sequence = value.strip().upper()
    if STANDARD_AA_ONLY and set(sequence) - AA:
        return None
    return sequence if 5 <= len(sequence) <= 50 else None


def composition(sequence):
    return tuple(sorted(Counter(sequence).items()))


def assay_tally(rows):
    tally = Counter()
    for row in rows:
        text = (row.get('assay_description') or '').lower()
        for keyword in ('elisa', 'spr', 'kd', 'inhibition', 'immunization',
                        'western blot', 'competition', 'binding assay'):
            if keyword in text:
                tally[keyword] += 1
                break
        else:
            tally['other'] += 1
    return dict(tally)


def assemble_family(accession, negative_count):
    # accession IDs are alnum/.-_ and need no URL quoting; the ->> operator
    # must survive verbatim, so only the value is interpolated.
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
                        'organism': antigen.get('source_organism_name'),
                        'antigen_length_hint': antigen.get('ending_position')}
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
    used = set()
    for binder in sorted(binders):
        bucket = by_composition.get(composition(binder))
        if bucket:
            partner = bucket.pop()
            used.add(partner)
            matched_pairs.append({'binder': binder, 'negative': partner})
        else:
            unmatched.append(binder)
    yield_ratio = len(matched_pairs) / len(binders) if binders else 0.0

    return {
        'accession': accession,
        'identity': identity,
        'n_positive_records': len(positive_rows),
        'n_negative_records': len(negative_rows),
        'n_distinct_binders': len(binders),
        'n_distinct_negatives': len(negatives),
        'composition_matched_pairs': len(matched_pairs),
        'composition_match_yield': round(yield_ratio, 4),
        'assay_tally_positive': assay_tally(positive_rows),
        'assay_tally_negative': assay_tally(negative_rows),
        'matched_pairs_preview': matched_pairs[:MAX_PERSISTED_PAIRS],
        'unmatched_binders_preview': unmatched[:20],
        'passes_pilot': (len(binders) >= 2 and negatives
                         and yield_ratio >= YIELD_GATE),
    }


LENGTH_FILTER = LENGTH_FILTER  # noqa: F811 (kept at top; single definition)


def main():
    report = read(OUT / 'feasibility_report_v1.json')
    targets = report['top_antigens_with_both_labels']
    freeze(OUT / 'pilot_protocol.json', {
        'classification': 'assembly_pilot_no_model_scoring',
        'targets': [t['accession'] for t in targets],
        'pass_rule': {'min_distinct_binders': 2,
                      'min_distinct_negatives': 1,
                      'composition_match_yield': YIELD_GATE},
        'matching': 'exact multiset-composition pairing within the same source '
                    'antigen; negatives drawn from measured records, not shuffles',
        'boundary': 'family packages are inert until the Phase D preregistration '
                    'is frozen; no model, judge or generator is run on them',
        'script_sha256': digest(Path(__file__))})

    families = []
    for target in targets:
        accession = target['accession']
        family = assemble_family(accession, target['negative_records'])
        families.append(family)
        freeze(OUT / 'pilot_families' / f'{accession}.json', family)
        print(accession, 'binders', family['n_distinct_binders'],
              'negatives', family['n_distinct_negatives'],
              'yield', family['composition_match_yield'],
              'PASS' if family['passes_pilot'] else 'FAIL', flush=True)
        time.sleep(1.5)

    passing = [f for f in families if f['passes_pilot']]
    summary = {
        'families_attempted': len(families),
        'families_passed': len(passing),
        'mean_yield_passing': (round(sum(f['composition_match_yield']
                                         for f in passing) / len(passing), 4)
                               if passing else None),
        'total_distinct_binders': sum(f['n_distinct_binders'] for f in families),
        'total_matched_pairs': sum(f['composition_matched_pairs'] for f in families),
        'verdict': ('assembly_chain_works_proceed_to_full_curated_assembly'
                    if len(passing) == len(families)
                    else 'failure_modes_present_fix_rules'),
        'classification': 'pilot_summary_no_gate_claim',
    }
    freeze(OUT / 'pilot_summary.json', summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
