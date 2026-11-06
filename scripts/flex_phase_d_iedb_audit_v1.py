"""FLEX-BINDER Phase D metadata-level feasibility audit (IEDB leg).

Aggregate-count audit against the IEDB query API (PostgREST style). Red-line
compliance: only counts and distinct-name tallies are persisted; no epitope
sequences, no record-level label tables are written to disk. Antigen names are
streamed through distinct-counters and discarded.

Questions answered (metadata only):
  I1 linear-epitope negative records (5-50 aa)
  I2 of those, with a curated source antigen (family anchor capacity)
  I3 of those, distinct curated source antigens (upper bound on families)
  I4 positive records with PDB and 5-50 aa (structure-anchored binders)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import freeze, digest

OUT = ROOT / 'results/flex_phase_d_v1'
API = 'https://query-api.iedb.org/bcell_search'
LENGTH = 'linear_sequence_length=gte.5&linear_sequence_length=lte.50'


def count(query):
    request = urllib.request.Request(
        f'{API}?{query}&limit=1',
        headers={'Prefer': 'count=exact', 'Range': '0-0',
                 'User-Agent': 'DisorderFlow-phaseD-feasibility/1'})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_range = response.headers.get('Content-Range', '')
    total = content_range.split('/')[-1] if '/' in content_range else None
    return int(total) if total and total.isdigit() else None


def fetch_rows(query, offset, limit=10000, retries=6):
    import time
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f'{query}&limit={limit}&offset={offset}',
                headers={'User-Agent': 'DisorderFlow-phaseD-feasibility/1',
                         'Accept': 'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())
        except Exception as error:
            last = error
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f'page offset={offset} failed after {retries}: {last}')


def antigen_key(value):
    """curated_source_antigen is a nested object (accession/name/organism)."""
    if isinstance(value, dict):
        return value.get('accession') or value.get('name')
    return value


def stream_distinct(query, field, top_tail=20):
    """Stream one field and persist only distinct-count + top names."""
    distinct = set()
    tally = Counter()
    offset = 0
    ordered = f'{API}?{query}&select={field}&order=bcell_id.asc'
    while True:
        rows = fetch_rows(ordered, offset)
        if not rows:
            break
        for row in rows:
            key = antigen_key(row.get(field))
            if key:
                distinct.add(key)
                tally[key] += 1
        offset += len(rows)
        import time
        time.sleep(0.3)
        if offset % 200_000 == 0:
            print(f'  streamed {offset} rows, distinct {len(distinct)}', flush=True)
    return len(distinct), tally.most_common(top_tail)


def main():
    counts = {}
    counts['I1_negative_linear_5_50'] = count(
        f'qualitative_measure=eq.Negative&{LENGTH}')
    counts['I2_negative_linear_5_50_with_antigen'] = count(
        f'qualitative_measure=eq.Negative&{LENGTH}&curated_source_antigen=not.is.null')
    counts['I4_positive_linear_5_50_with_pdb'] = count(
        f'qualitative_measure=eq.Positive&{LENGTH}&pdb_id=not.is.null')
    counts['I5_positive_linear_5_50'] = count(
        f'qualitative_measure=eq.Positive&{LENGTH}')
    counts['I6_total_bcell_records'] = count('')
    for key, value in counts.items():
        print(key, '=', value, flush=True)

    distinct_antigens, top_antigens = stream_distinct(
        f'qualitative_measure=eq.Negative&{LENGTH}&curated_source_antigen=not.is.null',
        'curated_source_antigen')
    counts['I3_distinct_negative_source_antigens'] = distinct_antigens

    result = {
        'classification': 'metadata_level_feasibility_counts_no_records_persisted',
        'red_line_note': 'antigen names were streamed through distinct counters and '
                         'discarded; only counts and top-name tallies are stored; no '
                         'epitope sequences or record-level label tables are persisted',
        'counts': counts,
        'top_negative_source_antigens_by_records': top_antigens,
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'iedb_feasibility_audit.json', result)
    print(json.dumps({k: v for k, v in result.items()
                      if k != 'top_negative_source_antigens_by_records'}, indent=1))


if __name__ == '__main__':
    main()
