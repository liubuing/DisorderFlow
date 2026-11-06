"""FLEX-BINDER Phase D control-pool builder (A' fork, rule-preserving).

Streams the full IEDB B-cell negative universe (linear 5-50 aa) and persists
the distinct standard-AA sequence pool used for composition-exact matching.
This does NOT relax the frozen v3 matching rule: same rule, larger pool.

Resume-able via checkpoint (offset + partial pool) every 20 pages. Red-line
note: sequences enter an inert control pool for assembly; no model scoring
before the Phase D preregistration is frozen.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_phase_d_assembly_pilot_v1 import standard_sequence

OUT = ROOT / 'results/flex_phase_d_v1'
API = 'https://query-api.iedb.org/bcell_search'
LENGTH = 'linear_sequence_length=gte.5&linear_sequence_length=lte.50'
PAGE = 10000
CHECKPOINT_EVERY = 20


def fetch_page(offset, retries=6):
    url = (f'{API}?qualitative_measure=eq.Negative&{LENGTH}'
           f'&linear_sequence=not.is.null'
           f'&select=linear_sequence&order=bcell_id.asc'
           f'&limit={PAGE}&offset={offset}')
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={'User-Agent': 'DisorderFlow-phaseD-pool/1',
                              'Accept': 'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())
        except Exception as error:
            last = error
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f'offset={offset}: {last}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / 'negative_pool_checkpoint.json'
    pool = {}
    offset = 0
    pages_since_checkpoint = 0
    if checkpoint.exists():
        state = read(checkpoint)
        pool = state['pool']
        offset = state['offset']
        print(f'resuming at offset {offset}, pool {len(pool)}', flush=True)
    else:
        total = fetch_page(0)
        del total

    while True:
        rows = fetch_page(offset)
        if not rows:
            break
        for row in rows:
            sequence = standard_sequence(row.get('linear_sequence'))
            if sequence and sequence not in pool:
                pool[sequence] = 0
            if sequence:
                pool[sequence] += 1
        offset += len(rows)
        pages_since_checkpoint += 1
        if pages_since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint.write_text(json.dumps({'offset': offset, 'pool': pool}),
                                  encoding='utf-8')
            pages_since_checkpoint = 0
            print(f'offset {offset}, distinct {len(pool)}', flush=True)
        time.sleep(0.3)

    checkpoint.write_text(json.dumps({'offset': offset, 'pool': pool}),
                          encoding='utf-8')
    from collections import Counter
    compositions = {}
    for sequence in pool:
        key = ''.join(f'{aa}{n},' for aa, n in sorted(Counter(sequence).items()))
        compositions.setdefault(key, []).append(sequence)
    result = {
        'classification': 'inert_control_pool_no_model_scoring',
        'source_query': f'IEDB B-cell negatives, {LENGTH}',
        'n_records_streamed': offset,
        'n_distinct_sequences': len(pool),
        'n_distinct_compositions': len(compositions),
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'negative_pool_summary.json', result)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
