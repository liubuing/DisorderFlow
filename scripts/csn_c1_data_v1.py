"""CSN Phase C1 (v1) data acquisition: parent accessions, IEDB metadata counts,
parent sequences, ATLAS availability. Freeze prerequisite for the C1 prereg.

Read-only cross-campaign reuse: ESEL E0b accession/counts for the 8 anchored
rows are imported as-is (disclosed); only E0b's 6 unresolved rows are re-run
through the same curation ladder here, with a CSN-local cache so ESEL's frozen
artifact tree is never written to.

Red lines (inherited): IEDB access stays metadata-level (counts + query
strings, zero record-level sequences on disk); every network response is
cached write-once and resume-able; nothing here reads outcome labels.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402

QUEUE = ROOT / 'results/flex_j0_v1/preregistration.json'
E0B_MAP = ROOT / 'results/esel_e0b_v1/accession_map.json'
E0B_COUNTS = ROOT / 'results/esel_e0b_v1/iedb_counts.json'
OUT = ROOT / 'results/csn_c1_v1'
CACHE = OUT / 'cache'

IEDB_API = 'https://query-api.iedb.org/bcell_search'
RCSB_ENTRY = 'https://data.rcsb.org/rest/v1/core/entry/{}'
RCSB_ENTITY = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}'
UNIPROT_FASTA = 'https://rest.uniprot.org/uniprotkb/{}.fasta'
UNIPROT_PEPTIDE_SEARCH = 'https://peptidesearch.uniprot.org/asyncrest'
ATLAS_API = 'https://www.dsimb.inserm.fr/ATLAS/api'
AA = set('ACDEFGHIKLMNPQRSTVWY')
DATE = '2026-11-08'
CLASSIFICATION = 'csn_c1_freeze_prerequisite_data_not_labels'
# Mechanics adapted from scripts/esel_e0b_curation_v1.py (ESEL campaign);
# caches redirected to the CSN tree on purpose.


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def http_get(url: str, headers: dict | None = None, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - resume-able retry ladder
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f'GET {url} failed after {retries} attempts: {last}')


def cached_json(path: Path, producer):
    if path.exists():
        return read_json(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = producer()
    freeze(path, payload)
    return payload


def cached_text(path: Path, producer) -> str:
    if path.exists():
        return path.read_text(encoding='utf-8', errors='replace')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = producer()
    path.write_text(payload, encoding='utf-8')
    time.sleep(1.0)
    return payload


def rcsb_entry(pdb_id: str):
    def produce():
        entry = json.loads(http_get(RCSB_ENTRY.format(pdb_id)))
        antigen = []
        for entity_id in entry.get('rcsb_entry_container_identifiers', {}).get(
                'polymer_entity_ids', []):
            entity = json.loads(http_get(RCSB_ENTITY.format(pdb_id, entity_id)))
            identifiers = entity.get('rcsb_polymer_entity_container_identifiers', {})
            antigen.append({
                'entity_id': entity_id,
                'uniprot_ids': identifiers.get('uniprot_ids') or [],
                'length': entity.get('entity_poly', {}).get('pdbx_seq_one_letter_length'),
                'description': entity.get('rcsb_polymer_entity', {}).get('description', ''),
            })
        return {'pdb_id': pdb_id, 'polymer_entities': antigen}
    return cached_json(CACHE / 'rcsb' / f'{pdb_id}.json', produce)


def uniprot_fasta_sequence(accession: str) -> str:
    def produce():
        return http_get(UNIPROT_FASTA.format(accession)).decode('utf-8', errors='replace')
    text = cached_text(CACHE / 'fasta' / f'{accession}.fasta', produce)
    return ''.join(line.strip() for line in text.splitlines()
                   if not line.startswith('>'))


def uniprot_peptide_search(peptide: str) -> list:
    """Exact substring search across UniProtKB (async API: submit, then poll
    the Location URL; mechanics mirrored from scripts/esel_e0b_curation_v1.py)."""
    def produce():
        data = urllib.parse.urlencode(
            {'peqs': peptide, 'lEQi': 'off', 'spOnly': 'off'}).encode()
        request = urllib.request.Request(
            UNIPROT_PEPTIDE_SEARCH, data=data,
            headers={'User-Agent': 'csn-c1/1',
                     'Content-Type': 'application/x-www-form-urlencoded'})
        with urllib.request.urlopen(request, timeout=30) as response:
            poll_url = response.headers.get('Location')
        if not poll_url:
            raise RuntimeError(f'peptide search did not return a poll URL for {peptide}')
        for _ in range(30):
            time.sleep(3)
            try:
                with urllib.request.urlopen(poll_url, timeout=30) as poll:
                    body = poll.read().decode('utf-8', 'replace').strip()
                if body:
                    return sorted({line.split('|')[0] if '|' in line else line
                                   for line in body.splitlines() if line.strip()})
            except Exception:  # noqa: BLE001 - poll retries are expected
                continue
        return []
    return cached_json(CACHE / 'peptidesearch' / f'{peptide}.json', produce)


def iedb_counts(accession: str):
    def count_query(query: str):
        body = http_get(f'{IEDB_API}?{query}&limit=1',
                        headers={'Accept': 'application/json'})
        return json.loads(body).get('paging', {}).get('total', 0)

    def produce():
        # E0b ladder: curated accessions carry sequence versions; bare IDs return 0.
        counts = {}
        for label, query in [
            ('versioned', f'structure_group.parent.protein.accession="{accession}"'),
            ('bare', f'parent.protein.accession="{accession}"'),
        ]:
            counts[label] = count_query(query)
        return counts
    return cached_json(CACHE / 'iedb' / f'{accession}.json', produce)


def atlas_search(sequence: str, key: str):
    def produce():
        data = urllib.parse.urlencode({'sequence': sequence[:1000], 'header': key,
                                       'dataset': 'ATLAS'}).encode()
        request = urllib.request.Request(
            f'{ATLAS_API}/sequence-search', data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except Exception as error:  # noqa: BLE001 - 404 means no hit, not failure
            if '404' in str(error):
                return {'sequence_search_results': []}
            raise
    return cached_json(CACHE / 'atlas_search' / f'{key}.json', produce)


def resolve_accession(pdb_id: str, peptide: str):
    """E0b-style ladder: RCSB entity UniProt ids first, peptide-search fallback,
    containment verification (parent peptide must be a substring of the FASTA)."""
    entry = rcsb_entry(pdb_id)
    entities = entry['polymer_entities']
    # Antigen candidate entities: prefer short ones (peptides) whose UniProt
    # list is nonempty; the binder heavy chain is much longer.
    candidates = []
    for ent in entities:
        for acc in ent['uniprot_ids']:
            candidates.append((ent, acc))
    candidates.sort(key=lambda pair: pair[0].get('length') or 10**6)
    trials = []
    for ent, acc in candidates:
        seq = uniprot_fasta_sequence(acc)
        ok = bool(seq) and peptide in seq
        trials.append({'accession': acc, 'entity_id': ent['entity_id'],
                       'length': ent.get('length'), 'contains_peptide': ok})
        if ok:
            return {'status': 'anchored', 'accession': acc, 'trials': trials}
    # Fallback: UniProt peptide search, containment-verified.
    try:
        hits = uniprot_peptide_search(peptide)
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        hits = []
        trials.append({'peptide_search_error': repr(error)})
    for hit in hits[:5]:
        acc = hit if isinstance(hit, str) else hit.get('accession') or hit.get('uniprot_id')
        if not acc:
            continue
        try:
            seq = uniprot_fasta_sequence(acc)
        except Exception:  # noqa: BLE001
            continue
        ok = bool(seq) and peptide in seq
        trials.append({'accession': acc, 'source': 'uniprot_peptide_search',
                       'contains_peptide': ok})
        if ok:
            return {'status': 'anchored', 'accession': acc, 'trials': trials}
    return {'status': 'unresolved', 'trials': trials}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comps = read_json(QUEUE)['cohort']['components']
    e0b_map = read_json(E0B_MAP)['mapping']
    e0b_counts = read_json(E0B_COUNTS)['counts_by_accession']

    by_pdb = {}
    for c in comps:
        by_pdb.setdefault(c['pdb_id'], c['peptide_sequence'])

    mapping = {}
    for pdb_id, peptide in sorted(by_pdb.items()):
        instance = next(c['instance'] for c in comps if c['pdb_id'] == pdb_id)
        e0b_rows = [m for inst, m in e0b_map.items() if inst.startswith(pdb_id)]
        e0b_anchored = next((m for m in e0b_rows if m.get('status') == 'anchored'), None)
        if e0b_anchored:
            mapping[pdb_id] = {
                'status': 'anchored', 'source': 'esel_e0b_v1_readonly',
                'accession': e0b_anchored.get('anchor_accession'),
                'peptide': peptide, 'instance_example': instance}
            continue
        print(f'resolving {pdb_id} ({peptide})...', flush=True)
        mapping[pdb_id] = {'source': 'csn_c1_ladder', 'peptide': peptide,
                           'instance_example': instance,
                           **resolve_accession(pdb_id, peptide)}
        time.sleep(1.0)

    # IEDB metadata counts: reuse E0b where present, ladder for new accessions.
    counts = {}
    for pdb_id, m in mapping.items():
        acc = m.get('accession')
        if not acc:
            counts[pdb_id] = {'accession': None, 'status': 'no_accession'}
            continue
        if acc in e0b_counts:
            counts[pdb_id] = {'accession': acc, 'status': 'esel_e0b_v1_readonly',
                              **e0b_counts[acc]}
            continue
        c = iedb_counts(acc)
        counts[pdb_id] = {'accession': acc, 'status': 'csn_c1_ladder', **c}
        time.sleep(1.0)

    # Parent sequences + ATLAS availability per resolved accession.
    parents = {}
    for pdb_id, m in mapping.items():
        acc = m.get('accession')
        if not acc:
            continue
        seq = uniprot_fasta_sequence(acc)
        search = atlas_search(seq, acc)
        hits = search.get('sequence_search_results', [])
        parents[acc] = {
            'length': len(seq),
            'sequence_sha256': __import__('hashlib').sha256(seq.encode()).hexdigest(),
            'atlas_hits': len(hits),
            'atlas_best': (max((float(h.get('identity', 0)) for h in hits), default=None)),
            'peptide_offset': (seq.find(m['peptide']) if m['peptide'] in seq else None),
        }
        time.sleep(1.0)

    freeze(OUT / 'accession_map_csn.json', {
        'classification': CLASSIFICATION, 'date': DATE, 'mapping': mapping})
    freeze(OUT / 'iedb_counts_csn.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'metadata_only_note': 'counts and query strings only; zero record-level data',
        'counts_by_pdb': counts})
    freeze(OUT / 'parents.json', {
        'classification': CLASSIFICATION, 'date': DATE, 'parents': parents})
    anchored = {p: m['accession'] for p, m in mapping.items() if m.get('accession')}
    freeze(OUT / 'summary.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'n_pdbs': len(mapping), 'n_anchored': len(anchored),
        'anchored_accessions': anchored,
        'n_with_iedb_counts': len([c for c in counts.values() if c.get('accession')]),
        'n_parents_with_atlas_hits': len([p for p in parents.values()
                                          if (p['atlas_hits'] or 0) > 0]),
        'e0b_reuse_note': ('E0b anchored rows imported read-only; only the 6 unresolved '
                           'rows rerun through the ladder with a CSN-local cache'),
    })
    freeze(OUT / 'completion.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'inputs': {'queue_sha256': digest(QUEUE),
                   'e0b_map_sha256': digest(E0B_MAP),
                   'e0b_counts_sha256': digest(E0B_COUNTS)},
        'script_sha256': digest(Path(__file__)),
        'artifacts': ['accession_map_csn.json', 'iedb_counts_csn.json',
                      'parents.json', 'summary.json', 'completion.json']})
    print('C1 data frozen:', len(anchored), '/', len(mapping), 'PDBs anchored;',
          sum(1 for p in parents.values() if p['atlas_hits']), 'parents with ATLAS hits')
    for pdb_id, m in mapping.items():
        print(f"  {pdb_id}: {m.get('status')} {m.get('accession')}")


if __name__ == '__main__':
    main()
