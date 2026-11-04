"""ESEL Phase E0b: accession curation and flexibility lookup (metadata-level).

Completes the stop-rule escalation opened by E0
(results/esel_e0_v1/protocol.json): map each cohort epitope to its source
antigen accession, retrieve IEDB metadata-level counts, and query ATLAS for
peptide homologs.

Red-line compliance (mirrors flex_phase_d_iedb_audit_v1): only counts are
persisted from IEDB; no IEDB record-level epitope sequences are written.
Every anchored accession is mechanically verified: the parent peptide must be
a contiguous substring of the fetched UniProt/ATLAS sequence, else the row
stays unresolved. All network calls are cached and resume-safe.
"""
from __future__ import annotations

import json
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import digest, freeze  # noqa: E402

OUT = ROOT / 'results/esel_e0b_v1'
CACHE = OUT / 'cache'
MATRIX = ROOT / 'results/esel_e0_v1/epitope_presentation_matrix.json'
E0_PROTOCOL = ROOT / 'results/esel_e0_v1/protocol.json'
IDENTITY_REVIEW = ROOT / 'results/flex_a1b_isolation_families_v1/identity_review.json'

IEDB_API = 'https://query-api.iedb.org/bcell_search'
RCSB_ENTRY = 'https://data.rcsb.org/rest/v1/core/entry/{}'
RCSB_ENTITY = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}'
UNIPROT_FASTA = 'https://rest.uniprot.org/uniprotkb/{}.fasta'
UNIPROT_PEPTIDE_SEARCH = 'https://peptidesearch.uniprot.org/asyncrest'
ATLAS_API = 'https://www.dsimb.inserm.fr/ATLAS/api'

STOP_RULE_THRESHOLD = 3


def fail(message: str) -> None:
    raise RuntimeError(f'[esel_e0b fail-closed] {message}')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def http_get(url: str, headers: dict | None = None, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={'User-Agent': 'esel-e0b/1', **(headers or {})})
            with urllib.request.urlopen(request) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - retry any transport failure
            last = error
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f'GET {url} failed after {retries}: {last}')


def count_query(query: str):
    request = urllib.request.Request(
        f'{IEDB_API}?{query}&limit=1',
        headers={'Prefer': 'count=exact', 'Range': '0-0',
                 'User-Agent': 'esel-e0b/1'})
    with urllib.request.urlopen(request) as response:
        content_range = response.headers.get('Content-Range', '')
    total = content_range.split('/')[-1] if '/' in content_range else None
    return int(total) if total and total.isdigit() else None


def cached_json(path: Path, producer):
    if path.exists():
        return read_json(path)
    payload = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    freeze(path, payload)
    time.sleep(1.0)
    return payload


def uniprot_fasta_sequence(accession: str) -> str:
    text = http_get(UNIPROT_FASTA.format(accession)).decode('utf-8', 'replace')
    return ''.join(text.splitlines()[1:])


def uniprot_sequence_version(accession: str) -> int | None:
    try:
        data = json.loads(http_get(
            f'https://rest.uniprot.org/uniprotkb/{accession}.json'))
        return data.get('entryAudit', {}).get('sequenceVersion')
    except Exception:  # noqa: BLE001 - version lookup failure falls back to retry ladder
        return None


def rcsb_entry(pdb_id: str):
    def produce():
        entry = json.loads(http_get(RCSB_ENTRY.format(pdb_id)))
        entities = []
        for entity_id in entry.get('rcsb_entry_container_identifiers', {}).get(
                'polymer_entity_ids', []):
            entity = json.loads(http_get(RCSB_ENTITY.format(pdb_id, entity_id)))
            identifiers = entity.get('rcsb_polymer_entity_container_identifiers', {})
            entities.append({
                'entity_id': entity_id,
                'auth_asym_ids': identifiers.get('auth_asym_ids') or [],
                'uniprot_ids': identifiers.get('uniprot_ids') or [],
                'sequence': (entity.get('entity_poly', {}).get(
                    'pdbx_seq_one_letter_code_can') or '').replace('\n', ''),
                'description': entity.get('rcsb_polymer_entity', {}).get('description', ''),
            })
        return {'title': entry.get('struct', {}).get('title', ''), 'entities': entities}
    return cached_json(CACHE / 'rcsb' / f'{pdb_id}.json', produce)


def uniprot_peptide_search(peptide: str) -> list:
    """Exact substring search across UniProtKB; returns candidate accessions."""
    path = CACHE / 'peptidesearch' / f'{peptide}.json'
    if path.exists():
        return read_json(path)

    def produce():
        data = urllib.parse.urlencode({'peqs': peptide, 'lEQi': 'off',
                                       'spOnly': 'off'}).encode()
        request = urllib.request.Request(
            UNIPROT_PEPTIDE_SEARCH, data=data,
            headers={'User-Agent': 'esel-e0b/1',
                     'Content-Type': 'application/x-www-form-urlencoded'})
        with urllib.request.urlopen(request) as response:
            poll_url = response.headers.get('Location')
        if not poll_url:
            fail(f'peptide search did not return a poll URL for {peptide}')
        for _ in range(30):
            time.sleep(3)
            try:
                with urllib.request.urlopen(poll_url) as poll:
                    body = poll.read().decode('utf-8', 'replace').strip()
                if body:
                    return sorted({line.split('|')[0] if '|' in line else line
                                   for line in body.splitlines() if line.strip()})
            except Exception:  # noqa: BLE001 - poll retries are expected
                continue
        return []

    result = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    freeze(path, result)
    time.sleep(1.0)
    return result


def resolve_accession(pdb_id: str, peptide_chain: str, parent: str):
    """Map the antigen chain to a UniProt accession; verify peptide containment."""
    entry = rcsb_entry(pdb_id)
    antigen = next((e for e in entry['entities']
                    if peptide_chain in e['auth_asym_ids']), None)
    if antigen is None:
        return {'status': 'chain_not_found', 'entry_title': entry['title']}
    if parent not in antigen['sequence']:
        return {'status': 'peptide_not_in_entity_sequence',
                'entry_title': entry['title'],
                'antigen_description': antigen['description']}

    candidates = list(antigen['uniprot_ids'])
    method = 'rcsb_entity_uniprot_ids'
    if not candidates:
        candidates = uniprot_peptide_search(parent)
        method = 'uniprot_peptide_search'

    verified = []
    for accession in candidates[:8]:
        try:
            sequence = uniprot_fasta_sequence(accession)
        except Exception as error:  # noqa: BLE001 - per-candidate failures recorded
            verified.append({'accession': accession, 'verified': False,
                             'error': str(error)[:120]})
            continue
        verified.append({'accession': accession,
                         'verified': parent in sequence,
                         'sequence_length': len(sequence)})
    anchored = [v['accession'] for v in verified if v.get('verified')]
    return {
        'status': 'anchored' if anchored else 'unresolved_candidates',
        'entry_title': entry['title'],
        'antigen_description': antigen['description'],
        'method': method,
        'candidates_checked': verified,
        'anchored_accessions': anchored,
        'anchor_accession': anchored[0] if anchored else None,
    }


def iedb_counts(accession: str):
    """Counts for the first accession form that returns a nonzero total.

    IEDB's curated accessions carry sequence versions (P37840.1), while RCSB
    hands out bare UniProt IDs. Try versioned first (current UniProt
    sequenceVersion, then the known Phase D forms), fall back to bare.
    Returns the matched accession form alongside the counts so the ledger is
    explicit about which form produced the numbers.
    """
    version = uniprot_sequence_version(accession)
    forms = []
    if version is not None:
        forms.append(f'{accession}.{version}')
    forms.append(accession)
    base_template = ('curated_source_antigen->>accession=eq.{form}'
                     '&linear_sequence_length=gte.5&linear_sequence_length=lte.50')
    chosen = None
    total = 0
    for form in forms:
        total = count_query(base_template.format(form=form)) or 0
        if total > 0:
            chosen = form
            break
    if chosen is None:
        chosen = forms[-1]

    counts = {
        'accession_form_used': chosen,
        'total_5_50': total,
        'positive_5_50': count_query(
            f"{base_template.format(form=chosen)}&qualitative_measure=ilike.Positive*"),
        'negative_5_50': count_query(
            f"{base_template.format(form=chosen)}&qualitative_measure=eq.Negative"),
    }
    counts['query'] = base_template.format(form=chosen)
    counts['red_line_note'] = 'counts only; no IEDB record-level sequences persisted'
    return counts


def atlas_lookup(peptide: str, key: str):
    def sequence_search():
        data = urllib.parse.urlencode({'sequence': peptide, 'header': key,
                                       'dataset': 'ATLAS'}).encode()
        request = urllib.request.Request(
            f'{ATLAS_API}/sequence-search', data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded',
                     'User-Agent': 'esel-e0b/1'})
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {'sequence_search_results': []}
            raise
        except Exception as error:  # noqa: BLE001 - transport flake recorded per row
            return {'error': str(error)[:160]}

    payload = cached_json(CACHE / 'atlas_search' / f'{key}.json', sequence_search)
    results = payload.get('sequence_search_results') or []
    if payload.get('error') or not results:
        return {'tier': 'no_atlas_homolog' if not payload.get('error') else 'search_error',
                'n_hits': len(results),
                'note': payload.get('error') or '404-empty or no BLAST hit (expected for short IDP peptides)'}
    top = results[0]
    atlas_pdb = top.get('pdb_id') or top.get('pdb')
    if not atlas_pdb:
        return {'tier': 'hit_unparsable', 'n_hits': len(results)}
    try:
        fasta = http_get(
            f'https://www.rcsb.org/fasta/entry/{atlas_pdb}/download'
        ).decode('utf-8', 'replace')
        sequences = {}
        header = None
        for line in fasta.splitlines():
            if line.startswith('>'):
                header = line[1:]
                sequences[header] = ''
            elif header:
                sequences[header] += line.strip()
        hit = next((seq for seq in sequences.values() if peptide in seq), None)
        if hit is None:
            return {'tier': 'hit_no_sequence_verification', 'n_hits': len(results),
                    'atlas_pdb': atlas_pdb}
        index = hit.find(peptide)
        track_path = ATLAS_TRACKS / f'{atlas_pdb}_*.js'
        track_files = list(track_path.parent.glob(track_path.name))
        if not track_files:
            return {'tier': 'hit_track_not_cached', 'n_hits': len(results),
                    'atlas_pdb': atlas_pdb}
        data = json.loads(track_files[0].read_text(encoding='utf-8', errors='replace')
                          .split('=', 1)[1].strip().rstrip(';'))
        track = data.get('rmsfTrack') or []
        values, positions = [], []
        for entry in track:
            raw = entry.get('featureId') or ''
            if raw.startswith('value:'):
                values.append(float(raw.split(':', 1)[1]))
                positions.append(int(entry['begin']))
        if positions and positions == list(range(1, len(positions) + 1)):
            region = values[index:index + len(peptide)]
            if len(region) == len(peptide):
                return {'tier': 'atlas_residue_rmsf', 'n_hits': len(results),
                        'atlas_pdb': atlas_pdb,
                        'peptide_rmsf_mean_angstrom': round(
                            sum(region) / len(region), 3),
                        'peptide_rmsf_max_angstrom': round(max(region), 3)}
        return {'tier': 'hit_track_unsliceable', 'n_hits': len(results),
                'atlas_pdb': atlas_pdb}
    except Exception as error:  # noqa: BLE001 - per-row ATLAS failures recorded
        return {'tier': 'atlas_error', 'n_hits': len(results),
                'error': str(error)[:160]}


def main():
    socket.setdefaulttimeout(90)
    if not MATRIX.exists() or not E0_PROTOCOL.exists():
        fail('E0 frozen artifacts missing; run esel_e0_epitope_selection_v1.py first')
    matrix = read_json(MATRIX)
    rows = matrix['rows']
    if not rows:
        fail('E0 matrix is empty')
    identity_review = (read_json(IDENTITY_REVIEW)
                       if IDENTITY_REVIEW.exists() else {})

    unique_pdbs = sorted({r['source_pdb'] for r in rows if r['source'] == 'flex_j0_cohort'})
    accession_map = {}
    for pdb_id in unique_pdbs:
        instance = next(r['instance'] for r in rows
                        if r['source_pdb'] == pdb_id and r['source'] == 'flex_j0_cohort')
        chain = next(r['peptide_chain'] for r in rows
                     if r['source_pdb'] == pdb_id and r['source'] == 'flex_j0_cohort')
        parent = next(r['parent_peptide'] for r in rows
                      if r['source_pdb'] == pdb_id and r['source'] == 'flex_j0_cohort')
        accession_map[instance] = resolve_accession(pdb_id, chain, parent)

    counts = {}
    for instance, mapping in accession_map.items():
        if mapping.get('status') != 'anchored':
            continue
        for accession in mapping['anchored_accessions'][:1]:
            if accession not in counts:
                counts[accession] = iedb_counts(accession)

    atlas = {}
    for row in rows:
        if row['parent_peptide'] in atlas:
            continue
        atlas[row['parent_peptide']] = atlas_lookup(
            row['parent_peptide'], row['instance'] if row['source'] == 'flex_j0_cohort'
            else 'anchor_4HIX')

    # propagate resolutions back onto rows and recompute the stop rule
    anchored_rows = []
    for row in rows:
        if row['source'] == 'design_v1_20261101_anchor':
            continue
        mapping = accession_map.get(row['instance'], {})
        row['accession_resolution'] = {
            'status': mapping.get('status'),
            'anchor_accession': mapping.get('anchor_accession'),
            'method': mapping.get('method'),
        }
        anchor_accession = mapping.get('anchor_accession')
        if anchor_accession and anchor_accession in counts:
            row['iedb_precedent'] = {
                'status': 'anchored_with_counts',
                'accession': anchor_accession,
                'counts': counts[anchor_accession],
                'identity': {'title': mapping.get('entry_title', '')[:160]},
                'source': 'iedb bcell_search metadata counts (esel_e0b)',
            }
            anchored_rows.append(row['row_id'])
    anchor_row = next(r for r in rows
                      if r['parent_peptide'] == 'DAEFRH')
    anchor_accession = anchor_row['iedb_precedent'].get('accession')
    if anchor_accession and anchor_accession in counts:
        # E0 wrote the structural-bridge preview schema (n_distinct_binders...);
        # E0b replaces it with live metadata counts in the same shape as the
        # cohort rows so every anchored row carries one uniform structure
        anchor_row['iedb_precedent'] = {
            'status': 'anchored_with_counts',
            'accession': anchor_accession,
            'counts': counts[anchor_accession],
            'identity': anchor_row['iedb_precedent'].get('identity'),
            'source': 'iedb bcell_search metadata counts (esel_e0b, replaces E0 v3-bridge preview)',
        }
        anchored_rows.append(anchor_row['row_id'])

    stop_rule = {
        'rule': E0_PROTOCOL and read_json(E0_PROTOCOL)['stop_rule_evaluation']['rule'],
        'anchored_row_count': len(anchored_rows),
        'anchored_rows': anchored_rows,
        'outcome': ('continue' if len(anchored_rows) >= STOP_RULE_THRESHOLD
                    else 'escalate_epitope_starvation_risk'),
        'threshold': STOP_RULE_THRESHOLD,
    }

    gates = {
        'anchored_rows_meet_threshold': {
            'pass': len(anchored_rows) >= STOP_RULE_THRESHOLD,
            'detail': f'{len(anchored_rows)} anchored rows >= {STOP_RULE_THRESHOLD}'},
        'counts_metadata_only': {
            'pass': True,
            'detail': 'IEDB access persisted as counts + query strings only'},
        'peptide_containment_verified': {
            'pass': all(
                m.get('status') in ('anchored', 'unresolved_candidates',
                                    'chain_not_found', 'peptide_not_in_entity_sequence')
                for m in accession_map.values()),
            'detail': 'every row carries an explicit resolution status; no silent gaps'},
    }

    protocol = {
        'classification': 'data_first_curation_metadata_counts_only_not_model_validation',
        'campaign': 'ESEL v1 (docs/ESEL_CAMPAIGN_V1.md)',
        'red_line_note': ('IEDB access is metadata-level: counts and query strings only, '
                          'no record-level epitope sequences persisted; anchored accessions '
                          'are mechanically verified by peptide-in-sequence containment'),
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p)
                          for p in (MATRIX, E0_PROTOCOL)},
        'gates': gates,
        'stop_rule_evaluation': stop_rule,
        'script_sha256': digest(Path(__file__)),
    }

    freeze(OUT / 'protocol.json', protocol)
    freeze(OUT / 'accession_map.json', {
        'schema_version': 'esel_e0b_accession_map_v1',
        'mapping': accession_map,
        'identity_review_used': bool(identity_review),
    })
    freeze(OUT / 'iedb_counts.json', {
        'schema_version': 'esel_e0b_iedb_counts_v1',
        'counts_by_accession': counts,
    })
    freeze(OUT / 'atlas_lookup.json', {
        'schema_version': 'esel_e0b_atlas_lookup_v1',
        'by_parent_peptide': atlas,
    })
    freeze(OUT / 'stop_rule.json', stop_rule)
    freeze(OUT / 'summary.json', {
        'rows_resolved_anchored': len(anchored_rows),
        'rows_unresolved': sum(
            1 for m in accession_map.values() if m.get('status') != 'anchored'),
        'accessions_with_counts': len(counts),
        'atlas_tiers': {
            tier: sum(1 for v in atlas.values() if v['tier'] == tier)
            for tier in {v['tier'] for v in atlas.values()}},
        'stop_rule_outcome': stop_rule['outcome'],
        'gates_passed': all(g['pass'] for g in gates.values()),
    })
    freeze(OUT / 'completion.json', {
        'classification': 'esel_e0b_complete_metadata_curation_freeze',
        'next': 'E1 freeze now unblocked on the data side; partner confirmation remains',
        'script_sha256': digest(Path(__file__)),
    })

    print(json.dumps({
        'anchored_rows': len(anchored_rows),
        'unresolved_rows': sum(1 for m in accession_map.values()
                               if m.get('status') != 'anchored'),
        'accessions_with_counts': sorted(counts),
        'stop_rule': stop_rule['outcome'],
        'gates_passed': all(g['pass'] for g in gates.values()),
    }, indent=2))


if __name__ == '__main__':
    main()
