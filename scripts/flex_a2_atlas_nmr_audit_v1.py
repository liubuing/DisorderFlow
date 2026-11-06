"""FLEX-BINDER Phase A2 (ATLAS side): MD RMSF vs NMR-ensemble RMSF audit.

For each local NMR ensemble (flex_a2_nmr_rmsf_v1), find the best ATLAS protein
by sequence search, pair residues by global alignment of the ATLAS chain
sequence (RCSB FASTA), and compute within-protein rank agreement between
C-alpha NMR RMSF and ATLAS MD RMSF (3x100 ns replicates, per-residue mean).

Verdict gate (frozen before the run): median within-protein Spearman >= 0.30
labels ATLAS RMSF admissible as the replacement flexibility label source; the
AF2-seed RMSF proxy is then downgraded per the v5.1 rank-only convention.
Label infrastructure only: no training, no design, no claims about binders.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

NMR = ROOT / 'results/flex_a2_nmr_rmsf_v1/rmsf.json'
OUT = ROOT / 'results/flex_a2_atlas_nmr_audit_v1'
API = 'https://www.dsimb.inserm.fr/ATLAS/api'
REFERER = 'https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{pdb_id}/{pdb_id}.html'
USER_AGENT = 'DisorderFlow-flex-a2-label-audit/1'

MIN_IDENTITY = 90.0
MIN_QUERY_COVERAGE = 0.8
MIN_ALIGNED = 50
SPEARMAN_GATE = 0.30  # frozen: v5.1 pass_physical_label threshold


def http_get(url, referer=None, timeout=90, retries=4):
    headers = {'User-Agent': USER_AGENT}
    if referer:
        headers['Referer'] = referer
        headers['X-Requested-With'] = 'XMLHttpRequest'
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f'GET {url} failed after {retries} attempts: {last_error}')


def http_post(url, data, headers=None, timeout=120, retries=4):
    base = {'User-Agent': USER_AGENT}
    if headers:
        base.update(headers)
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data, headers=base)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f'POST {url} failed after {retries} attempts: {last_error}')


def sequence_search(sequence, pdb_key):
    """Best ATLAS hit for a sequence; resume-able raw JSON cache per entry."""
    path = OUT / 'search' / f'{pdb_key}.json'
    if path.exists():
        return read(path)
    data = urllib.parse.urlencode({'sequence': sequence, 'header': pdb_key,
                                   'dataset': 'ATLAS'}).encode()
    try:
        body = http_post(
            f'{API}/sequence-search', data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'})
    except RuntimeError as error:
        # The server answers 404 when BLAST produces no reportable hit at all;
        # that is an empty result, not a transport failure.
        if '404' in str(error):
            body = b'{"sequence_search_results": []}'
        else:
            raise
    payload = json.loads(body)
    freeze(path, payload)
    time.sleep(2.0)
    return payload


def fetch_fasta(pdb_id):
    path = OUT / 'fasta' / f'{pdb_id}.fasta'
    if not path.exists():
        payload = http_get(f'https://www.rcsb.org/fasta/entry/{pdb_id}/download')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        time.sleep(1.5)
    return path.read_text(encoding='utf-8', errors='replace')


def parse_fasta_chain(text, chain):
    """RCSB FASTA headers look like >{pdb}_{entity}|Chain {chain}(s)...; pick
    the entry whose chain list contains the requested auth chain."""
    blocks = []
    header = None
    lines = []
    for line in text.splitlines():
        if line.startswith('>'):
            if header is not None:
                blocks.append((header, ''.join(lines)))
            header, lines = line, []
        else:
            lines.append(line.strip())
    if header is not None:
        blocks.append((header, ''.join(lines)))
    for block_header, seq in blocks:
        if f'|Chain {chain}[' in block_header + '[' or f'|Chain {chain}|' in block_header \
                or block_header.endswith(f'Chain {chain}') or f'Chain {chain}(' in block_header:
            return seq
    return None


def fetch_tracks(pdb_id):
    path = OUT / 'tracks' / f'{pdb_id}.js'
    if not path.exists():
        payload = http_get(f'{API}/tracks/{pdb_id}.js',
                           referer=REFERER.format(pdb_id=pdb_id))
        if not payload.startswith(b'window.trackData'):
            raise RuntimeError(f'tracks payload for {pdb_id} is not trackData')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        time.sleep(1.5)
    text = path.read_text(encoding='utf-8', errors='replace')
    return json.loads(text.split('=', 1)[1].strip().rstrip(';'))


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        stop = index
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        ranks[order[index:stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    return ranks


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def global_pairs(seq_a, seq_b):
    """Global alignment index pairs via Bio.Align.PairwiseAligner (BLOSUM62)."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    alignment = aligner.align(seq_a, seq_b)[0]
    pairs = []
    for (a_start, a_end), (b_start, b_end) in zip(alignment.aligned[0],
                                                  alignment.aligned[1]):
        for offset in range(int(a_end) - int(a_start)):
            pairs.append((int(a_start) + offset, int(b_start) + offset))
    return pairs, alignment.score


def top_recall(flex_a, flex_b, fraction=0.2):
    """Fraction of top-flexible residues in A recalled inside the top set of B."""
    n = max(1, int(round(fraction * len(flex_a))))
    top_a = set(np.argsort(flex_a)[-n:])
    top_b = set(np.argsort(flex_b)[-n:])
    return len(top_a & top_b) / n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = read(NMR)['entries']
    protocol = {
        'classification': 'flexibility_label_audit_not_model_result',
        'nmr_source_sha256': digest(NMR),
        'match_rules': {'min_identity': MIN_IDENTITY,
                        'min_query_coverage': MIN_QUERY_COVERAGE,
                        'min_aligned_residues': MIN_ALIGNED},
        'verdict_gate': {'metric': 'median within-protein Spearman(NMR RMSF, ATLAS RMSF)',
                         'threshold': SPEARMAN_GATE,
                         'provenance': 'v5.1 pass_physical_label threshold'},
        'atlas_rmsf': 'rmsfTrack from the per-protein tracks endpoint (mean over 3 replicates, Angstrom)',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)

    audits = []
    for entry in entries:
        key = entry['pdb']
        sequence = entry['sequence']
        search = sequence_search(sequence, key)
        hits = search.get('sequence_search_results', [])
        eligible = []
        for hit in hits:
            identity = float(hit['identity'])
            coverage = int(hit['alignment_length']) / max(1, len(sequence))
            if identity >= MIN_IDENTITY and coverage >= MIN_QUERY_COVERAGE:
                eligible.append(hit)
        if not eligible:
            audits.append({'pdb': key, 'status': 'no_atlas_match',
                           'best_identity': max((float(h['identity']) for h in hits), default=None)})
            continue
        best = max(eligible, key=lambda h: float(h['identity']))
        target_pdb, target_chain = best['pdb_id'].split('_')
        try:
            chain_seq = parse_fasta_chain(fetch_fasta(target_pdb), target_chain)
            if not chain_seq:
                audits.append({'pdb': key, 'status': 'fasta_chain_not_found',
                               'target': best['pdb_id']})
                continue
            tracks = fetch_tracks(best['pdb_id'])
            rmsf_track = tracks['rmsfTrack']
            begins = np.array([t['begin'] for t in rmsf_track])
            if not np.all(np.diff(begins) == 1) or begins[0] != 1:
                audits.append({'pdb': key, 'status': 'track_positions_not_contiguous',
                               'target': best['pdb_id']})
                continue
            atlas_rmsf = np.array([float(t['value']) for t in rmsf_track])
            pairs, score = global_pairs(sequence, chain_seq)
            if len(pairs) < MIN_ALIGNED:
                audits.append({'pdb': key, 'status': 'too_few_aligned_residues',
                               'target': best['pdb_id'], 'aligned': len(pairs)})
                continue
            nmr_values, atlas_values = [], []
            for a_index, b_index in pairs:
                atlas_position = b_index + 1  # tracks are 1-based over the chain sequence
                if 1 <= atlas_position <= len(atlas_rmsf):
                    nmr_values.append(entry['rmsf'][a_index])
                    atlas_values.append(atlas_rmsf[atlas_position - 1])
            rho = spearman(nmr_values, atlas_values)
            audits.append({
                'pdb': key, 'status': 'audited', 'target': best['pdb_id'],
                'identity': best['identity'],
                'aligned_pairs': len(pairs), 'alignment_score': score,
                'spearman': rho,
                'recall_nmr_top20_in_atlas': top_recall(np.array(nmr_values),
                                                        np.array(atlas_values)),
                'recall_atlas_top20_in_nmr': top_recall(np.array(atlas_values),
                                                        np.array(nmr_values)),
            })
        except Exception as error:
            audits.append({'pdb': key, 'status': f'error: {error}',
                           'target': best['pdb_id']})
        print(f'[{len(audits)}/{len(entries)}] {key} {audits[-1]["status"]}', flush=True)

    audited = [a for a in audits if a['status'] == 'audited']
    rhos = np.array([a['spearman'] for a in audited if a['spearman'] is not None])
    verdict = {
        'nmr_entries': len(entries),
        'audited': len(audited),
        'rho_median': float(np.median(rhos)) if len(rhos) else None,
        'rho_iqr': [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))] if len(rhos) else None,
        'fraction_rho_ge_gate': float(np.mean(rhos >= SPEARMAN_GATE)) if len(rhos) else None,
        'median_recall_nmr_top20_in_atlas': float(np.median([a['recall_nmr_top20_in_atlas']
                                                             for a in audited])) if audited else None,
        'gate_threshold': SPEARMAN_GATE,
        'verdict': None,
    }
    verdict['verdict'] = ('pass_physical_label' if verdict['rho_median'] is not None
                          and verdict['rho_median'] >= SPEARMAN_GATE else 'fail_physical_label')
    freeze(OUT / 'audits.json', {'audits': audits})
    freeze(OUT / 'summary.json', verdict)
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == '__main__':
    main()
