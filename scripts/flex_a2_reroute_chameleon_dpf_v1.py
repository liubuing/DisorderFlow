"""FLEX-BINDER Phase A2 reroute leg 1: extend the MD-vs-NMR paired audit to the
Chameleon (16 proteins) and DPF (100 proteins) datasets of the same ATLAS
service, per the preregistered handling of the 2/129 pairing shortfall.

Same match rules and gates as flex_a2_atlas_nmr_audit_v1 (identity >= 90,
query coverage >= 0.8, aligned >= 50). The verdict here is descriptive:
paired-count-first, no gate claim below a meaningful n.
"""
import json
import sys
import time
import urllib.parse
import urllib.error
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_a2_atlas_nmr_audit_v1 import (
    MIN_ALIGNED, MIN_IDENTITY, MIN_QUERY_COVERAGE, SPEARMAN_GATE,
    average_ranks, fetch_fasta, global_pairs, spearman, top_recall)
import urllib.request

OUT = ROOT / 'results/flex_a2_reroute_chameleon_dpf_v1'
NMR = ROOT / 'results/flex_a2_nmr_rmsf_v1/rmsf.json'
API = 'https://www.dsimb.inserm.fr/ATLAS/api'
USER_AGENT = 'DisorderFlow-flex-a2-reroute/1'


def sequence_search(sequence, key, dataset):
    path = OUT / 'search' / dataset / f'{key}.json'
    if path.exists():
        return read(path)
    data = urllib.parse.urlencode({'sequence': sequence, 'header': key,
                                   'dataset': dataset}).encode()
    last = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                f'{API}/sequence-search', data=data,
                headers={'User-Agent': USER_AGENT,
                         'Content-Type': 'application/x-www-form-urlencoded'})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:
                payload = {'sequence_search_results': []}
                break
            last = error
            time.sleep(2 ** attempt)
        except Exception as error:
            last = error
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f'search failed for {key}/{dataset}: {last}')
    path.parent.mkdir(parents=True, exist_ok=True)
    freeze(path, payload)
    time.sleep(1.5)
    return payload


def fetch_tracks(pdb_id, dataset):
    path = OUT / 'tracks' / dataset / f'{pdb_id}.js'
    if not path.exists():
        referer = (f'https://www.dsimb.inserm.fr/ATLAS/database/'
                   f'{dataset}/{pdb_id}/{pdb_id}.html')
        request = urllib.request.Request(
            f'{API}/tracks/{pdb_id}.js',
            headers={'User-Agent': USER_AGENT, 'Referer': referer,
                     'X-Requested-With': 'XMLHttpRequest'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = response.read()
                break
            except Exception as error:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if not payload.startswith(b'window.trackData'):
            raise RuntimeError(f'tracks payload for {pdb_id} ({dataset}) is not trackData')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        time.sleep(1.5)
    text = path.read_text(encoding='utf-8', errors='replace')
    return json.loads(text.split('=', 1)[1].strip().rstrip(';'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = read(NMR)['entries']
    freeze(OUT / 'protocol.json', {
        'classification': 'descriptive_paired_extension_not_gate_claim',
        'datasets': ['chameleon', 'DPF'],
        'match_rules': {'min_identity': MIN_IDENTITY,
                        'min_query_coverage': MIN_QUERY_COVERAGE,
                        'min_aligned_residues': MIN_ALIGNED},
        'nmr_source_sha256': digest(NMR),
        'script_sha256': digest(Path(__file__))})

    audits = []
    for entry in entries:
        for dataset in ('chameleon', 'DPF'):
            key = f"{entry['pdb']}_{dataset}"
            hits = sequence_search(entry['sequence'], key, dataset).get(
                'sequence_search_results', [])
            eligible = [h for h in hits
                        if float(h['identity']) >= MIN_IDENTITY
                        and int(h['alignment_length']) / max(1, len(entry['sequence']))
                        >= MIN_QUERY_COVERAGE]
            if not eligible:
                audits.append({'pdb': entry['pdb'], 'dataset': dataset,
                               'status': 'no_match'})
                continue
            best = max(eligible, key=lambda h: float(h['identity']))
            target_pdb, target_chain = best['pdb_id'].split('_')
            try:
                from scripts.flex_a2_atlas_nmr_audit_v1 import parse_fasta_chain
                chain_seq = parse_fasta_chain(fetch_fasta(target_pdb), target_chain)
                if not chain_seq:
                    audits.append({'pdb': entry['pdb'], 'dataset': dataset,
                                   'status': 'fasta_chain_not_found'})
                    continue
                tracks = fetch_tracks(best['pdb_id'], dataset)
                track = tracks['rmsfTrack']
                begins = np.array([t['begin'] for t in track])
                if not np.all(np.diff(begins) == 1) or begins[0] != 1:
                    audits.append({'pdb': entry['pdb'], 'dataset': dataset,
                                   'status': 'track_positions_not_contiguous'})
                    continue
                rmsf = np.array([float(t['value']) for t in track])
                pairs, score = global_pairs(entry['sequence'], chain_seq)
                if len(pairs) < MIN_ALIGNED:
                    audits.append({'pdb': entry['pdb'], 'dataset': dataset,
                                   'status': 'too_few_aligned_residues'})
                    continue
                nmr_values, md_values = [], []
                for a_index, b_index in pairs:
                    position = b_index + 1
                    if 1 <= position <= len(rmsf):
                        nmr_values.append(entry['rmsf'][a_index])
                        md_values.append(rmsf[position - 1])
                rho = spearman(nmr_values, md_values)
                audits.append({'pdb': entry['pdb'], 'dataset': dataset,
                               'status': 'audited', 'target': best['pdb_id'],
                               'identity': best['identity'],
                               'aligned_pairs': len(pairs), 'spearman': rho,
                               'recall_nmr_top20_in_md': top_recall(
                                   np.array(nmr_values), np.array(md_values))})
            except Exception as error:
                audits.append({'pdb': entry['pdb'], 'dataset': dataset,
                               'status': f'error: {error}'})
            print(audits[-1], flush=True)

    audited = [a for a in audits if a['status'] == 'audited']
    rhos = [a['spearman'] for a in audited if a['spearman'] is not None]
    summary = {
        'entries_searched': len(entries),
        'datasets': ['chameleon', 'DPF'],
        'audited_pairs': len(audited),
        'spearman_values': rhos,
        'spearman_median': float(np.median(rhos)) if rhos else None,
        'gate_threshold_reference': SPEARMAN_GATE,
        'note': 'paired count first; no gate verdict is claimed at this n',
    }
    freeze(OUT / 'audits.json', {'audits': audits})
    freeze(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
