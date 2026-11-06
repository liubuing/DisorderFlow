"""FLEX-BINDER Phase A2 reroute leg 2: large-n consistency audit between ATLAS
MD RMSF and AlphaFold2 pLDDT across the full ATLAS library (1,938 proteins).

Question: does AF2 confidence rank-order residue flexibility as measured by
3x100 ns MD? This is the label-relevance audit for the whole AF2-derived
flexibility-supervision line (v5 AF2-seed RMSF labels included).

Label infrastructure only; no training and no design claims. Per-protein
results are cached resume-ably; the frozen summary is written at completion.
"""
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import freeze, digest
from scripts.flex_a2_atlas_nmr_audit_v1 import average_ranks

OUT = ROOT / 'results/flex_a2_atlas_plddt_audit_v1'
PARSABLE = ROOT / 'data/flex_atlas_v1/atlas_parsable.zip'
API = 'https://www.dsimb.inserm.fr/ATLAS/api'
REFERER = 'https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{pdb_id}/{pdb_id}.html'
USER_AGENT = 'DisorderFlow-flex-a2-plddt-audit/1'
SLEEP_SEC = 1.2


def library_ids():
    with zipfile.ZipFile(PARSABLE) as bundle:
        data = bundle.read('ATLAS_parsable_latest/2023_03_09_ATLAS_info.tsv').decode(
            'utf-8', errors='replace')
    rows = data.splitlines()[1:]
    return sorted(line.split('\t')[0] for line in rows if line.strip())


def spearman(a, b):
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def fetch_tracks(pdb_id):
    path = OUT / 'tracks' / f'{pdb_id}.js'
    if path.exists():
        text = path.read_text(encoding='utf-8', errors='replace')
        return json.loads(text.split('=', 1)[1].strip().rstrip(';'))
    request = urllib.request.Request(
        f'{API}/tracks/{pdb_id}.js',
        headers={'User-Agent': USER_AGENT,
                 'Referer': REFERER.format(pdb_id=pdb_id),
                 'X-Requested-With': 'XMLHttpRequest'})
    payload = None
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
        raise RuntimeError('tracks payload is not trackData')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    time.sleep(SLEEP_SEC)
    text = path.read_text(encoding='utf-8', errors='replace')
    return json.loads(text.split('=', 1)[1].strip().rstrip(';'))


def audit_one(pdb_id):
    cache = OUT / 'per_protein' / f'{pdb_id}.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    result = {'pdb_id': pdb_id}
    try:
        tracks = fetch_tracks(pdb_id)
        rmsf_track = tracks.get('rmsfTrack') or []
        plddt_track = tracks.get('plddtTrack') or []
        rmsf_map = {t['begin']: float(t['value']) for t in rmsf_track}
        plddt_map = {t['begin']: float(t['value']) for t in plddt_track}
        common = sorted(set(rmsf_map) & set(plddt_map))
        if len(common) < 10:
            result['status'] = 'insufficient_track_overlap'
        else:
            rmsf = np.array([rmsf_map[p] for p in common])
            plddt = np.array([plddt_map[p] for p in common])
            result.update({
                'status': 'audited',
                'n_residues': len(common),
                'spearman_rmsf_neg_plddt': spearman(rmsf, -plddt),
                'rmsf_mean': round(float(rmsf.mean()), 4),
                'plddt_mean': round(float(plddt.mean()), 2),
            })
    except Exception as error:
        result['status'] = f'error: {error}'
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding='utf-8')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = library_ids()
    freeze(OUT / 'protocol.json', {
        'classification': 'label_relevance_audit_not_model_result',
        'library': 'ATLAS 2023_03_09 info.tsv (1938 proteins expected)',
        'n_library': len(ids),
        'metric': 'per-protein Spearman(MD RMSF, -pLDDT) over tracks-overlap residues',
        'parsable_sha256': digest(PARSABLE),
        'script_sha256': digest(Path(__file__))})
    print('library size:', len(ids), flush=True)

    results = []
    for index, pdb_id in enumerate(ids, 1):
        results.append(audit_one(pdb_id))
        if index % 100 == 0:
            audited = sum(1 for r in results if r['status'] == 'audited')
            print(f'[{index}/{len(ids)}] audited so far: {audited}', flush=True)

    rhos = np.array([r['spearman_rmsf_neg_plddt'] for r in results
                     if r.get('spearman_rmsf_neg_plddt') is not None])
    summary = {
        'library': len(ids),
        'audited': sum(1 for r in results if r['status'] == 'audited'),
        'errors': sum(1 for r in results if str(r['status']).startswith('error')),
        'insufficient_overlap': sum(1 for r in results
                                    if r['status'] == 'insufficient_track_overlap'),
        'rho_median': float(np.median(rhos)) if len(rhos) else None,
        'rho_iqr': ([float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))]
                    if len(rhos) else None),
        'fraction_rho_ge_0.30': float(np.mean(rhos >= 0.30)) if len(rhos) else None,
        'classification': 'descriptive_distribution_not_gate_claim',
    }
    freeze(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
