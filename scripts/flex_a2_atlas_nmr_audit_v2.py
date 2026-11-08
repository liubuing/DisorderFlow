"""FLEX A2 (ATLAS side) v2: re-audit MD RMSF vs corrected NMR-ensemble RMSF.

Re-examines results/flex_a2_atlas_nmr_audit_v1 with the NMR side recomputed by
flex_a2_nmr_rmsf_v2 (corrected Kabsch; the legacy function applied the
transposed rotation). ATLAS search/fasta/tracks responses are reused read-only
from the v1 audit caches; this re-audit is fully OFFLINE and fail-closed on any
missing cache (no network calls, nothing refetched).

Verdict rules are inherited unchanged from v1 (same gate, same thresholds);
only the NMR label source moves v1 -> v2. A per-protein v1-vs-v2 disclosure is
frozen alongside.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402
from scripts.flex_a2_atlas_nmr_audit_v1 import (  # noqa: E402
    MIN_ALIGNED, MIN_IDENTITY, MIN_QUERY_COVERAGE, SPEARMAN_GATE,
    global_pairs, parse_fasta_chain, spearman, top_recall,
)

NMR_V1 = ROOT / 'results/flex_a2_nmr_rmsf_v1/rmsf.json'
NMR_V2 = ROOT / 'results/flex_a2_nmr_rmsf_v2/rmsf.json'
V1_AUDIT = ROOT / 'results/flex_a2_atlas_nmr_audit_v1'
OUT = ROOT / 'results/flex_a2_atlas_nmr_audit_v2'
DATE = '2026-11-08'
CLASSIFICATION = 'flexibility_label_reaudit_not_model_result'


def cached(path):
    if not path.exists():
        raise SystemExit(f'cache missing, refusing network: {path}')
    return path.read_text(encoding='utf-8', errors='replace')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.loads(NMR_V2.read_text(encoding='utf-8'))['entries']
    freeze(OUT / 'protocol.json', {
        'classification': CLASSIFICATION,
        'date': DATE,
        'nmr_source_sha256': digest(NMR_V2),
        'nmr_source_supersedes': digest(NMR_V1),
        'atlas_caches': f'{V1_AUDIT.relative_to(ROOT)}/search, /fasta, /tracks (read-only reuse)',
        'offline': True,
        'match_rules': {'min_identity': MIN_IDENTITY,
                        'min_query_coverage': MIN_QUERY_COVERAGE,
                        'min_aligned_residues': MIN_ALIGNED},
        'verdict_gate': {'metric': 'median within-protein Spearman(NMR RMSF, ATLAS RMSF)',
                         'threshold': SPEARMAN_GATE,
                         'provenance': 'v5.1 pass_physical_label threshold (inherited unchanged)'},
        'script_sha256': digest(Path(__file__))})

    v1_audits = {a['pdb']: a for a in json.loads(
        (V1_AUDIT / 'audits.json').read_text(encoding='utf-8'))['audits']}
    audits = []
    for entry in entries:
        key = entry['pdb']
        sequence = entry['sequence']
        search = json.loads(cached(V1_AUDIT / 'search' / f'{key}.json'))
        hits = search.get('sequence_search_results', [])
        eligible = [h for h in hits
                    if float(h['identity']) >= MIN_IDENTITY
                    and int(h['alignment_length']) / max(1, len(sequence)) >= MIN_QUERY_COVERAGE]
        if not eligible:
            audits.append({'pdb': key, 'status': 'no_atlas_match'})
            continue
        best = max(eligible, key=lambda h: float(h['identity']))
        target_pdb, target_chain = best['pdb_id'].split('_')
        chain_seq = parse_fasta_chain(cached(V1_AUDIT / 'fasta' / f'{target_pdb}.fasta'),
                                      target_chain)
        if not chain_seq:
            audits.append({'pdb': key, 'status': 'fasta_chain_not_found',
                           'target': best['pdb_id']})
            continue
        tracks = json.loads(cached(V1_AUDIT / 'tracks' / f"{best['pdb_id']}.js")
                            .split('=', 1)[1].strip().rstrip(';'))
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
            atlas_position = b_index + 1
            if 1 <= atlas_position <= len(atlas_rmsf):
                nmr_values.append(entry['rmsf'][a_index])
                atlas_values.append(atlas_rmsf[atlas_position - 1])
        rho = spearman(nmr_values, atlas_values)
        audits.append({
            'pdb': key, 'status': 'audited', 'target': best['pdb_id'],
            'identity': best['identity'], 'aligned_pairs': len(pairs),
            'alignment_score': score, 'spearman': rho,
            'recall_nmr_top20_in_atlas': top_recall(np.array(nmr_values),
                                                    np.array(atlas_values)),
            'recall_atlas_top20_in_nmr': top_recall(np.array(atlas_values),
                                                    np.array(nmr_values)),
        })
    freeze(OUT / 'audits.json', {'classification': CLASSIFICATION, 'audits': audits})

    audited = [a for a in audits if a['status'] == 'audited']
    rhos = [a['spearman'] for a in audited]
    rho_median = float(np.median(rhos)) if rhos else None
    verdict = ('pass_physical_label' if rho_median is not None
               and rho_median >= SPEARMAN_GATE else 'fail_physical_label')
    disclosure = []
    for a in audited:
        old = v1_audits.get(a['pdb'])
        disclosure.append({'pdb': a['pdb'], 'target': a['target'],
                           'v1_spearman': None if old is None else old.get('spearman'),
                           'v2_spearman': a['spearman'],
                           'delta': (None if old is None or old.get('spearman') is None
                                     else round(a['spearman'] - old['spearman'], 4))})
    summary = {
        'classification': CLASSIFICATION,
        'date': DATE,
        'nmr_entries': len(entries),
        'audited': len(audited),
        'rho_median': None if rho_median is None else round(rho_median, 4),
        'gate_threshold': SPEARMAN_GATE,
        'verdict': verdict,
        'v1_reference': {'rho_median': 0.6558, 'verdict': 'pass_physical_label',
                         'usage': 'read-only; superseded NMR side'},
        'v1_vs_v2_disclosure': disclosure,
        'claims_boundary': ('Label-infrastructure re-audit only; no model result and no '
                            'design claim. The ATLAS caches are raw service responses '
                            'reused read-only from the frozen v1 audit directory.'),
    }
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'protocol_sha256': digest(OUT / 'protocol.json'),
        'script_sha256': digest(Path(__file__)),
        'artifacts': ['audits.json', 'summary.json', 'completion.json']})
    print('ATLAS v2 re-audit:', verdict, '| audited', len(audited),
          '| rho_median', None if rho_median is None else round(rho_median, 4))
    for d in disclosure:
        print(' ', d['pdb'], '->', d['target'], 'v1', d['v1_spearman'],
              'v2', d['v2_spearman'], 'delta', d['delta'])


if __name__ == '__main__':
    run()
