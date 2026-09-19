"""Metadata sequence triage; never claims structural or pretraining independence."""
import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_successor_v3_isolation import AXES, clean_sequence, write_fasta, run_mmseqs_search, mmseqs_version, connected_components

OUT = ROOT / 'data/pae_public_crosscheck_v8'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def freeze(p, value):
    if p.exists():
        assert read(p) == json.loads(json.dumps(value)), f'Changed frozen artifact {p}'
    else:
        p.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ungapped(a, b, threshold):
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < .8:
        return False
    for offset in range(-len(b)+1, len(a)):
        start, stop = max(0, offset), min(len(a), offset+len(b))
        n = stop-start
        if n / len(a) >= .8 and n / len(b) >= .8:
            if sum(a[i] == b[i-offset] for i in range(start, stop)) / n >= threshold:
                return True
    return False


def prepare():
    if (OUT / 'sequence_protocol.json').exists():
        raise FileExistsError('Preparation already frozen; use run stage')
    audit = read(OUT / 'audit.json')
    sab = {}
    for r in csv.DictReader((OUT / 'sabdab_snapshot/all-summary.csv').open(encoding='utf-8')):
        sab.setdefault(r['INSTANCE'], r)
    records = {}
    missing = []
    for r in audit['rows']:
        if not r['candidate_for_structural_audit']:
            continue
        for chain in r['short_antigen_chains']:
            s = sab[chain['instance']]
            item = {'instance': chain['instance'] + '-' + chain['antigen'], 'pdb_id': r['pdb_id'],
                'vh_sequence': clean_sequence(s['VH']), 'vl_sequence': clean_sequence(s['VL']),
                'cdr_h3_sequence': clean_sequence(s['CDR-H3']),
                'paired_cdr_sequence': ''.join(clean_sequence(s[k]) for k in ('CDR-H1','CDR-H2','CDR-H3','CDR-L1','CDR-L2','CDR-L3')),
                'antigen_sequence': clean_sequence(chain['sequence']), 'chains': chain,
                'prior_metadata_seen': r['in_prior_metadata_exclusions']}
            if any(not item[f] for f, _ in AXES.values()):
                missing.append(item['instance'])
            else:
                records[item['instance']] = item
    freeze(OUT / 'sequence_candidates.json', {'classification': 'sabdab_metadata_sequences_not_coordinate_annotation',
        'records': list(records.values()), 'missing_sequence_instances': sorted(set(missing)),
        'source_audit_sha256': digest(OUT / 'audit.json')})
    freeze(OUT / 'sequence_protocol.json', {'classification': 'retrospective_metadata_triage',
        'thresholds': AXES, 'both_coverage': .8,
        'reference_sha256': digest(ROOT / 'data/pae_independent_v7/reference_union.json'),
        'candidates_sha256': digest(OUT / 'sequence_candidates.json'),
        'numbering_caveat': 'SAbDab CDR definitions may differ from project Chothia93-102; coordinate reannotation required for final eligibility',
        'short_sensitivity': 'separate ungapped H3/antigen check; low short-sequence identity is an operational hit, not proof of evolutionary homology',
        'script_sha256': digest(Path(__file__))})
    print(json.dumps({'sequence_instances': len(records), 'pdbs': len({r['pdb_id'] for r in records.values()}), 'missing': len(set(missing))}), flush=True)


def run():
    protocol = read(OUT / 'sequence_protocol.json')
    current_hash = digest(Path(__file__))
    if current_hash != protocol['script_sha256']:
        amendment = read(OUT / 'sequence_implementation_amendment.json')
        assert amendment['original_sha256'] == protocol['script_sha256']
        assert amendment['corrected_sha256'] == current_hash
    source = ROOT / 'data/pae_independent_v7/reference_union.json'
    assert digest(source) == protocol['reference_sha256']
    assert digest(OUT / 'sequence_candidates.json') == protocol['candidates_sha256']
    candidates = read(OUT / 'sequence_candidates.json')['records']
    refs = read(source)['records']
    version, _ = mmseqs_version('mmseqs')
    hits = {}
    commands = []
    with tempfile.TemporaryDirectory(prefix='pae_v8_') as temporary:
        temp = Path(temporary)
        for axis, (field, threshold) in AXES.items():
            q, t = temp / ('query_'+axis+'.fasta'), temp / ('reference_'+axis+'.fasta')
            write_fasta(q, candidates, field, 'instance')
            write_fasta(t, refs, field, 'reference_id')
            hits[axis], command = run_mmseqs_search('mmseqs', q, t, temp / (axis+'.tsv'), temp / ('work_'+axis), threshold, 4)
            commands.append([v.replace(str(temp), '<TEMP>') for v in command])
            print(axis, len(hits[axis]), flush=True)
    rows = []
    for c in candidates:
        matched = {axis: [h for h in values if h['query'] == c['instance']] for axis, values in hits.items()}
        sensitivity = {}
        for axis, field, threshold in [('h3', 'cdr_h3_sequence', .5), ('antigen', 'antigen_sequence', .3)]:
            sensitivity[axis] = [r['reference_id'] for r in refs if ungapped(c[field], clean_sequence(r.get(field, '')), threshold)]
        rows.append({'instance': c['instance'], 'pdb_id': c['pdb_id'],
            'failed_mmseqs_axes': [a for a, h in matched.items() if h],
            'mmseqs_best_hits': {a: sorted(h, key=lambda x: -float(x['identity']))[:3] for a,h in matched.items()},
            'short_sensitivity_hit_counts': {a: len(h) for a,h in sensitivity.items()},
            'short_sensitivity_examples': {a:h[:3] for a,h in sensitivity.items()},
            'passes_mmseqs': not any(matched.values()),
            'passes_both': not any(matched.values()) and not any(sensitivity.values())})
    summary = {'instances': len(rows), 'pdbs': len({r['pdb_id'] for r in rows}),
        'passes_mmseqs_instances': sum(r['passes_mmseqs'] for r in rows),
        'passes_mmseqs_pdbs': sorted({r['pdb_id'] for r in rows if r['passes_mmseqs']}),
        'passes_both_instances': sum(r['passes_both'] for r in rows),
        'passes_both_pdbs': sorted({r['pdb_id'] for r in rows if r['passes_both']}),
        'mmseqs_excluded_by_axis': dict(Counter(a for r in rows for a in r['failed_mmseqs_axes']))}
    freeze(OUT / 'sequence_audit.json', {'classification': 'metadata_sequence_triage_not_final_independence',
        'summary': summary, 'rows': rows, 'mmseqs_version': version, 'commands': commands,
        'protocol_sha256': digest(OUT / 'sequence_protocol.json'), 'structural_validation_done': False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run)()
