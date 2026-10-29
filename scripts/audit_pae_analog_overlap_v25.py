"""Retrospective exposure review of v22 analog cohorts against prior project sources.

Checks: exact PDB exposure, peptide/receptor sequence overlap against the v3
reference union (all previously examined PAE structures) at the documented
isolation axes. Scans are 4-mer seeded and ranked by shared-seed count with a
documented scan cap; seeding and capping bound sensitivity, so absence of hits
is not proof of absence from all pretraining. This is an exposure review, not a
family assignment.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/pae_analog_overlap_v25'
SOURCE = ROOT / 'data/pae_analog_scope_v22'
V24 = ROOT / 'results/pae_analog_chain_identity_v24'
REFERENCE = ROOT / 'data/successor_v3_confirmatory/reference_union_manifest_v3.json'
SEED = 4
AXES = {'vh': 0.9, 'vl': 0.9, 'antigen': 0.3, 'coverage': 0.8}
SCAN_CAP = 500
PEPTIDE_CAP = 5000


def normalize(value):
    return ''.join((value or '').split()).upper()


def best_overlap(a, b):
    """Best ungapped offset under the v9 rule: >=80% coverage on both sequences."""
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < AXES['coverage']:
        return None
    best = None
    for offset in range(-len(b) + 1, len(a)):
        start, stop = max(0, offset), min(len(a), offset + len(b))
        n = stop - start
        if n / len(a) < AXES['coverage'] or n / len(b) < AXES['coverage']:
            continue
        matches = sum(a[i] == b[i - offset] for i in range(start, stop))
        hit = {'identity': matches / n, 'aligned_length': n,
               'query_coverage': n / len(a), 'reference_coverage': n / len(b)}
        if best is None or hit['identity'] > best['identity']:
            best = hit
    return best


class SeedIndex:
    """Exact 4-mer index. A shared 4-mer is necessary for any qualifying
    ungapped alignment, so candidates outside the index are impossible; inside
    it, ranking by shared-seed count puts true high-identity hits first."""

    def __init__(self, sequences):
        self.sequences = [normalize(s) for s in sequences]
        self.index = {}
        for i, seq in enumerate(self.sequences):
            for j in range(len(seq) - SEED + 1):
                self.index.setdefault(seq[j:j + SEED], []).append(i)

    def ranked_candidates(self, query, cap):
        query = normalize(query)
        per_candidate = Counter()
        for j in range(len(query) - SEED + 1):
            for i in self.index.get(query[j:j + SEED], ()):
                per_candidate[i] += 1
        ranked = sorted(per_candidate, key=lambda i: -per_candidate[i])
        trimmed = ranked[:cap]
        detail = {i: per_candidate[i] for i in trimmed}
        return trimmed, detail, len(ranked)


def scan_axis(index, query, axis, field, reference_ids, cap=SCAN_CAP):
    """Return first qualifying hit at the axis identity threshold, plus scan stats."""
    candidates, detail, total = index.ranked_candidates(query, cap)
    scanned = 0
    for i in candidates:
        target = index.sequences[i]
        if not target:
            continue
        scanned += 1
        hit = best_overlap(normalize(query), target)
        if hit and hit['identity'] >= AXES[axis]:
            return {'reference_id': reference_ids[i], 'identity': round(hit['identity'], 3),
                    'aligned_length': hit['aligned_length']}, scanned, total
    return None, scanned, total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = read(REFERENCE)
    records = ref['records']
    reference_ids = [r['reference_id'] for r in records]
    union_pdbs = {r['pdb_id'].lower() for r in records if r.get('pdb_id')}
    exposed_pdbs = {p.lower() for p in ref.get('exact_exposed_pdb_ids', [])}

    indexes = {
        'vh': SeedIndex([r.get('vh_sequence') for r in records]),
        'vl': SeedIndex([r.get('vl_sequence') for r in records]),
        'antigen': SeedIndex([r.get('antigen_sequence') for r in records]),
    }
    print('reference union:', len(records), 'records; exposed pdbs:', len(exposed_pdbs), flush=True)

    protocol = {
        'classification': 'retrospective_exposure_review_not_family_assignment',
        'reference_union_sha256': digest(REFERENCE),
        'source_sha256': {name: digest(SOURCE / name) for name in
                          ('tcr_candidates.json', 'tcr_meta.json', 'nano_passed.json', 'nano_meta.json')},
        'v24_audit_sha256': digest(V24 / 'audit.json'),
        'axes': AXES,
        'method': 'exact PDB id; exact/containment peptide match; 4-mer seeded ungapped best_overlap at 0.8 both-coverage; antigen axis 0.3, receptor axes 0.9; ranked shared-seed scan caps',
        'scan_caps': {'receptor_axes': SCAN_CAP, 'peptide_axis': PEPTIDE_CAP},
        'sensitivity': 'shared 4-mer is necessary for any qualifying alignment; ranked capping can miss a qualifying hit ranked beyond the cap (true hits share abundant seeds and rank first; residual risk disclosed, not eliminated). MMseqs (v8) unavailable here; absence of hits is not proof of absence from all pretraining',
        'boundary': 'exposure review before preregistration; no model run, no generation, no labels',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)

    v24 = read(V24 / 'audit.json')
    gate_pass = {c: set(v24[c]['passed_pdbs']) for c in ('tcr', 'vhh')}

    entries = {}
    for pdb, entry in read(SOURCE / 'tcr_meta.json').items():
        entries.setdefault(pdb.lower(), {'cohort': 'tcr', 'meta': entry})
    for pdb, entry in read(SOURCE / 'nano_meta.json').items():
        entries.setdefault(pdb.lower(), {'cohort': 'vhh', 'meta': entry})

    peptides = {}
    for cand in read(SOURCE / 'tcr_candidates.json')['candidates']:
        for pep in cand['peptides']:
            seq = normalize(pep['seq'])
            if not seq:
                continue
            peptides.setdefault(seq, {'descs': set(), 'pdbs': set()})
            peptides[seq]['descs'].add(pep.get('desc', ''))
            peptides[seq]['pdbs'].add(cand['pdb'].lower())
    for cand in read(SOURCE / 'nano_passed.json'):
        for pep in cand['peptides']:
            seq = normalize(pep['seq'])
            if not seq:
                continue
            peptides.setdefault(seq, {'descs': set(), 'pdbs': set()})
            peptides[seq]['descs'].add(pep.get('desc', ''))
            peptides[seq]['pdbs'].add(cand['pdb'].lower())
    print('unique peptide sequences:', len(peptides), flush=True)

    antigen_seqs = [normalize(r.get('antigen_sequence')) for r in records]
    peptide_rows = []
    for seq in sorted(peptides):
        row = {'sequence': seq, 'length': len(seq), 'pdbs': sorted(peptides[seq]['pdbs']),
               'exact_reference_hits': [], 'contained_in_reference': [], 'contains_reference': [],
               'overlap_hits': []}
        for i, anti in enumerate(antigen_seqs):
            if not anti:
                continue
            if seq == anti:
                row['exact_reference_hits'].append(reference_ids[i])
            elif len(seq) < len(anti) and seq in anti:
                row['contained_in_reference'].append(reference_ids[i])
            elif len(anti) < len(seq) and anti in seq:
                row['contains_reference'].append(reference_ids[i])
        candidates, detail, total = indexes['antigen'].ranked_candidates(seq, PEPTIDE_CAP)
        scanned = 0
        for i in candidates:
            anti = antigen_seqs[i]
            if not anti or seq == anti or seq in anti or anti in seq:
                continue
            scanned += 1
            hit = best_overlap(seq, anti)
            if hit and hit['identity'] >= AXES['antigen']:
                row['overlap_hits'].append({'reference_id': reference_ids[i], 'identity': round(hit['identity'], 3),
                                            'aligned_length': hit['aligned_length']})
        row['scan'] = {'candidates': len(candidates), 'scored': scanned, 'ranked_total': total, 'cap': PEPTIDE_CAP}
        row['family_axis_hits'] = (len(row['exact_reference_hits']) + len(row['contained_in_reference']) +
                                   len(row['contains_reference']) + len(row['overlap_hits']))
        peptide_rows.append(row)

    entry_rows = []
    for pdb in sorted(entries):
        rec = entries[pdb]
        chains = []
        for pe in rec['meta'].get('polymer_entities', []):
            seq = normalize(pe.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can'))
            if not 80 <= len(seq) <= 600:
                continue
            flags = {}
            for axis in ('vh', 'vl'):
                hit, scanned, total = scan_axis(indexes[axis], seq, axis, axis, reference_ids)
                if hit:
                    flags[axis] = hit | {'scanned': scanned, 'ranked_total': total}
            chains.append({'entity': pe['rcsb_id'], 'desc': pe['rcsb_polymer_entity']['pdbx_description'],
                           'length': len(seq), 'flags': flags})
        entry_rows.append({
            'pdb': pdb, 'cohort': rec['cohort'],
            'v24_gate_pass': pdb in gate_pass[rec['cohort']],
            'pdb_in_reference_union': pdb in union_pdbs,
            'pdb_exact_exposed': pdb in exposed_pdbs,
            'receptor_axis_flags': [c for c in chains if c['flags']],
            'chains_reviewed': len(chains),
        })

    summary = {
        'entries': {'total': len(entry_rows),
                    'tcr': sum(1 for r in entry_rows if r['cohort'] == 'tcr'),
                    'vhh': sum(1 for r in entry_rows if r['cohort'] == 'vhh'),
                    'pdb_in_reference_union': sum(1 for r in entry_rows if r['pdb_in_reference_union']),
                    'pdb_exact_exposed': sum(1 for r in entry_rows if r['pdb_exact_exposed']),
                    'receptor_axis_flagged_entries': sum(1 for r in entry_rows if r['receptor_axis_flags'])},
        'peptides': {'unique': len(peptide_rows),
                     'family_axis_flagged': sum(1 for r in peptide_rows if r['family_axis_hits']),
                     'exact_hits': sum(1 for r in peptide_rows if r['exact_reference_hits']),
                     'contained': sum(1 for r in peptide_rows if r['contained_in_reference'] or r['contains_reference']),
                     'overlap_only': sum(1 for r in peptide_rows if r['overlap_hits'])},
    }
    freeze(OUT / 'audit.json', {'summary': summary, 'entries': entry_rows, 'peptides': peptide_rows})
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {'status': 'completed', 'model_training_started': False,
                                     'af2_started': False, 'audit_sha256': digest(OUT / 'audit.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
