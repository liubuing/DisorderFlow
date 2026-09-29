"""Offline integrity/eligibility audit of v22 metadata; never scores candidates.

Sequence matches are duplicate diagnostics, not biological family assignments.
No record is declared structurally qualified from title/description metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
AA = set('ACDEFGHIKLMNPQRSTVWY')
NUCLEIC = re.compile(r'\b(?:DNA|RNA|aptamer|ribonucleic|deoxyribonucleic|oligonucleotide)\b', re.I)
RECEPTOR_PART = re.compile(r'\bCD3\b|\bT[ -]?cell (?:surface|receptor)|\bTCR\b|\bMHC\b|\bHLA\b|histocompatibility|microglobulin|immunoglobulin', re.I)
VHH = re.compile(r'\b(?:VHH|nanobody|nanobodies)\b|single[ -]domain antibod', re.I)
TCR = re.compile(r'\bT[ -]?cell receptor\b|\bTCR\b', re.I)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen output differs: {path}')
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def normalize(value):
    return ''.join((value or '').split()).upper()


def verify_receipt(source):
    receipt = read(source/'HARVEST_RECEIPT.json')
    checked = {}
    for name, expected in receipt['files'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError('Receipt path escapes source directory')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Source integrity mismatch: {name}')
        checked[name] = expected['sha256']
    for required in ('tcr_candidates.json', 'nano_passed.json', 'tcr_meta.json', 'nano_meta.json'):
        if required not in checked:
            raise ValueError(f'Unreceipted input: {required}')
    return checked


def peptide_checks(peptide, entities):
    seq = normalize(peptide.get('seq'))
    desc = peptide.get('desc', '')
    errors, pending = [], []
    if not 5 <= len(seq) <= 50:
        errors.append('sequence_length_out_of_scope')
    if len(seq) != peptide.get('len'):
        errors.append('reported_length_differs_from_sequence')
    if set(seq)-AA:
        errors.append('nonstandard_or_unknown_sequence_symbols')
    if NUCLEIC.search(desc):
        errors.append('description_identifies_nucleic_acid')
    if RECEPTOR_PART.search(desc):
        errors.append('description_identifies_receptor_or_accessory_chain')
    matches = [e for e in entities if normalize(e.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can')) == seq]
    if not matches:
        errors.append('sequence_not_found_in_source_entities')
    elif len(matches) != 1:
        pending.append('ambiguous_entity_mapping')
    for entity in matches:
        poly = entity.get('entity_poly', {})
        kind = poly.get('type')
        if kind is None:
            pending.append('polymer_type_not_retrieved')
        elif kind != 'polypeptide(L)':
            errors.append('polymer_type_outside_standard_L_peptide_scope')
        if NUCLEIC.search(entity.get('rcsb_polymer_entity', {}).get('pdbx_description', '')):
            errors.append('source_entity_identifies_nucleic_acid')
    return {
        'sequence_sha256': hashlib.sha256(seq.encode()).hexdigest(),
        'sequence_length': len(seq), 'description': desc,
        'matching_entity_ids': sorted(e['rcsb_id'] for e in matches),
        'errors': sorted(set(errors)), 'pending': sorted(set(pending)),
    }


def audit_cohort(candidates, metadata, cohort):
    if len({r['pdb'] for r in candidates}) != len(candidates):
        raise ValueError(f'Duplicate PDB rows in {cohort}')
    rows, seq_entries = [], defaultdict(set)
    for row in candidates:
        pdb = row['pdb']
        source = metadata.get(pdb)
        errors, pending = [], []
        if source is None:
            source = {}
            errors.append('missing_source_metadata')
        entities = source.get('polymer_entities', [])
        title = source.get('struct', {}).get('title', row.get('title', ''))
        receptor_entities = []
        for e in entities:
            poly = e.get('entity_poly', {})
            desc = e.get('rcsb_polymer_entity', {}).get('pdbx_description', '')
            length = len(normalize(poly.get('pdbx_seq_one_letter_code_can')))
            if length >= 80 and (VHH if cohort == 'vhh' else TCR).search(desc):
                receptor_entities.append(e['rcsb_id'])
        if not receptor_entities:
            pending.append('no_explicit_receptor_entity_annotation')
        if cohort == 'vhh' and not receptor_entities and not VHH.search(title):
            pending.append('no_specific_VHH_evidence_beyond_search_keyword')
        if not row.get('peptides'):
            errors.append('no_candidate_peptide')
        details = [peptide_checks(p, entities) for p in row.get('peptides', [])]
        if details and all(p['errors'] for p in details):
            errors.append('all_candidate_peptides_fail_input_checks')
        for p, detail in zip(row.get('peptides', []), details):
            pending.extend(detail['pending'])
            seq_entries[normalize(p['seq'])].add(pdb)
        # These gates cannot be inferred from titles or canonical sequence strings.
        pending.extend(['receptor_chain_identity_unverified', 'coordinate_chemistry_unverified',
                        'direct_receptor_peptide_contact_unverified',
                        'training_overlap_and_family_independence_unverified'])
        rows.append({'pdb_id': pdb, 'source_title': title,
                     'annotated_receptor_entities': receptor_entities,
                     'candidate_peptides': details, 'errors': sorted(set(errors)),
                     'pending': sorted(set(pending)),
                     'status': 'input_check_failed' if errors else 'requires_review',
                     'eligible_for_model_run': False})
    exact = [{'sequence_sha256': hashlib.sha256(seq.encode()).hexdigest(),
              'length': len(seq), 'pdb_ids': sorted(ids)}
             for seq, ids in sorted(seq_entries.items()) if len(ids) > 1]
    containment = []
    sequences = sorted(seq_entries)
    for i, a in enumerate(sequences):
        for b in sequences[i+1:]:
            short, long = sorted((a, b), key=len)
            if len(short) != len(long) and short in long:
                containment.append({'short_sha256': hashlib.sha256(short.encode()).hexdigest(),
                                    'long_sha256': hashlib.sha256(long.encode()).hexdigest(),
                                    'pdb_ids': sorted(seq_entries[a] | seq_entries[b])})
    reasons = Counter(reason for r in rows for reason in r['errors'])
    peptide_reasons = Counter(reason for r in rows for p in r['candidate_peptides'] for reason in p['errors'])
    return {'summary': {'input_entries': len(rows),
                        'input_peptide_records': sum(len(r['candidate_peptides']) for r in rows),
                        'unique_raw_sequences': len(seq_entries),
                        'input_check_failed_entries': sum(bool(r['errors']) for r in rows),
                        'remaining_requires_review': sum(not r['errors'] for r in rows),
                        'eligible_for_model_run': 0,
                        'repeated_exact_sequence_groups': len(exact),
                        'strict_containment_pairs': len(containment),
                        'entry_error_counts': dict(sorted(reasons.items())),
                        'peptide_error_counts': dict(sorted(peptide_reasons.items()))},
            'rows': rows, 'exact_sequence_duplicates': exact,
            'strict_sequence_containment': containment}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'data/pae_analog_scope_v22')
    parser.add_argument('--out', type=Path, default=ROOT/'results/pae_analog_audit_v23')
    args = parser.parse_args()
    hashes = verify_receipt(args.source)
    freeze(args.out/'protocol.json', {
        'classification': 'offline_input_audit_not_model_validation',
        'source_sha256': hashes, 'script_sha256': digest(__file__),
        'rules': '5-50 canonical sequence length; length agreement; standard alphabet; source entity mapping; explicit negative identity annotations; receptor annotation review',
        'missing_metadata': 'unknown, never pass; canonical alphabet alone does not establish polymer type or chemistry',
        'duplicates': 'exact/strict containment diagnostics only; not antigen-family or homology assignments',
        'eligibility': 'no model run until chain identity, chemistry, direct contact, provenance and split gates are resolved',
        'fitting_and_generation': 'none',
    })
    tcr = read(args.source/'tcr_candidates.json')
    output = {'classification': 'offline_input_audit_not_model_validation',
              'source_sha256': hashes,
              'tcr': audit_cohort(tcr['candidates'], read(args.source/'tcr_meta.json'), 'tcr'),
              'vhh': audit_cohort(read(args.source/'nano_passed.json'), read(args.source/'nano_meta.json'), 'vhh')}
    freeze(args.out/'audit.json', output)
    summary = {key: output[key]['summary'] for key in ('tcr', 'vhh')}
    freeze(args.out/'summary.json', summary)
    completion = args.out/'completion.json'
    if not completion.exists():
        freeze(completion, {'status': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat(),
                            'audit_sha256': digest(args.out/'audit.json'),
                            'protocol_sha256': digest(args.out/'protocol.json'),
                            'model_training_started': False, 'af2_started': False})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
