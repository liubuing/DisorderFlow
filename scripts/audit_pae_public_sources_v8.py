"""Versioned, model-free public-source crosscheck, separate from strict v7."""
import base64
import csv
import io
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.build.acquire_rcsb_candidate_interface_extension import fetch_entries

OUT = ROOT / 'data/pae_public_crosscheck_v8'
REPO = 'Zhaonan99/Antibody-antigen-complex-structure-benchmark-dataset'
REVISION = '6891e6e6790c7a898a08471193f494adf6f16b48'


def normalize(value):
    value = value.lower().strip()
    if value.startswith('pdb_'):
        value = value[4:].lstrip('0')
    return value.split('_')[0]


def get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'DisorderFlow-public-source-audit'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'protocol.json', {
        'classification': 'retrospective_public_source_feasibility_not_confirmatory',
        'sabdab_snapshot_sha256': digest(OUT / 'sabdab_snapshot/all-summary.csv'),
        'abag_repository': REPO, 'abag_revision': REVISION,
        'selection': 'All SAbDab PEPTIDE-annotated entries plus all ABAG cases; RCSB chain sequences determine5-50aa scope',
        'metadata_exposure': 'reported separately; not an automatic exclusion in this new retrospective audit',
        'sequence_structure_reference': 'conservative union; overlap reported, no automatic claim of training leakage',
        'strict_v7_protocol': 'unchanged; no retrospective candidates relabeled as v7 independent confirmation',
        'next_gate': 'pairedHL, experimental resolution<=4A, actual5-50aa antigen chain, H3 interface, five-axis isolation and pretraining audit before scoring',
        'no_model_or_teacher_scores': True,
        'script_sha256': digest(Path(__file__))})
    if not (OUT / 'abag_listing.json').exists():
        url = f'https://api.github.com/repos/{REPO}/contents/ABAG-Docking_benchmark_dataset_PDBID.txt?ref={REVISION}'
        payload = get_json(url)
        freeze(OUT / 'abag_listing.json', {'url': url, 'response': payload})
    text = base64.b64decode(read(OUT / 'abag_listing.json')['response']['content']).decode()
    cases = [line.strip() for line in text.splitlines() if line.strip()]
    abag = {normalize(c) for c in cases}
    sabdab = list(csv.DictReader(io.StringIO((OUT / 'sabdab_snapshot/all-summary.csv').read_text(encoding='utf-8-sig'))))
    peptide = [r for r in sabdab if 'PEPTIDE' in r['antigen_type'].upper()]
    peptide_pdbs = {normalize(r['PDB']) for r in peptide}
    union = peptide_pdbs | abag
    if not (OUT / 'rcsb_entries.json').exists():
        entries = fetch_entries(sorted(p.upper() for p in union))
        freeze(OUT / 'rcsb_entries.json', {'entries': entries, 'source': 'https://data.rcsb.org/graphql', 'ids': sorted(union)})
    reference = read(ROOT / 'data/pae_independent_v7/reference_union.json')
    seq_ref = {normalize(r['pdb_id']) for r in reference['records']}
    metadata = {normalize(p) for p in reference['exact_exposed_pdb_ids']}
    freeze(OUT / 'reference_receipt.json', {'path': 'data/pae_independent_v7/reference_union.json',
        'sha256': digest(ROOT / 'data/pae_independent_v7/reference_union.json'),
        'note': 'Sequence/structure union mixes prior training and audits; absence here does not establish absence from all pretraining.'})
    rows = []
    for e in read(OUT / 'rcsb_entries.json')['entries']:
        pdb = normalize(e['rcsb_id'])
        matches = [r for r in sabdab if normalize(r['PDB']) == pdb]
        paired = [r for r in matches if r['Hchain'] not in ('', 'NA') and r['Lchain'] not in ('', 'NA')]
        short_chains = []
        for r in paired:
            for chain, kind in zip(r['antigen_chain'].split('|'), r['antigen_type'].split('|')):
                if kind.upper() not in ('PEPTIDE', 'PROTEIN'):
                    continue
                for entity in e.get('polymer_entities') or []:
                    identifiers = entity.get('rcsb_polymer_entity_container_identifiers') or {}
                    if chain not in (identifiers.get('auth_asym_ids') or []):
                        continue
                    seq = ''.join((entity.get('entity_poly') or {}).get('pdbx_seq_one_letter_code_can', '').split())
                    if 5 <= len(seq) <= 50:
                        short_chains.append({'instance': r['INSTANCE'], 'heavy': r['Hchain'], 'light': r['Lchain'],
                            'antigen': chain, 'sequence': seq, 'type': kind})
        resolution = (e.get('rcsb_entry_info') or {}).get('resolution_combined') or []
        methods = [m['method'] for m in e.get('exptl') or []]
        structure_ok = bool(resolution) and min(resolution) <= 4 and any(m in ('X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY') for m in methods)
        rows.append({'pdb_id': pdb, 'sources': (['sabdab_peptide'] if pdb in peptide_pdbs else []) + (['abag'] if pdb in abag else []),
            'sabdab_instances': len(matches), 'paired_HL_instances': len(paired), 'short_antigen_chains': short_chains,
            'experimental_resolution_pass': structure_ok, 'methods': methods, 'resolution': resolution,
            'in_prior_metadata_exclusions': pdb in metadata, 'in_prior_sequence_structure_reference': pdb in seq_ref,
            'short_task_metadata_eligible': bool(short_chains) and structure_ok,
            'candidate_for_structural_audit': bool(short_chains) and structure_ok and pdb not in seq_ref,
            'independence_established': False})
    counts = {'sabdab_rows': len(sabdab), 'sabdab_peptide_pdbs': len(peptide_pdbs), 'abag_cases': len(cases), 'abag_unique_pdbs': len(abag),
        'source_intersection_peptide_pdbs': len(peptide_pdbs & abag), 'union_pdbs': len(union),
        'short_task_metadata_eligible_pdbs': sum(r['short_task_metadata_eligible'] for r in rows),
        'short_task_without_exact_sequence_reference': sum(r['candidate_for_structural_audit'] for r in rows),
        'abag_short_task_eligible': sum(r['short_task_metadata_eligible'] and 'abag' in r['sources'] for r in rows)}
    freeze(OUT / 'audit.json', {'classification': 'metadata_crosscheck_not_independent_validation', 'counts': counts, 'rows': rows,
        'candidate_ids': [r['pdb_id'] for r in rows if r['candidate_for_structural_audit']],
        'rcsb_sha256': digest(OUT / 'rcsb_entries.json'), 'teacher_labels_accessed': False})
    print(json.dumps({'counts': counts, 'candidate_ids': [r['pdb_id'] for r in rows if r['candidate_for_structural_audit']]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
