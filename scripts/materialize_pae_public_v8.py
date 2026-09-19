"""Coordinate-check public metadata survivors using SAbDab chain roles."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.build import materialize_rcsb_candidate_interface_extension as original

OUT = ROOT / 'data/pae_public_crosscheck_v8'


def main():
    audit = read(OUT / 'audit.json')
    sequence = read(OUT / 'sequence_audit.json')
    passed = set(sequence['summary']['passes_mmseqs_pdbs'])
    seen = {r['pdb_id'] for r in sequence['rows']}
    # Missing numbering is unresolved, never counted as a negative result.
    passed.update(p for p in audit['candidate_ids'] if p not in seen)
    rows = {r['pdb_id'].upper(): r for r in audit['rows'] if r['pdb_id'] in passed}
    snapshot = OUT / 'structure_snapshot'
    snapshot.mkdir(exist_ok=True)
    entries = [e for e in read(OUT / 'rcsb_entries.json')['entries'] if e['rcsb_id'].upper() in rows]
    freeze(snapshot / 'entries.json', {'entries': entries})
    freeze(snapshot / 'acquisition.json', {'classification': 'metadata_only',
        'parent_rcsb_sha256': digest(OUT / 'rcsb_entries.json'),
        'files': {'entries.json': {'sha256': digest(snapshot / 'entries.json')}}})
    discovery = {'classification': 'metadata_only', 'snapshot_dir': snapshot.relative_to(ROOT).as_posix(),
        'snapshot_manifest_sha256': digest(snapshot / 'acquisition.json'), 'entries': []}
    for e in entries:
        r = rows[e['rcsb_id']]
        discovery['entries'].append({'pdb_id': e['rcsb_id'], 'status': 'candidate',
            'resolution': min(r['resolution']), 'methods': r['methods'],
            'release_date': (e.get('rcsb_accession_info') or {}).get('initial_release_date')})
    freeze(OUT / 'structural_discovery.json', discovery)
    freeze(OUT / 'structural_protocol.json', {'classification': 'retrospective_structural_feasibility',
        'selected_pdbs': sorted(passed), 'selection': 'all MMseqs metadata survivors plus missing-sequence case; short sensitivity not used to silently drop cases',
        'chain_roles': 'SAbDab heavy/light and mapped5-50aa antigen chains; coordinate pairing and ANARCII Chothia reannotation required',
        'remaining_rules': 'original materializer resolution, chain length, geometric pairing, H3 contacts and representative selection',
        'scope': 'selected representative perPDB; not exhaustive independent components',
        'wrapper_sha256': digest(Path(__file__)),
        'materializer_sha256': digest(Path(original.__file__)),
        'sequence_audit_sha256': digest(OUT / 'sequence_audit.json')})

    def role_chains(entry, title):
        chains = rows[entry['rcsb_id']]['short_antigen_chains']
        return tuple(sorted({r[key] for r in chains}) for key in ('heavy', 'light', 'antigen'))

    original.role_chains = role_chains
    result = original.materialize(OUT / 'structural_discovery.json', OUT / 'structures')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
