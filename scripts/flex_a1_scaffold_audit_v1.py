"""FLEX-BINDER Phase A1: single-domain scaffold x peptide coordinate/identity gate.

Reuses the audited v26 analog-coordinate machinery (strict residue parsing,
ANARCII numbering, Chothia CDR-H3 region extraction, 4.5 A contact metrics)
on the 57 no-light-chain x peptide-antigen PDB candidates from the frozen
2026-10-29 SAbDab summary scan.

Classification: coordinate/identity gate only. No candidate generation, no
model training, no AF2 labels. Homology isolation, exposure cleanliness and
family splitting are separate later gates and are NOT asserted here.
"""
import json
import sys
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.materialize_pae_analog_structures_v26 import (
    BACKBONE, MIN_RECEPTOR_COVERAGE, backbone_complete, download_entry,
    heavy_coords, region_contact_metrics, region_indices, seq_key,
    strict_residues,
)

OUT = ROOT / 'results/flex_a1_scaffold_audit_v1'
SCAN = ROOT / 'results/flex_campaign_v1_scaffold_scan.json'
CIF_DIR = OUT / 'cif'
REGION_SCHEME = 'chothia'
REGION_RANGE = (93, 102)   # CDR-H3, mirroring the v26 VHH cohort / project H3 definition
PEPTIDE_MIN_LEN, PEPTIDE_MAX_LEN = 5, 50
REGION_MIN_LEN, REGION_MAX_LEN = 3, 40
USER_AGENT = 'DisorderFlow-flex-a1-scaffold-audit/1'
AA = set('ACDEFGHIKLMNPQRSTVWY')


def number_pool(pool, outdir):
    """Resume-able ANARCII antibody numbering keyed by sequence hash (v26 logic)."""
    outdir.mkdir(parents=True, exist_ok=True)
    from anarcii import Anarcii
    model = None
    chothia_model = None
    labels = {}
    items = sorted(pool.items())
    for start in range(0, len(items), 32):
        path = outdir / f'batch_{start:04d}.json'
        if not path.exists():
            if model is None:
                model = Anarcii(seq_type='antibody', mode='accuracy', batch_size=32,
                                cpu=True, ncpu=2, verbose=False)
            batch = dict(items[start:start + 32])
            numbered = model.number(batch)
            missing = set(batch) - set(numbered)
            if missing:
                raise RuntimeError(f'numbering dropped {len(missing)} sequences')
            records = {}
            for name, r in numbered.items():
                records[batch[name]] = {k: r.get(k) for k in
                                        ('chain_type', 'score', 'query_start', 'query_end',
                                         'error', 'numbering')}
            chothia_model = model.to_scheme('chothia')
            for name, r in chothia_model.items():
                if batch[name] in records:
                    records[batch[name]]['chothia_numbering'] = r.get('numbering')
                    records[batch[name]]['chothia_query_start'] = r.get('query_start')
            freeze(path, {'records': records})
        labels.update(read(path)['records'])
        print(f'numbered {len(labels)}/{len(items)}', flush=True)
    if len(labels) != len(items):
        raise RuntimeError(f'label count {len(labels)} != pool {len(items)}')
    return labels


def coordinate_coverage(chain, rows):
    polymer = [r for r in chain.get_residues() if r.id[0] == ' ']
    if not polymer:
        return 0.0
    return round(len(rows) / len(polymer), 3)


def count_nonstandard(residues):
    """Standard-residue check must compare 1-letter codes: resname is 3-letter.

    Regression context: comparing 3-letter resnames against the 1-letter AA set
    silently marked every residue nonstandard, emptied the numbering pool and
    produced a false all-negative audit (caught by the empty-pool guard).
    """
    return sum(1 for r in residues
               if r.id[0] == ' '
               and seq1(r.resname, custom_map={'MSE': 'M'}) not in AA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scan = read(SCAN)
    rows = scan['candidate_pdbs_no_light_peptide']
    by_pdb = {}
    for row in rows:
        entry = by_pdb.setdefault(row['pdb'].lower(), {'summary_rows': [], 'antigen_chains': set()})
        entry['summary_rows'].append({k: row.get(k) for k in
                                      ('method', 'resolution', 'affinity', 'pmid', 'engineered')})
        for chain in (row.get('antigen_chain') or '').split('|'):
            chain = chain.strip()
            if chain:
                entry['antigen_chains'].add(chain)

    protocol = {
        'classification': 'coordinate_identity_contact_gate_not_model_validation',
        'scan_sha256': digest(SCAN),
        'n_candidate_pdbs': len(by_pdb),
        'region': {'scheme': REGION_SCHEME, 'range': list(REGION_RANGE),
                   'note': 'CDR-H3 as in the v26 VHH cohort and project H3 definition'},
        'contact_cutoff_angstrom': 4.5,
        'peptide_length_range': [PEPTIDE_MIN_LEN, PEPTIDE_MAX_LEN],
        'heavy_coordinate_coverage_min': MIN_RECEPTOR_COVERAGE,
        'gates': ['heavy_chain_numbered_as_antibody', 'no_light_chain_in_structure',
                  'peptide_in_sabdab_antigen_annotation', 'peptide_length_5_to_50',
                  'peptide_strict_standard', 'peptide_fully_backbone_resolved',
                  'peptide_chain_not_antibody', 'heavy_coordinate_coverage_at_least_0.7',
                  'region_backbone_complete', 'region_length_3_to_40',
                  'region_peptide_contact_within_4.5A'],
        'pending_separate_gates': ['exposure_cleanliness', 'homology_isolation_vs_ECLS_and_training',
                                   'antigen_family_split', 'biological_identity_review'],
        'source_convention': 'SAbDab all-summary snapshot; single-domain inferred from missing '
                             'light chain annotation, then verified by ANARCII on coordinates',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)
    print('candidate PDBs:', len(by_pdb), flush=True)

    cif_dir = OUT / 'cif'
    cif_dir.mkdir(exist_ok=True)
    receipts = {}
    download_errors = {}
    for pdb in sorted(by_pdb):
        path = cif_dir / f'{pdb}.cif'
        if not path.exists():
            try:
                download_entry(pdb, path)
            except Exception as error:
                download_errors[pdb] = f'download_failed: {error}'
                continue
        receipts[pdb] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    freeze(OUT / 'download_receipt.json', {'receipts': receipts, 'errors': download_errors})
    print('downloaded', len(receipts), 'failed', len(download_errors), flush=True)

    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    structures = {}
    numbering_pool = {}
    for pdb in sorted(receipts):
        try:
            structure = parser.get_structure(pdb, str(cif_dir / f'{pdb}.cif'))
            model = next(structure.get_models())
        except Exception as error:
            structures[pdb] = {'error': f'parse_failed: {error}'}
            continue
        chains = {}
        for chain in model:
            rows_parsed = strict_residues(chain, allow_mse=True)
            if len(rows_parsed) < 5:
                continue
            seq = ''.join(r['aa'] for r in rows_parsed)
            nonstandard = count_nonstandard(chain.get_residues())
            chains[chain.id] = {'rows': rows_parsed, 'seq': seq,
                                'nonstandard_count': nonstandard,
                                'mse_count': sum(1 for r in chain.get_residues()
                                                 if r.id[0] == ' ' and r.resname == 'MSE'),
                                'coverage': coordinate_coverage(chain, rows_parsed)}
            if 80 <= len(seq) <= 600 and nonstandard == 0:
                numbering_pool[seq_key(seq)] = seq
        structures[pdb] = {'chains': chains}
    print('parsed', sum(1 for s in structures.values() if 'chains' in s), 'structures', flush=True)
    if not numbering_pool:
        raise RuntimeError(
            'numbering pool is empty across parsed structures; refusing to emit '
            'an all-negative summary (empty pool means the parse logic is broken, '
            'not that the candidates failed)')

    labels = number_pool(numbering_pool, OUT / 'numbering')

    records = []
    exclusions = []
    for pdb, bundle in sorted(structures.items()):
        if 'error' in bundle:
            exclusions.append({'pdb': pdb, 'reason': bundle['error']})
            continue
        entry = by_pdb[pdb]
        resolved = []
        for chain_id, info in bundle['chains'].items():
            label = labels.get(info['seq'])
            resolved.append({'chain_id': chain_id, 'info': info,
                             'label': label or {},
                             'antibody_type': (label or {}).get('chain_type')
                                              if not (label or {}).get('error') else None})
        heavies = [r for r in resolved if r['antibody_type'] == 'H'
                   and r['info']['coverage'] >= MIN_RECEPTOR_COVERAGE]
        lights = [r for r in resolved if r['antibody_type'] in ('K', 'L')]
        no_light = not lights
        peptides = []
        for r in resolved:
            info = r['info']
            if not (PEPTIDE_MIN_LEN <= len(info['seq']) <= PEPTIDE_MAX_LEN):
                continue
            if r['chain_id'] not in entry['antigen_chains']:
                continue
            if r['antibody_type'] is not None:
                continue
            peptides.append(r)
        if not heavies:
            exclusions.append({'pdb': pdb, 'reason': 'no_confirmed_heavy_chain',
                               'n_numbered_chains': len(resolved)})
            continue
        if not peptides:
            exclusions.append({'pdb': pdb, 'reason': 'no_eligible_peptide_chain',
                               'n_antigen_annotated_chains': len(entry['antigen_chains'])})
            continue

        lo, hi = REGION_RANGE
        for heavy in heavies:
            numbering = heavy['label'].get('chothia_numbering')
            query_start = heavy['label'].get('chothia_query_start',
                                             heavy['label'].get('query_start'))
            if not numbering or query_start is None:
                exclusions.append({'pdb': pdb, 'chain': heavy['chain_id'],
                                   'reason': 'heavy_chain_missing_chothia_numbering'})
                continue
            region = region_indices(numbering, query_start, lo, hi)
            region_rows = [heavy['info']['rows'][i] for i in region
                           if 0 <= i < len(heavy['info']['rows'])]
            region_seq = ''.join(r['aa'] for r in region_rows)
            for peptide in peptides:
                gate = {
                    'heavy_chain_numbered_as_antibody': True,
                    'no_light_chain_in_structure': no_light,
                    'peptide_in_sabdab_antigen_annotation': True,
                    'peptide_length_5_to_50': True,
                    'peptide_strict_standard': (peptide['info']['nonstandard_count'] == 0
                                                and peptide['info']['mse_count'] == 0),
                    'peptide_fully_backbone_resolved': backbone_complete(peptide['info']['rows']),
                    'peptide_chain_not_antibody': peptide['antibody_type'] is None,
                    'heavy_coordinate_coverage_at_least_0.7': heavy['info']['coverage'] >= MIN_RECEPTOR_COVERAGE,
                    'region_backbone_complete': backbone_complete(region_rows),
                    'region_length_3_to_40': REGION_MIN_LEN <= len(region_seq) <= REGION_MAX_LEN,
                }
                contacts = region_contact_metrics(region_rows, peptide['info']['rows'])
                gate['region_peptide_contact_within_4.5A'] = bool(
                    contacts['minimum_distance'] is not None
                    and contacts['minimum_distance'] <= 4.5)
                instance = f"{pdb}_{heavy['chain_id']}_{peptide['chain_id']}"
                records.append({
                    'instance': instance, 'pdb_id': pdb,
                    'summary': entry['summary_rows'][0],
                    'n_summary_rows': len(entry['summary_rows']),
                    'chains': {'heavy': heavy['chain_id'], 'peptide': peptide['chain_id']},
                    'sequences': {'heavy': heavy['info']['seq'],
                                  'peptide': peptide['info']['seq'],
                                  'region': region_seq},
                    'heavy_length': len(heavy['info']['seq']),
                    'peptide_length': len(peptide['info']['seq']),
                    'heavy_coordinate_coverage': heavy['info']['coverage'],
                    'anarcii_heavy_score': heavy['label'].get('score'),
                    'contacts': contacts,
                    'gates': gate,
                    'passes': all(gate.values()),
                })

    summary = {
        'candidate_pdbs': len(by_pdb),
        'pdbs_downloaded': len(receipts),
        'pdbs_excluded': len(exclusions),
        'components': len(records),
        'components_passed': sum(1 for r in records if r['passes']),
        'pdbs_with_passing_component': len({r['pdb_id'] for r in records if r['passes']}),
        'exclusion_reasons': {},
    }
    for exc in exclusions:
        summary['exclusion_reasons'][exc['reason']] = \
            summary['exclusion_reasons'].get(exc['reason'], 0) + 1

    freeze(OUT / 'manifest.json', {'summary': summary, 'records': records,
                                   'exclusions': exclusions})
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {
        'status': 'completed', 'model_training_started': False,
        'candidate_generation_started': False, 'af2_started': False,
        'manifest_sha256': digest(OUT / 'manifest.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
