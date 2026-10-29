"""Materialize bound structures for the gate-passing v22 analog cohorts.

Downloads mmCIF for every entry that passed the v24 chain-identity gate,
resolves peptide and receptor chains by sequence matching, numbers the
coordinate-derived receptor sequences (ANARCII; IMGT for TCR, Chothia for
VHH), extracts the candidate region (CDR3-beta IMGT 105-117; Chothia 93-102
CDR3 mirroring the project H3 definition), computes candidate-region to
peptide contact metrics, and writes canonical component PDBs.

No candidate generation, no AF2 labels, no model training: coordinate gate only.
"""
import copy as _copy
import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.build.materialize_successor_v3_confirmatory import CONTACT_CUTOFF

OUT = ROOT / 'results/pae_analog_structures_v26'
SOURCE = ROOT / 'data/pae_analog_scope_v22'
V24 = ROOT / 'results/pae_analog_chain_identity_v24'
DOWNLOAD_URL = 'https://files.rcsb.org/download/{pdb_id}.cif'
USER_AGENT = 'DisorderFlow-pae-analog-coordinate-gate/1'
AA = set('ACDEFGHIKLMNPQRSTVWY')
BACKBONE = ('N', 'CA', 'C', 'O')
MIN_RECEPTOR_COVERAGE = 0.7

COHORTS = {
    'tcr': {'receptor_types': ('B',), 'partner_type': 'A',
            'region_scheme': 'imgt', 'region_range': (105, 117),
            'chain_names': {'partner': 'A', 'receptor': 'B', 'peptide': 'P'},
            'meta': 'tcr_meta.json', 'anarcii_type': 'tcr'},
    'vhh': {'receptor_types': ('H',), 'partner_type': None,
            'region_scheme': 'chothia', 'region_range': (93, 102),
            'chain_names': {'receptor': 'V', 'peptide': 'P'},
            'meta': 'nano_meta.json', 'anarcii_type': 'antibody'},
}


def strict_residues(chain, allow_mse):
    rows = []
    for residue in chain.get_residues():
        if residue.id[0] != ' ' or 'CA' not in residue:
            continue
        aa = seq1(residue.resname, custom_map={'MSE': 'M'})
        if aa not in AA:
            continue
        if residue.resname == 'MSE' and not allow_mse:
            continue
        rows.append({'aa': aa, 'auth_seq_id': int(residue.id[1]),
                     'insertion_code': residue.id[2].strip(), 'residue': residue})
    return rows


def heavy_coords(residues):
    arrays = [np.asarray([a.coord for a in r['residue'].get_atoms() if a.element != 'H'], dtype=float)
              for r in residues]
    return [a for a in arrays if len(a)]


def region_contact_metrics(region_residues, peptide_residues):
    region = heavy_coords(region_residues)
    peptide = heavy_coords(peptide_residues)
    minimum = float('inf')
    contacting = 0
    residue_contacts = 0
    for coords in region:
        position_contacts = 0
        for other in peptide:
            distance = float(np.linalg.norm(coords[:, None, :] - other[None, :, :], axis=-1).min())
            minimum = min(minimum, distance)
            if distance <= CONTACT_CUTOFF:
                position_contacts += 1
        contacting += int(position_contacts > 0)
        residue_contacts += position_contacts
    return {'minimum_distance': None if minimum == float('inf') else round(minimum, 3),
            'n_contacting_region_positions': contacting,
            'n_region_peptide_residue_contacts': residue_contacts}


def backbone_complete(rows):
    return all(all(name in r['residue'] for name in BACKBONE) for r in rows)


def seq_key(sequence):
    return 'seq_' + hashlib.sha256(sequence.encode()).hexdigest()[:16]


def number_chains(sequences, seq_type):
    """Resume-able numbering of coordinate-derived chains keyed by sequence hash.

    ``sequences`` maps hash keys to sequences; batch records are frozen keyed by
    sequence so assembly can look labels up by chain sequence.
    """
    outdir = OUT / 'numbering' / seq_type
    outdir.mkdir(parents=True, exist_ok=True)
    from anarcii import Anarcii
    model = None
    labels = {}
    items = sorted(sequences.items())
    for start in range(0, len(items), 32):
        path = outdir / f'batch_{start:04d}.json'
        if not path.exists():
            if model is None:
                model = Anarcii(seq_type=seq_type, mode='accuracy', batch_size=32,
                                cpu=True, ncpu=2, verbose=False)
            batch = dict(items[start:start + 32])
            numbered = model.number(batch)
            missing = set(batch) - set(numbered)
            if missing:
                raise RuntimeError(f'{seq_type}: numbering dropped {len(missing)} sequences')
            records = {}
            for name, r in numbered.items():
                records[batch[name]] = {k: r.get(k) for k in
                                        ('chain_type', 'score', 'query_start', 'query_end', 'error', 'numbering')}
            if seq_type == 'antibody':
                chothia = model.to_scheme('chothia')
                for name, r in chothia.items():
                    if batch[name] in records:
                        records[batch[name]]['chothia_numbering'] = r.get('numbering')
                        records[batch[name]]['chothia_query_start'] = r.get('query_start')
            freeze(path, {'records': records, 'seq_type': seq_type})
        labels.update(read(path)['records'])
        print(f'[{seq_type}] numbered {len(labels)}/{len(items)}', flush=True)
    if len(labels) != len(items):
        raise RuntimeError(f'{seq_type}: label count {len(labels)} != pool {len(items)}')
    return labels


def region_indices(numbering, query_start, lo, hi):
    """Input-sequence indices of numbered positions in [lo, hi] (extension convention)."""
    indices = []
    index = int(query_start)
    for (position, _insertion), aa in numbering:
        if aa == '-':
            continue
        if lo <= int(position) <= hi:
            indices.append(index)
        index += 1
    return indices


def download_entry(pdb_id, path, retries=3):
    request = urllib.request.Request(DOWNLOAD_URL.format(pdb_id=pdb_id),
                                     headers={'User-Agent': USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload.startswith(b'data_'):
                raise RuntimeError('RCSB response is not an mmCIF file')
            path.write_bytes(payload)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def best_entity_match(sequence, entities):
    """Best ungapped entity match via 4-mer offset voting (shared 4-mer is
    necessary for any >=0.9-identity ungapped alignment)."""
    best = None
    for entity in entities:
        target = entity['seq']
        if not target or len(target) < 20:
            continue
        positions = {}
        for i in range(len(target) - 3):
            positions.setdefault(target[i:i + 4], []).append(i)
        votes = Counter()
        for j in range(len(sequence) - 3):
            for i in positions.get(sequence[j:j + 4], ()):
                votes[j - i] += 1
        for offset, _votes in votes.most_common(3):
            start = max(0, offset)
            stop = min(len(sequence), offset + len(target))
            n = stop - start
            if n < 20:
                continue
            matches = sum(1 for k in range(start, stop) if sequence[k] == target[k - offset])
            identity = matches / n
            hit = {'entity_id': entity['entity_id'], 'identity': round(identity, 3),
                   'resolved_coverage': round(n / len(sequence), 3)}
            if best is None or identity > best['identity']:
                best = hit
    return best


def atom_line(serial, name, altloc, resname, chain, resseq, icode, coord, occ, bfactor, element):
    if len(name) < 4 and len(element.strip()) < 2:
        name = ' ' + name
    x, y, z = coord
    return (f"ATOM  {serial:5d} {name:<4s}{altloc:1s}{resname:>3s} {chain:1s}{resseq:4d}{icode:1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfactor:6.2f}          {element:>2s}\n")


def write_component_pdb(path, ordered_chains):
    """Manual PDB writer: residue rows in order, chains renamed, residues
    renumbered 1..N per chain. Avoids Bio.PDB deepcopy of the large parent
    graph and the single-character chain-id limit."""
    lines = []
    serial = 1
    for final_name, rows in ordered_chains:
        number = 0
        for row in rows:
            number += 1
            icode = (row['insertion_code'] or ' ')[:1]
            for atom in row['residue'].get_unpacked_list():
                element = (atom.element or 'X').strip().upper() or 'X'
                if element == 'H':
                    continue
                occ = atom.get_occupancy()
                occ = 1.0 if occ is None else occ
                bfactor = atom.get_bfactor()
                bfactor = 0.0 if bfactor is None else bfactor
                lines.append(atom_line(serial, atom.get_name(), (atom.get_altloc() or ' ')[:1],
                                       row['residue'].resname, final_name, number, icode,
                                       atom.coord, occ, bfactor, element))
                serial += 1
        lines.append(f"TER   {serial:5d}\n")
        serial += 1
    lines.append("END\n")
    Path(path).write_text(''.join(lines), encoding='ascii')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v24 = read(V24 / 'audit.json')
    passed = {c: {p.lower() for p in v24[c]['passed_pdbs']} for c in COHORTS}

    entity_chains = {}
    for cohort in COHORTS:
        per_entry = {}
        for row in v24[cohort]['rows']:
            per_entry[row['pdb']] = row['chains']
        entity_chains[cohort] = per_entry

    peptide_pool = {}
    for cand in read(SOURCE / 'tcr_candidates.json')['candidates']:
        for pep in cand['peptides']:
            peptide_pool.setdefault(cand['pdb'].lower(), set()).add(''.join(pep['seq'].split()).upper())
    for cand in read(SOURCE / 'nano_passed.json'):
        for pep in cand['peptides']:
            peptide_pool.setdefault(cand['pdb'].lower(), set()).add(''.join(pep['seq'].split()).upper())

    protocol = {
        'classification': 'coordinate_materialization_and_contact_gate_not_model_validation',
        'source_sha256': {name: digest(SOURCE / name) for name in
                          ('tcr_candidates.json', 'tcr_meta.json', 'nano_passed.json', 'nano_meta.json')},
        'v24_audit_sha256': digest(V24 / 'audit.json'),
        'contact_cutoff_angstrom': CONTACT_CUTOFF,
        'region_schemes': {c: {'scheme': cfg['region_scheme'], 'range': list(cfg['region_range'])}
                           for c, cfg in COHORTS.items()},
        'gates': ['peptide_exact_match_to_v23_candidate', 'peptide_strict_standard',
                  'peptide_fully_backbone_resolved',
                  'region_length_matches_entity_cdr3', 'region_backbone_complete',
                  'region_peptide_contact_within_cutoff', 'receptor_coverage_at_least_0.7',
                  'used_chains_no_nonstandard_residues', 'partner_present_for_tcr'],
        'chemistry_convention': 'receptor chains: MSE read as M and counted, any other nonstandard residue disqualifies; peptide: strict standard residues only (MSE disqualifies), stricter than the antibody pipeline and disclosed',
        'boundary': 'no candidate generation, no AF2 labels, no training; bound geometry and coordinate gates only',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)

    cif_dir = OUT / 'cif'
    pdb_dir = OUT / 'pdb'
    cif_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)

    plan = []
    for cohort, cfg in COHORTS.items():
        meta_all = read(SOURCE / cfg['meta'])
        for pdb in sorted(passed[cohort]):
            cif_path = cif_dir / f'{pdb}.cif'
            if not cif_path.exists():
                try:
                    download_entry(pdb, cif_path)
                except Exception as error:
                    plan.append((cohort, pdb, cfg, meta_all.get(pdb.upper()), f'download_failed: {error}'))
                    continue
            plan.append((cohort, pdb, cfg, meta_all.get(pdb.upper()), None))
    freeze(OUT / 'download_receipt.json',
           {pdb: {'sha256': digest(cif_dir / f'{pdb}.cif'), 'bytes': (cif_dir / f'{pdb}.cif').stat().st_size}
           for _c, pdb, _cfg, _m, err in plan if err is None})
    print('entries planned:', len(plan), flush=True)

    structures = {}
    numbering_pool = {}
    for cohort, pdb, cfg, meta, error in plan:
        if error:
            structures[pdb] = {'error': error}
            continue
        try:
            structure = parser.get_structure(pdb, str(cif_dir / f'{pdb}.cif'))
            model = next(structure.get_models())
        except Exception as parse_error:
            structures[pdb] = {'error': f'parse_failed: {parse_error}'}
            continue
        chains = {}
        for chain in model:
            rows = strict_residues(chain, allow_mse=True)
            if len(rows) < 5:
                continue
            seq = ''.join(r['aa'] for r in rows)
            nonstandard = sum(1 for r in chain.get_residues()
                              if r.id[0] == ' ' and seq1(r.resname, custom_map={'MSE': 'M'}) not in AA)
            mse = sum(1 for r in chain.get_residues() if r.id[0] == ' ' and r.resname == 'MSE')
            chains[chain.id] = {'rows': rows, 'seq': seq,
                                'nonstandard_count': nonstandard, 'mse_count': mse}
            if 80 <= len(seq) <= 600 and nonstandard == 0:
                numbering_pool.setdefault(cfg['anarcii_type'], {})[seq_key(seq)] = seq
        structures[pdb] = {'chains': chains, 'cohort': cohort, 'cfg': cfg, 'meta': meta}
    print('entries parsed:', sum(1 for s in structures.values() if 'chains' in s), flush=True)

    labels = {t: number_chains(pool, t) for t, pool in sorted(numbering_pool.items())}

    # All parsed structures stay live for the assembly loop; with millions of
    # resident objects, generational GC scans dominate runtime. Refcounting
    # handles the small assembly allocations, so collection is unnecessary.
    import gc
    gc.disable()

    records = []
    exclusions = []
    for pdb, bundle in sorted(structures.items()):
        print('assembling', pdb, flush=True)
        if 'error' in bundle:
            exclusions.append({'pdb': pdb, 'reason': bundle['error']})
            continue
        cohort = bundle['cohort']
        cfg = bundle['cfg']
        meta = bundle['meta'] or {}
        entities = [{'entity_id': pe['rcsb_id'],
                     'seq': ''.join((pe.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can') or '').split()).upper()}
                    for pe in meta.get('polymer_entities', [])]
        entity_cdr3 = {}
        for row in entity_chains[cohort].get(pdb.upper(), entity_chains[cohort].get(pdb, [])):
            if cohort == 'tcr' and row.get('cdr3_beta_imgt'):
                entity_cdr3[row['entity']] = row['cdr3_beta_imgt']
            if cohort == 'vhh' and row.get('cdr3_imgt'):
                entity_cdr3[row['entity']] = row['cdr3_imgt']

        resolved = []
        for chain_id, info in bundle['chains'].items():
            label = labels.get(cfg['anarcii_type'], {}).get(info['seq'])
            if not label or label.get('error') or not label.get('chain_type'):
                continue
            match = best_entity_match(info['seq'], entities) if info['nonstandard_count'] == 0 else None
            resolved.append({'chain_id': chain_id, 'chain_type': label['chain_type'], 'label': label,
                             'match': match, **info})

        receptors = [r for r in resolved if r['chain_type'] in cfg['receptor_types'] and r['match']
                     and r['match']['identity'] >= 0.9 and r['match']['resolved_coverage'] >= MIN_RECEPTOR_COVERAGE]
        partners = ([r for r in resolved if r['chain_type'] == cfg['partner_type'] and r['match']
                     and r['match']['identity'] >= 0.9 and r['match']['resolved_coverage'] >= MIN_RECEPTOR_COVERAGE]
                    if cfg['partner_type'] else [None])
        allowed = peptide_pool.get(pdb, set())
        peptides = [{'chain_id': cid, **info} for cid, info in bundle['chains'].items()
                    if 5 <= len(info['seq']) <= 50 and info['seq'] in allowed]

        if not receptors or not peptides or not partners:
            exclusions.append({'pdb': pdb, 'reason': 'no_resolved_receptor_partner_or_peptide'})
            continue
        lo, hi = cfg['region_range']
        combos = 0
        for receptor in receptors:
            scheme = cfg['region_scheme']
            if scheme == 'chothia':
                numbering = receptor['label'].get('chothia_numbering')
                query_start = receptor['label'].get('chothia_query_start', receptor['label']['query_start'])
            else:
                numbering = receptor['label']['numbering']
                query_start = receptor['label']['query_start']
            if not numbering or query_start is None:
                continue
            region = region_indices(numbering, query_start, lo, hi)
            region_rows = [receptor['rows'][i] for i in region if 0 <= i < len(receptor['rows'])]
            region_seq = ''.join(r['aa'] for r in region_rows)
            entity_cdr3_seq = entity_cdr3.get(receptor['match']['entity_id'])
            for peptide in peptides:
                for partner in partners:
                    combos += 1
                    gate = {
                        'peptide_exact_match_to_v23_candidate': True,
                        'peptide_strict_standard': peptide['nonstandard_count'] == 0 and peptide['mse_count'] == 0,
                        'peptide_fully_backbone_resolved': backbone_complete(peptide['rows']),
                        'region_length_matches_entity_cdr3': bool(entity_cdr3_seq)
                                                             and len(region_seq) == len(entity_cdr3_seq),
                        'region_backbone_complete': backbone_complete(region_rows),
                        'receptor_coverage_at_least_0.7': receptor['match']['resolved_coverage'] >= MIN_RECEPTOR_COVERAGE,
                        'used_chains_no_nonstandard_residues': (receptor['nonstandard_count'] + peptide['nonstandard_count']
                                                                + (partner['nonstandard_count'] if partner else 0)) == 0,
                        'partner_present_for_tcr': cohort != 'tcr' or partner is not None,
                    }
                    contacts = region_contact_metrics(region_rows, peptide['rows'])
                    gate['region_peptide_contact_within_cutoff'] = bool(
                        contacts['minimum_distance'] is not None and contacts['minimum_distance'] <= CONTACT_CUTOFF)
                    name = cfg['chain_names']
                    parts = [partner['chain_id'] if partner else '', receptor['chain_id'], peptide['chain_id']]
                    instance = '_'.join([pdb] + [p for p in parts if p])
                    record = {'instance': instance, 'cohort': cohort, 'pdb_id': pdb,
                              'title': (meta.get('struct') or {}).get('title'),
                              'method': (meta.get('rcsb_entry_info') or {}).get('experimental_method'),
                              'resolution': ((meta.get('rcsb_entry_info') or {}).get('resolution_combined') or [None])[0],
                              'chains': {'receptor': receptor['chain_id'],
                                         'partner': partner['chain_id'] if partner else None,
                                         'peptide': peptide['chain_id']},
                              'entity_ids': {'receptor': receptor['match']['entity_id'],
                                             'partner': partner['match']['entity_id'] if partner else None},
                              'sequences': {'receptor': receptor['seq'], 'peptide': peptide['seq'],
                                            'partner': partner['seq'] if partner else None},
                              'chain_chemistry': {'receptor': {'nonstandard': receptor['nonstandard_count'],
                                                               'mse_converted': receptor['mse_count']},
                                                  'peptide': {'nonstandard': peptide['nonstandard_count'],
                                                              'mse_converted': peptide['mse_count']}},
                              'candidate_region': {'scheme': scheme, 'range': [lo, hi],
                                                   'indices_zero_based': region,
                                                   'sequence': region_seq,
                                                   'entity_cdr3_sequence': entity_cdr3_seq,
                                                   'auth_residues': [{'auth_seq_id': r['auth_seq_id'],
                                                                      'ins': r['insertion_code']}
                                                                     for r in region_rows]},
                              'contacts': contacts,
                              'gates': gate,
                              'passes': all(gate.values())}
                    if record['passes']:
                        name = cfg['chain_names']
                        ordered = ([(name['partner'], partner['rows'])] if partner else []) + \
                                  [(name['receptor'], receptor['rows']),
                                   (name['peptide'], peptide['rows'])]
                        out_path = pdb_dir / f'{instance}.pdb'
                        write_component_pdb(out_path, ordered)
                        record['pdb_path'] = f'results/pae_analog_structures_v26/pdb/{instance}.pdb'
                        record['pdb_sha256'] = digest(out_path)
                    records.append(record)
        if combos == 0:
            exclusions.append({'pdb': pdb, 'reason': 'no_component_combinations_built'})

    summary = {'entries_planned': len(plan),
               'entries_parsed': sum(1 for s in structures.values() if 'chains' in s),
               'entries_excluded': len(exclusions),
               'components': len(records),
               'components_passed': sum(1 for r in records if r['passes']),
               'by_cohort': {c: {'components': sum(1 for r in records if r['cohort'] == c),
                                 'passed': sum(1 for r in records if r['cohort'] == c and r['passes'])}
                             for c in COHORTS}}
    freeze(OUT / 'manifest.json', {'summary': summary, 'records': records, 'exclusions': exclusions})
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {'status': 'completed', 'model_training_started': False,
                                     'af2_started': False, 'manifest_sha256': digest(OUT / 'manifest.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
