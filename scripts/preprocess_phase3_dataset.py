#!/usr/bin/env python
"""Preprocess antibody-antigen complexes for Phase 3 cross-chain fine-tuning.

Reads 48 RCSB PDB files from data/antibody_complexes/, identifies CDR regions
using a geometry-based approach (no ANARCI/Chothia renumbering needed), and
builds a dataset compatible with the BFN training pipeline.

Output: LMDB database in data/phase3_processed/ that mimics the SAbDab format.
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path

import torch
import lmdb
from Bio import PDB
from tqdm import tqdm

sys.path.insert(0, '.')
from disorderflow.utils.protein import parsers
from disorderflow.utils.protein.constants import (
    Fragment, AA, BBHeavyAtom, max_num_heavyatoms,
    restype_to_heavyatom_names,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Chothia CDR definitions (approximate for non-Chothia-numbered PDBs)
# We use structural alignment: CDRs are the loops that protrude from the Ig fold
# Fallback: mark the Fv region (first ~120 residues) for identification

# Standard germline FR/CDR boundaries (IMGT, approximate residue indices in
# sequential numbering from the start of the variable domain)
CDR_APPROX_RANGES = {
    'H': {'H1': (25, 35), 'H2': (50, 60), 'H3': (95, 110)},
    'L': {'L1': (23, 37), 'L2': (50, 58), 'L3': (88, 100)},
}

# Distance threshold for interface residue identification
INTERFACE_CUTOFF = 8.0  # Angstrom


def identify_antibody_chains(structure, chain_map):
    """Identify heavy, light, and antigen chains in the structure.

    Args:
        structure: Bio.PDB Structure object (first model)
        chain_map: dict mapping pdbcode -> {'H': str, 'L': str, 'Ag': str}

    Returns:
        (heavy_chains, light_chains, antigen_chains, all_chain_ids)
    """
    all_chains = list(structure.get_chains())
    all_ids = [c.id for c in all_chains]

    h_id = chain_map.get('H', '')
    l_id = chain_map.get('L', '')
    ag_id = chain_map.get('Ag', '')

    heavy_chains = [c for c in all_chains if c.id == h_id] if h_id else []
    light_chains = [c for c in all_chains if c.id == l_id] if l_id else []
    antigen_chains = [c for c in all_chains if c.id == ag_id] if ag_id else []

    return heavy_chains, light_chains, antigen_chains, all_ids


def guess_fv_boundary(positions_ca):
    """Guess Fv boundary by finding where CA density drops (domain boundary).

    The Fv domain (first ~120 residues) ends where there's a structural gap.
    For nanobodies (VHH), the entire chain is the variable domain (~125 res).
    """
    if len(positions_ca) < 50:
        return len(positions_ca)

    # Compute CA-CA distances along the chain
    dists = np.linalg.norm(positions_ca[1:] - positions_ca[:-1], axis=-1)
    # Find first gap > 10A after position 100 (domain boundary)
    gaps = np.where(dists > 10.0)[0]
    for g in gaps:
        if g > 90:  # After residue 90
            return int(g) + 1

    # Fallback: first 120 residues or all if shorter
    return min(120, len(positions_ca))


def label_cdr_by_geometry(positions_ca, residue_indices, chain_type='H'):
    """Identify CDR residues using geometric properties.

    CDRs are the most solvent-exposed loops of the variable domain.
    For Phase 3, a simplified approach: use positional heuristics
    based on Fv architecture.

    Returns:
        cdr_flag: np.array of CDR type (1-6) or 0 for framework
    """
    n = len(positions_ca)
    cdr_flag = np.zeros(n, dtype=np.int32)

    if n < 80:
        return cdr_flag  # Too short to be a proper Fv

    # Normalize the length to ~120 residues (standard Fv length)
    # Map approximate CDR positions to the actual sequence
    scale = n / 120.0

    ranges = CDR_APPROX_RANGES.get(chain_type, CDR_APPROX_RANGES['H'])
    for cdr_name, (start, end) in ranges.items():
        cdr_type = {
            'H1': 1, 'H2': 2, 'H3': 3,
            'L1': 4, 'L2': 5, 'L3': 6,
        }[cdr_name]

        scaled_start = int(start * scale)
        scaled_end = int(end * scale)

        # Ensure within bounds
        scaled_start = max(0, min(scaled_start, n - 1))
        scaled_end = max(scaled_start + 1, min(scaled_end, n))

        cdr_flag[scaled_start:scaled_end] = cdr_type

    return cdr_flag


def label_interface_residues(ab_positions_ca, ag_positions_ca, cutoff=INTERFACE_CUTOFF):
    """Identify antibody residues at the antigen interface.

    Returns boolean mask for interface residues on the antibody side.
    """
    if len(ag_positions_ca) == 0:
        return np.zeros(len(ab_positions_ca), dtype=bool)

    dists = np.linalg.norm(
        ab_positions_ca[:, None, :] - ag_positions_ca[None, :, :],
        axis=-1,
    )
    return dists.min(axis=1) < cutoff


def preprocess_single_complex(pdb_path, pdbcode, chain_map_entry, output_dir):
    """Preprocess a single antibody-antigen complex PDB.

    Args:
        pdb_path: path to PDB file
        pdbcode: 4-letter PDB code
        chain_map_entry: {'H': str, 'L': str, 'Ag': str}
        output_dir: directory for output files (per-complex JSON)

    Returns:
        dict mimicking SAbDab preprocessed format, or None on failure
    """
    parser = PDB.PDBParser(QUIET=True)

    try:
        model = parser.get_structure(pdbcode, pdb_path)[0]
    except Exception as e:
        logging.warning(f"[{pdbcode}] Parse error: {e}")
        return None

    h_id = chain_map_entry.get('H', '')
    l_id = chain_map_entry.get('L', '')
    ag_id = chain_map_entry.get('Ag', '')

    result = {
        'id': pdbcode,
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }

    # Parse heavy chain
    if h_id and h_id in model:
        try:
            data, seqmap = parsers.parse_biopython_structure(model[h_id])
            # Label CDR regions geometrically
            ca_mask = data['mask_heavyatom'][:, BBHeavyAtom.CA]
            positions_ca = data['pos_heavyatom'][ca_mask, BBHeavyAtom.CA].numpy()
            cdr_flag = label_cdr_by_geometry(positions_ca, list(range(len(positions_ca))), 'H')

            # Validate: need some CDR residues
            if (cdr_flag > 0).sum() < 5:
                logging.warning(f"[{pdbcode}] Heavy chain has too few CDR residues, skipping")
                data = None
            else:
                data['cdr_flag'] = torch.from_numpy(cdr_flag).long()
                result['heavy'] = data
                result['heavy_seqmap'] = seqmap
        except Exception as e:
            logging.warning(f"[{pdbcode}] Heavy chain parse error: {e}")

    # Parse light chain
    if l_id and l_id in model:
        try:
            data, seqmap = parsers.parse_biopython_structure(model[l_id])
            ca_mask = data['mask_heavyatom'][:, BBHeavyAtom.CA]
            positions_ca = data['pos_heavyatom'][ca_mask, BBHeavyAtom.CA].numpy()
            cdr_flag = label_cdr_by_geometry(positions_ca, list(range(len(positions_ca))), 'L')

            if (cdr_flag > 0).sum() < 3:
                logging.warning(f"[{pdbcode}] Light chain has too few CDR residues")
            else:
                data['cdr_flag'] = torch.from_numpy(cdr_flag).long()
                result['light'] = data
                result['light_seqmap'] = seqmap
        except Exception as e:
            logging.warning(f"[{pdbcode}] Light chain parse error: {e}")

    # Require at least heavy or light
    if result['heavy'] is None and result['light'] is None:
        logging.warning(f"[{pdbcode}] No valid antibody chains")
        return None

    # Parse antigen
    if ag_id and ag_id in model:
        try:
            data, seqmap = parsers.parse_biopython_structure(model[ag_id])
            if data['aa'].size(0) >= 5:  # Need at least 5 residues
                data['cdr_flag'] = torch.zeros_like(data['aa'])
                result['antigen'] = data
                result['antigen_seqmap'] = seqmap
        except Exception as e:
            logging.warning(f"[{pdbcode}] Antigen parse error: {e}")

    # Label interface residues on antibody chains
    if result['antigen'] is not None:
        ag_ca_mask = result['antigen']['mask_heavyatom'][:, BBHeavyAtom.CA]
        ag_positions = result['antigen']['pos_heavyatom'][ag_ca_mask, BBHeavyAtom.CA].numpy()

        for chain_key in ['heavy', 'light']:
            chain_data = result[chain_key]
            if chain_data is not None:
                ab_ca_mask = chain_data['mask_heavyatom'][:, BBHeavyAtom.CA]
                ab_positions = chain_data['pos_heavyatom'][ab_ca_mask, BBHeavyAtom.CA].numpy()
                interface_mask = label_interface_residues(ab_positions, ag_positions)
                chain_data['interface_flag'] = torch.from_numpy(interface_mask).bool()

    # Save individual complex JSON for debugging
    debug_out = os.path.join(output_dir, f"{pdbcode}_processed.json")
    serializable = {
        'pdbcode': pdbcode,
        'heavy_residues': len(result['heavy']['aa']) if result['heavy'] else 0,
        'light_residues': len(result['light']['aa']) if result['light'] else 0,
        'antigen_residues': len(result['antigen']['aa']) if result['antigen'] else 0,
        'heavy_cdr_count': int((result['heavy']['cdr_flag'] > 0).sum().item()) if result['heavy'] else 0,
        'light_cdr_count': int((result['light']['cdr_flag'] > 0).sum().item()) if result['light'] else 0,
        'heavy_interface_count': int(result['heavy']['interface_flag'].sum().item()) if (result['heavy'] and 'interface_flag' in result['heavy']) else 0,
        'light_interface_count': int(result['light']['interface_flag'].sum().item()) if (result['light'] and 'interface_flag' in result['light']) else 0,
    }
    with open(debug_out, 'w') as f:
        json.dump(serializable, f, indent=2)

    return result


def build_lmdb(data_list, lmdb_path, map_size=4 * 1024 * 1024 * 1024):
    """Build LMDB database from list of processed complex dicts."""
    db_conn = lmdb.open(
        lmdb_path,
        map_size=map_size,
        create=True,
        subdir=False,
        readonly=False,
    )
    ids = []
    with db_conn.begin(write=True, buffers=True) as txn:
        for data in tqdm(data_list, desc='Writing to LMDB'):
            if data is None:
                continue
            ids.append(data['id'])
            txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    with open(lmdb_path + '-ids', 'wb') as f:
        pickle.dump(ids, f)

    db_conn.close()
    logging.info(f"Wrote {len(ids)} entries to {lmdb_path}")
    return ids


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess antibody-antigen complexes for Phase 3 fine-tuning'
    )
    parser.add_argument(
        '--pdb_dir', type=str,
        default='./data/antibody_complexes',
        help='Directory containing PDB files'
    )
    parser.add_argument(
        '--chain_map', type=str,
        default='./data/antibody_complexes/chain_map.json',
        help='JSON file mapping pdbcode -> {H, L, Ag} chains'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./data/phase3_processed',
        help='Output directory for LMDB and debug files'
    )
    parser.add_argument(
        '--val_pdbcodes', type=str, nargs='*',
        default=['4FQI', '6APC', '6W41', '7CDI', '7DPM'],
        help='PDB codes to use for validation (default: 5 complexes)'
    )
    parser.add_argument(
        '--seed', type=int, default=2022,
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load chain map
    with open(args.chain_map, encoding='utf-8') as f:
        chain_map = json.load(f)

    # Find all PDB files
    pdb_files = sorted(Path(args.pdb_dir).glob('*.pdb'))
    logging.info(f"Found {len(pdb_files)} PDB files in {args.pdb_dir}")

    # Preprocess each complex
    all_data = []
    successes = 0
    for pdb_file in tqdm(pdb_files, desc='Preprocessing complexes'):
        pdbcode = pdb_file.stem
        if pdbcode not in chain_map:
            logging.warning(f"[{pdbcode}] Not in chain_map, skipping")
            continue

        result = preprocess_single_complex(
            str(pdb_file), pdbcode, chain_map[pdbcode], args.output_dir
        )
        if result is not None:
            all_data.append(result)
            successes += 1

    logging.info(f"Successfully processed {successes}/{len(pdb_files)} complexes")

    # Split into train/val
    import random
    rng = random.Random(args.seed)
    val_set = set(args.val_pdbcodes)
    train_data = [d for d in all_data if d['id'] not in val_set]
    val_data = [d for d in all_data if d['id'] in val_set]

    logging.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Build LMDB databases
    train_lmdb = os.path.join(args.output_dir, 'train.lmdb')
    val_lmdb = os.path.join(args.output_dir, 'val.lmdb')

    train_ids = build_lmdb(train_data, train_lmdb)
    val_ids = build_lmdb(val_data, val_lmdb)

    # Save split info
    split_info = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'train_count': len(train_ids),
        'val_count': len(val_ids),
    }
    with open(os.path.join(args.output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Print summary
    print(f"\n=== Phase 3 Dataset Summary ===")
    print(f"Total complexes: {len(all_data)}")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val:   {len(val_ids)}")

    # Statistics
    fab_count = sum(1 for d in all_data if d['heavy'] and d['light'])
    nanobody_count = sum(1 for d in all_data if d['heavy'] and not d['light'])
    has_ag_count = sum(1 for d in all_data if d['antigen'] is not None)

    print(f"  Fab:        {fab_count}")
    print(f"  Nanobody:   {nanobody_count}")
    print(f"  With Ag:    {has_ag_count}")

    total_cdr = sum(
        int((d['heavy']['cdr_flag'] > 0).sum().item()) if d['heavy'] else 0
        for d in all_data
    ) + sum(
        int((d['light']['cdr_flag'] > 0).sum().item()) if d['light'] else 0
        for d in all_data
    )
    print(f"  Total CDR residues: {total_cdr}")

    total_interface = sum(
        int(d['heavy']['interface_flag'].sum().item()) if (d['heavy'] and 'interface_flag' in d['heavy']) else 0
        for d in all_data
    ) + sum(
        int(d['light']['interface_flag'].sum().item()) if (d['light'] and 'interface_flag' in d['light']) else 0
        for d in all_data
    )
    print(f"  Interface residues: {total_interface}")

    print(f"\nOutput: {args.output_dir}/train.lmdb, val.lmdb")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
