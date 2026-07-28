#!/usr/bin/env python
"""
Preprocess SAbDab antibody-antigen complexes into Phase 3 LMDB format.

Reads metadata from the SAbDab summary TSV, loads downloaded PDB files, and
processes them using the same geometry-based CDR labeling as Phase 3.

Prerequisites:
    python scripts/download_sabdab.py   # Download PDB files first

Usage:
    python scripts/preprocess_sabdab_phase3.py                    # default paths
    python scripts/preprocess_sabdab_phase3.py --max 200          # first 200 only
    python scripts/preprocess_sabdab_phase3.py --pdb_dir custom/
"""

import os
import sys
import json
import pickle
import logging
import argparse
import random
import numpy as np
from pathlib import Path

import torch
import lmdb
import pandas as pd
from Bio import PDB
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from disorderflow.utils.protein import parsers
from disorderflow.utils.protein.constants import BBHeavyAtom
from disorderflow.utils.homology_split import (
    annotate_homology_clusters,
    grouped_train_val_split,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# CDR approximate residue ranges for geometry-based labeling
CDR_APPROX_RANGES = {
    'H': {'H1': (25, 35), 'H2': (50, 60), 'H3': (95, 110)},
    'L': {'L1': (23, 37), 'L2': (50, 58), 'L3': (88, 100)},
}

INTERFACE_CUTOFF = 8.0

# Default val set: diverse antigens, different species, mixed resolution
DEFAULT_VAL_PDBS = [
    '4FQI', '6APC', '6W41', '7CDI', '7DPM', '1BJ1', '3PP4', '6H6Y',
    '7BWJ', '7C01', '9EB7', '8F7K', '5CZX', '3U1S', '4KDT',
]


def label_cdr_by_geometry(positions_ca, chain_type='H'):
    """Geometry-based CDR labeling (same as Phase 3)."""
    n = len(positions_ca)
    cdr_flag = np.zeros(n, dtype=np.int32)
    if n < 80:
        return cdr_flag

    scale = n / 120.0
    ranges = CDR_APPROX_RANGES.get(chain_type, CDR_APPROX_RANGES['H'])
    for cdr_name, (start, end) in ranges.items():
        cdr_type = {'H1': 1, 'H2': 2, 'H3': 3, 'L1': 4, 'L2': 5, 'L3': 6}[cdr_name]
        s = max(0, min(int(start * scale), n - 1))
        e = max(s + 1, min(int(end * scale), n))
        cdr_flag[s:e] = cdr_type
    return cdr_flag


def label_interface_residues(ab_positions_ca, ag_positions_ca, cutoff=INTERFACE_CUTOFF):
    """Identify antibody residues at the antigen interface."""
    if len(ag_positions_ca) == 0:
        return np.zeros(len(ab_positions_ca), dtype=bool)
    dists = np.linalg.norm(ab_positions_ca[:, None, :] - ag_positions_ca[None, :, :], axis=-1)
    return dists.min(axis=1) < cutoff


def parse_chain_data(model, chain_id):
    """Parse a single chain from Bio.PDB model. Returns (data_dict, seqmap) or (None, None)."""
    if chain_id not in model:
        return None, None
    try:
        data, seqmap = parsers.parse_biopython_structure(model[chain_id])
        return data, seqmap
    except Exception as e:
        logger.debug(f"Chain {chain_id} parse error: {e}")
        return None, None


def preprocess_one(pdbcode, pdb_path, h_chain, l_chain, ag_chains):
    """Preprocess a single SAbDab complex into Phase 3 format.

    Returns dict or None on failure.
    """
    parser = PDB.MMCIFParser(QUIET=True) if str(pdb_path).lower().endswith('.cif') else PDB.PDBParser(QUIET=True)
    try:
        model = parser.get_structure(pdbcode, pdb_path)[0]
    except Exception as e:
        logger.debug(f"[{pdbcode}] Parse error: {e}")
        return None

    result = {
        'id': pdbcode,
        'pdb_id': str(pdbcode).casefold(),
        'heavy_chain_id': h_chain,
        'light_chain_id': l_chain,
        'antigen_chain_ids': list(ag_chains),
        'heavy': None,
        'light': None,
        'antigen': None,
    }

    # Parse heavy chain
    if h_chain:
        data, seqmap = parse_chain_data(model, h_chain)
        if data is not None:
            ca_mask = data['mask_heavyatom'][:, BBHeavyAtom.CA]
            positions_ca = data['pos_heavyatom'][ca_mask, BBHeavyAtom.CA].numpy()
            cdr_flag = label_cdr_by_geometry(positions_ca, 'H')
            if (cdr_flag > 0).sum() >= 5:
                data['cdr_flag'] = torch.from_numpy(cdr_flag).long()
                result['heavy'] = data
            else:
                logger.debug(f"[{pdbcode}] Heavy CDR too small: {(cdr_flag > 0).sum()}")

    # Parse light chain
    if l_chain:
        data, seqmap = parse_chain_data(model, l_chain)
        if data is not None:
            ca_mask = data['mask_heavyatom'][:, BBHeavyAtom.CA]
            positions_ca = data['pos_heavyatom'][ca_mask, BBHeavyAtom.CA].numpy()
            cdr_flag = label_cdr_by_geometry(positions_ca, 'L')
            if (cdr_flag > 0).sum() >= 3:
                data['cdr_flag'] = torch.from_numpy(cdr_flag).long()
                result['light'] = data
            else:
                logger.debug(f"[{pdbcode}] Light CDR too small")

    # Need at least one antibody chain
    if result['heavy'] is None and result['light'] is None:
        return None

    # Parse antigen chains (merge if multiple)
    ag_data_list = []
    for ag_id in ag_chains:
        data, __ = parse_chain_data(model, ag_id)
        if data is not None and data['aa'].size(0) >= 5:
            data['cdr_flag'] = torch.zeros_like(data['aa'])
            ag_data_list.append(data)

    if ag_data_list:
        # Merge antigen chains
        if len(ag_data_list) == 1:
            result['antigen'] = ag_data_list[0]
        else:
            # Concatenate all antigen chains
            merged = {}
            for key in ag_data_list[0].keys():
                values = [d[key] for d in ag_data_list]
                if isinstance(values[0], torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], list):
                    merged[key] = sum(values, start=[])
                else:
                    merged[key] = values[0]  # take first for metadata
            result['antigen'] = merged

    # Label interface residues
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

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    to_sequence = lambda values: ''.join(
        alphabet[int(value)] if int(value) < len(alphabet) else 'X' for value in values)
    if result['heavy'] is not None:
        result['vh_sequence'] = to_sequence(result['heavy']['aa'])
        h3_mask = result['heavy']['cdr_flag'] == 3
        result['cdr_h3_sequence'] = to_sequence(result['heavy']['aa'][h3_mask])
    if result['light'] is not None:
        result['vl_sequence'] = to_sequence(result['light']['aa'])
    if result['antigen'] is not None:
        result['antigen_sequence'] = to_sequence(result['antigen']['aa'])

    return result


def build_lmdb(data_list, lmdb_path, map_size=16 * 1024 * 1024 * 1024):
    """Build LMDB database."""
    db_conn = lmdb.open(lmdb_path, map_size=map_size, create=True, subdir=False, readonly=False)
    ids = []
    n_written = 0
    with db_conn.begin(write=True, buffers=True) as txn:
        for data in tqdm(data_list, desc='Writing to LMDB'):
            if data is None:
                continue
            ids.append(data['id'])
            txn.put(data['id'].encode('utf-8'), pickle.dumps(data))
            n_written += 1
    with open(lmdb_path + '-ids', 'wb') as f:
        pickle.dump(ids, f)
    db_conn.close()
    logger.info(f"Wrote {n_written} entries to {lmdb_path}")
    return ids


def main():
    parser = argparse.ArgumentParser(description='Preprocess SAbDab complexes for Phase 3')
    parser.add_argument('--summary', default=os.path.join(PROJECT_DIR, 'data', 'sabdab_summary_all.tsv'))
    parser.add_argument('--pdb_dir', default=os.path.join(PROJECT_DIR, 'data', 'all_structures', 'chothia'))
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_DIR, 'data', 'sabdab_phase3_processed'))
    parser.add_argument('--max', type=int, default=None, help='Max complexes to process')
    parser.add_argument('--val_ratio', type=float, default=0.10, help='Validation ratio')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--resolution_cutoff', type=float, default=3.0)
    parser.add_argument('--mmseqs', default='mmseqs', help='MMseqs2 executable')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load SAbDab summary
    logger.info(f"Loading SAbDab summary from {args.summary}...")
    df = pd.read_csv(args.summary, sep='\t')

    # Filter for quality complexes
    has_h = df['Hchain'].notna() & (df['Hchain'] != 'NA') & (df['Hchain'] != '')
    has_ag = df['antigen_chain'].notna() & (df['antigen_chain'] != 'NA') & (df['antigen_chain'] != '')
    protein_ag = df['antigen_type'].str.contains('protein', na=False, case=False)
    not_scfv = ~df.get('scfv', pd.Series([False] * len(df))).astype(bool)
    resolution = pd.to_numeric(df['resolution'], errors='coerce')
    res_ok = (resolution <= args.resolution_cutoff) & (resolution > 0)

    mask = has_h & has_ag & protein_ag & not_scfv & res_ok
    filtered = df[mask]
    logger.info(f"Filtered to {len(filtered)} entries (res <= {args.resolution_cutoff}, protein Ag, has H+Ag)")

    # Get unique PDB entries
    unique_pdbs = filtered.drop_duplicates(subset=['pdb']).copy()
    logger.info(f"Unique PDBs: {len(unique_pdbs)}")

    if args.max:
        unique_pdbs = unique_pdbs.head(args.max)
        logger.info(f"Limited to {len(unique_pdbs)} PDBs")

    # Process each complex
    all_data = []
    n_skipped = 0
    n_missing = 0
    for _, row in tqdm(unique_pdbs.iterrows(), total=len(unique_pdbs), desc='Preprocessing'):
        pdbcode = row['pdb']
        pdb_path = os.path.join(args.pdb_dir, f"{pdbcode}.pdb")

        if not os.path.exists(pdb_path):
            n_missing += 1
            continue

        h_chain = str(row['Hchain']) if pd.notna(row['Hchain']) else None
        l_chain = str(row['Lchain']) if pd.notna(row['Lchain']) and row['Lchain'] != 'NA' else None
        ag_str = str(row['antigen_chain']) if pd.notna(row['antigen_chain']) else ''
        ag_chains = [c.strip() for c in ag_str.split('|') if c.strip()]

        result = preprocess_one(pdbcode, pdb_path, h_chain, l_chain, ag_chains)
        if result is not None:
            all_data.append(result)
        else:
            n_skipped += 1

    logger.info(f"Processed: {len(all_data)} succeeded, {n_skipped} skipped, {n_missing} PDBs missing")

    # Cluster every biological identity used for generalization claims, then
    # assign connected clusters as indivisible train/validation groups.
    annotate_homology_clusters(
        all_data,
        {'vh_sequence': 0.5, 'vl_sequence': 0.5,
         'cdr_h3_sequence': 0.5, 'antigen_sequence': 0.3},
        executable=args.mmseqs,
    )
    val_set = set(DEFAULT_VAL_PDBS)
    cluster_fields = [
        'vh_sequence_cluster_id', 'vl_sequence_cluster_id',
        'cdr_h3_sequence_cluster_id', 'antigen_sequence_cluster_id',
    ]
    train_data, val_data = grouped_train_val_split(
        all_data, args.val_ratio, args.seed, cluster_fields, val_set)

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Build LMDBs
    build_lmdb(train_data, os.path.join(args.output_dir, 'train.lmdb'))
    build_lmdb(val_data, os.path.join(args.output_dir, 'val.lmdb'))

    # Summary
    fab_count = sum(1 for d in all_data if d['heavy'] and d['light'])
    nano_count = sum(1 for d in all_data if d['heavy'] and not d['light'])
    has_ag_count = sum(1 for d in all_data if d['antigen'] is not None)

    print(f"\n=== SAbDab Phase 3 Dataset Summary ===")
    print(f"Total complexes processed: {len(all_data)}")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Fab + Ag:       {fab_count}")
    print(f"  Nanobody + Ag:  {nano_count}")
    print(f"  Has antigen:    {has_ag_count}")

    # Save metadata
    meta = {
        'train_count': len(train_data),
        'val_count': len(val_data),
        'total': len(all_data),
        'resolution_cutoff': args.resolution_cutoff,
        'sources': 'SAbDab summary + RCSB PDB download',
        'train_ids': [d['id'] for d in train_data],
        'val_ids': [d['id'] for d in val_data],
        'split_strategy': 'MMseqs2 connected homology clusters',
        'cluster_fields': cluster_fields,
    }
    with open(os.path.join(args.output_dir, 'dataset_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput: {args.output_dir}/")
    print(f"  train.lmdb ({os.path.getsize(os.path.join(args.output_dir, 'train.lmdb')) / 1024**2:.1f} MB)")
    print(f"  val.lmdb ({os.path.getsize(os.path.join(args.output_dir, 'val.lmdb')) / 1024**2:.1f} MB)")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
