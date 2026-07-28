#!/usr/bin/env python
"""
Build Phase3 expanded LMDB with AF2 proxy confidence labels.

Computes proxy labels from crystal structure data:
  - pLDDT: derived from CDR type (framework 0.95, CDR 0.75, interface 0.90)
           with B-factor-like Gaussian noise for per-residue variation
  - ipTM:  from inter-chain contact density between antibody and antigen
  - PAE:   from distance matrix / 31.0 (AF2 normalization scale)

These proxy labels provide realistic variation to keep confidence heads
calibrated during joint co-design + confidence training.

Output: data/phase3_expanded_conf/train.lmdb, val.lmdb
Each entry = original Phase3 structure dict + af2_plddt, af2_iptm, af2_pae_matrix.

Usage:
  python scripts/build/build_phase3_confidence.py
"""

import os
import sys
import lmdb
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_project_root)

SRC_DIR = 'data/phase3_expanded'
DST_DIR = 'data/phase3_expanded_conf'

CONTACT_CUTOFF = 5.0       # Heavy atom contact distance (Å)
CONTACT_ATOMS = [0, 1, 2]  # N, CA, C for contact detection
PAE_SCALE = 31.0            # AF2 PAE normalization
PAD_AA = 21                 # Padding token


def compute_chain_plddt(chain, noise_std=0.02):
    """Per-residue pLDDT proxy from CDR type + noise.

    Framework: 0.95, CDR: 0.75, interface: 0.90.
    Adds Gaussian noise for per-residue variation.
    """
    cdr_flag = chain.get('cdr_flag', None)
    interface_flag = chain.get('interface_flag', None)
    n = len(chain['aa'])

    plddt = torch.full((n,), 0.95)  # default: framework

    if cdr_flag is not None:
        is_cdr = cdr_flag > 0
        plddt[is_cdr] = 0.75

    if interface_flag is not None:
        is_interface = interface_flag > 0
        plddt[is_interface] = 0.90

    # Add noise for per-residue variation
    noise = torch.randn(n) * noise_std
    plddt = (plddt + noise).clamp(0.05, 1.0)

    return plddt


def compute_iptm(entry, cutoff=CONTACT_CUTOFF):
    """ipTM proxy from antibody-antigen contact density.

    Counts heavy atom contacts between antibody (H+L) and antigen.
    Normalizes by geometric mean of chain lengths.
    """
    # Collect antibody CA positions
    ab_positions = []
    for chain_name in ['heavy', 'light']:
        chain = entry.get(chain_name)
        if chain is None:
            continue
        pos = chain['pos_heavyatom']  # (L, 37, 3)
        mask = chain['mask_heavyatom']  # (L, 37)
        for atom_idx in CONTACT_ATOMS:
            atom_mask = mask[:, atom_idx].bool()
            if atom_mask.any():
                ab_positions.append(pos[atom_mask, atom_idx])

    if not ab_positions:
        return torch.tensor(0.5)

    ab_pos = torch.cat(ab_positions, dim=0)  # (N_ab_atoms, 3)

    # Collect antigen positions
    ag_chain = entry.get('antigen')
    if ag_chain is None:
        return torch.tensor(0.5)

    ag_positions = []
    pos = ag_chain['pos_heavyatom']
    mask = ag_chain['mask_heavyatom']
    for atom_idx in CONTACT_ATOMS:
        atom_mask = mask[:, atom_idx].bool()
        if atom_mask.any():
            ag_positions.append(pos[atom_mask, atom_idx])

    if not ag_positions:
        return torch.tensor(0.5)

    ag_pos = torch.cat(ag_positions, dim=0)

    # Contact density
    dists = torch.cdist(ab_pos, ag_pos)
    n_contacts = (dists < cutoff).sum().float()

    # Normalize by geometric mean of atom counts (capped)
    n_ab = len(ab_pos)
    n_ag = len(ag_pos)
    norm = torch.sqrt(torch.tensor(n_ab * n_ag, dtype=torch.float))

    # ipTM = contact_density * scale
    contact_density = n_contacts / norm.clamp(min=1)
    # Map to [0.2, 0.95] range — typical AF2 ipTM range
    iptm = 0.2 + 0.75 * (contact_density / 0.3).clamp(0, 1)

    return iptm


def compute_pae(all_positions):
    """PAE matrix proxy from distance matrix.

    PAE_ij = distance_ij / 31.0, clipped to [0, 31] (AF2 convention).
    """
    dists = torch.cdist(all_positions, all_positions)  # (L, L)
    pae = dists / PAE_SCALE
    pae = pae.clamp(0, 1)  # Normalize to [0, 1]
    return pae


def process_entry(entry):
    """Add proxy confidence labels to a Phase3 entry."""
    # Per-chain pLDDT
    plddt_parts = []
    all_ca_pos = []
    all_ca_mask = []

    for chain_name in ['heavy', 'light', 'antigen']:
        chain = entry.get(chain_name)
        if chain is None:
            continue
        plddt = compute_chain_plddt(chain)
        plddt_parts.append(plddt)

        ca_pos = chain['pos_heavyatom'][:, 1]  # CA
        ca_mask = chain['mask_heavyatom'][:, 1].bool()
        all_ca_pos.append(ca_pos)
        all_ca_mask.append(ca_mask)

    # Merged pLDDT (concatenate in merge_chains order)
    if plddt_parts:
        entry['af2_plddt'] = torch.cat(plddt_parts, dim=0)
    else:
        entry['af2_plddt'] = torch.tensor([0.9])

    # ipTM
    entry['af2_iptm'] = compute_iptm(entry)

    # PAE from full CA distance matrix
    merged_ca = torch.cat([p[m] for p, m in zip(all_ca_pos, all_ca_mask)], dim=0)
    entry['af2_pae_matrix'] = compute_pae(merged_ca)

    return entry


def copy_lmdb_with_labels(src_path, dst_path):
    """Copy LMDB, adding confidence labels to each entry."""
    # Read source
    src_env = lmdb.open(src_path, readonly=True, lock=False, subdir=False)
    entries = {}
    with src_env.begin() as txn:
        for key_bytes, value_bytes in tqdm(list(txn.cursor()), desc=f'  Reading {os.path.basename(src_path)}'):
            key = key_bytes.decode()
            entries[key] = pickle.loads(value_bytes)
    src_env.close()

    # Process
    for key in tqdm(entries, desc='  Computing labels'):
        entries[key] = process_entry(entries[key])

    # Write destination
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    dst_env = lmdb.open(dst_path, map_size=8 * 1024 * 1024 * 1024, subdir=False)
    with dst_env.begin(write=True) as txn:
        for key, entry in tqdm(entries.items(), desc='  Writing'):
            txn.put(key.encode(), pickle.dumps(entry))
    dst_env.close()

    # Copy -ids file
    ids_path = src_path + '-ids'
    if os.path.exists(ids_path):
        import shutil
        dst_ids_path = dst_path + '-ids'
        shutil.copy(ids_path, dst_ids_path)

    return len(entries)


def main():
    print("=" * 70)
    print("Building Phase3 expanded LMDB with AF2 proxy confidence labels")
    print("=" * 70)

    total = 0
    for split in ['train', 'val']:
        src = os.path.join(SRC_DIR, f'{split}.lmdb')
        dst = os.path.join(DST_DIR, f'{split}.lmdb')

        if not os.path.exists(src):
            print(f"[SKIP] {src} not found")
            continue

        print(f"\n--- {split} ---")
        n = copy_lmdb_with_labels(src, dst)
        total += n
        print(f"  Done: {n} entries -> {dst}")

    print(f"\n{'=' * 70}")
    print(f"Total: {total} entries written to {DST_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
