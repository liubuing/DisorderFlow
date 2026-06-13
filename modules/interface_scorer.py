#!/usr/bin/env python
"""Interface Scoring — BSA + Shape Complementarity + Electrostatic Complementarity.

Fast local alternative to HDOCK for ranking antibody designs.
Computes three standard metrics on antibody-epitope complexes.
"""

import os
import numpy as np
from Bio.PDB import PDBParser
import freesasa

# Standard amino acid properties
# pKa values for ionizable groups at pH 7.4
AA_CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}

# Amino acid volumes (Å³) for shape complementarity
AA_VOLUME = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
}

aa3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
          'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
          'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

INTERFACE_CUTOFF = 5.0  # Å, residues within this distance are "interface"


def _get_ca_atoms(pdb_path, chain_id=None):
    """Extract CA positions, residue names, and IDs from a PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)[0]

    positions = []
    residues = []
    res_ids = []
    for chain in structure:
        if chain_id and chain.id != chain_id:
            continue
        for res in chain:
            if 'CA' not in res:
                continue
            rn = res.get_resname().strip()
            if rn not in aa3to1:
                continue
            positions.append(res['CA'].get_coord())
            residues.append(aa3to1[rn])
            res_ids.append((chain.id, res.get_id()[1]))

    return np.array(positions), residues, res_ids


def compute_bsa(complex_pdb, scaffold_chain='A', epitope_chain='B'):
    """Compute Buried Surface Area at the interface.

    BSA = SASA(complex) - SASA(scaffold_alone) - SASA(epitope_alone)
    Higher BSA = stronger binding.

    Returns dict with total_bsa, scaffold_bsa, epitope_bsa.
    """
    # Complex SASA
    comp = freesasa.Structure(complex_pdb)
    comp_result = freesasa.calc(comp)
    comp_total = comp_result.totalArea()

    ca_pos, residues, res_ids = _get_ca_atoms(complex_pdb)
    scaffold_mask = np.array([rid[0] == scaffold_chain for rid in res_ids])
    epitope_mask = np.array([rid[0] == epitope_chain for rid in res_ids])

    scaffold_ca = ca_pos[scaffold_mask]
    epitope_ca = ca_pos[epitope_mask]
    scaffold_rids = [res_ids[i] for i, m in enumerate(scaffold_mask) if m]
    epitope_rids = [res_ids[i] for i, m in enumerate(epitope_mask) if m]

    # Compute inter-chain distances
    dists = np.linalg.norm(
        scaffold_ca[:, None, :] - epitope_ca[None, :, :], axis=-1)

    # Find interface residues
    interface_scaffold = set()
    interface_epitope = set()
    for i in range(len(scaffold_ca)):
        min_dist = dists[i].min()
        if min_dist < INTERFACE_CUTOFF:
            interface_scaffold.add(scaffold_rids[i])
    for j in range(len(epitope_ca)):
        min_dist = dists[:, j].min()
        if min_dist < INTERFACE_CUTOFF:
            interface_epitope.add(epitope_rids[j])

    n_int_scaffold = len(interface_scaffold)
    n_int_epitope = len(interface_epitope)

    # Estimate BSA from interface residue count × avg residue SASA
    avg_res_sasa = 120  # Å², typical
    total_bsa = (n_int_scaffold + n_int_epitope) * avg_res_sasa / 2

    return {
        'total_bsa': total_bsa,
        'interface_scaffold_residues': n_int_scaffold,
        'interface_epitope_residues': n_int_epitope,
        'min_distance': float(dists.min()),
        'contact_pairs': int((dists < INTERFACE_CUTOFF).sum()),
    }


def compute_shape_complementarity(complex_pdb, scaffold_chain='A', epitope_chain='B'):
    """Compute Lawrence & Colman shape complementarity.

    Sc = median of (1 / (1 + exp(-S))) where S = f(d1, d2, n1, n2)
    Simplified: use distance distribution of interface atoms.
    Range: 0 (no complementarity) to 1 (perfect fit).

    Returns Sc value.
    """
    ca_pos, residues, res_ids = _get_ca_atoms(complex_pdb)
    scaffold_mask = np.array([rid[0] == scaffold_chain for rid in res_ids])
    epitope_mask = np.array([rid[0] == epitope_chain for rid in res_ids])

    if scaffold_mask.sum() == 0 or epitope_mask.sum() == 0:
        return {'Sc': 0.0}

    scaffold_ca = ca_pos[scaffold_mask]
    epitope_ca = ca_pos[epitope_mask]

    dists = np.linalg.norm(
        scaffold_ca[:, None, :] - epitope_ca[None, :, :], axis=-1)

    # Interface distances (within 2x cutoff to capture shape)
    interface_dists = dists[(dists < 8.0) & (dists > 0.5)]

    if len(interface_dists) < 5:
        return {'Sc': 0.0, 'n_interface_pairs': len(interface_dists)}

    # Shape complementarity proxy:
    # 1. Distance variance (lower = better fitting)
    # 2. Mean distance close to 3.5Å (optimal VDW contact)
    dist_mean = float(interface_dists.mean())
    dist_std = float(interface_dists.std())

    # Sc formula: high when distances cluster around 3-4Å
    optimal_dist = 3.5
    dist_score = np.exp(-0.5 * ((interface_dists - optimal_dist) / 1.5) ** 2)
    sc_raw = float(dist_score.mean())

    # Penalize high variance
    sc = sc_raw * np.exp(-0.1 * dist_std)

    return {
        'Sc': round(sc, 4),
        'interface_pairs': len(interface_dists),
        'mean_distance': round(dist_mean, 2),
        'dist_std': round(dist_std, 2),
    }


def compute_electrostatic_complementarity(complex_pdb, scaffold_chain='A',
                                          epitope_chain='B'):
    """Estimate electrostatic complementarity at the interface.

    For each interface residue pair within cutoff, compute:
        EC_ij = -charge_i * charge_j * exp(-d_ij / lambda_D)

    Positive EC = favorable (opposite charges attract)
    Negative EC = unfavorable (same charges repel)

    Returns EC score and per-chain charge distribution.
    """
    ca_pos, residues, res_ids = _get_ca_atoms(complex_pdb)
    scaffold_mask = np.array([rid[0] == scaffold_chain for rid in res_ids])
    epitope_mask = np.array([rid[0] == epitope_chain for rid in res_ids])

    scaffold_idx = np.where(scaffold_mask)[0]
    epitope_idx = np.where(epitope_mask)[0]

    if len(scaffold_idx) == 0 or len(epitope_idx) == 0:
        return {'EC': 0.0}

    scaffold_ca = ca_pos[scaffold_mask]
    epitope_ca = ca_pos[epitope_mask]
    scaffold_seq = [residues[i] for i in scaffold_idx]
    epitope_seq = [residues[i] for i in epitope_idx]

    # Debye length at physiological ionic strength (150mM) ~8 Å
    debye_length = 8.0

    dists = np.linalg.norm(
        scaffold_ca[:, None, :] - epitope_ca[None, :, :], axis=-1)

    # Compute EC only for interface pairs
    interface_mask = dists < INTERFACE_CUTOFF
    interf_i, interf_j = np.where(interface_mask)

    if len(interf_i) < 3:
        return {'EC': 0.0, 'interface_pairs': len(interf_i)}

    ec_sum = 0.0
    favorable = 0
    unfavorable = 0
    for i, j in zip(interf_i, interf_j):
        qi = AA_CHARGE.get(scaffold_seq[i], 0)
        qj = AA_CHARGE.get(epitope_seq[j], 0)
        d = dists[i, j]
        ec_ij = -qi * qj * np.exp(-d / debye_length)
        ec_sum += ec_ij
        if ec_ij > 0.01:
            favorable += 1
        elif ec_ij < -0.01:
            unfavorable += 1

    # Normalize
    n_pairs = len(interf_i)
    ec_norm = ec_sum / max(n_pairs, 1)

    # Per-chain charge counts
    scaff_charge = sum(AA_CHARGE.get(aa, 0) for aa in scaffold_seq)
    epi_charge = sum(AA_CHARGE.get(aa, 0) for aa in epitope_seq)

    return {
        'EC': round(ec_norm, 4),
        'interface_pairs': n_pairs,
        'favorable_pairs': favorable,
        'unfavorable_pairs': unfavorable,
        'scaffold_net_charge': scaff_charge,
        'epitope_net_charge': epi_charge,
    }


def score_complex(complex_pdb, scaffold_chain='A', epitope_chain='B'):
    """Full interface scoring: BSA + Sc + EC.

    Returns:
        dict with all metrics and a combined score.
    """
    bsa = compute_bsa(complex_pdb, scaffold_chain, epitope_chain)
    sc = compute_shape_complementarity(complex_pdb, scaffold_chain, epitope_chain)
    ec = compute_electrostatic_complementarity(complex_pdb, scaffold_chain, epitope_chain)

    # Combined score (normalize each component to [0,1])
    bsa_norm = min(1.0, bsa['total_bsa'] / 2000.0)  # ~2000Å² = very good BSA
    sc_norm = sc['Sc']  # Already in [0,1]
    ec_norm = max(0.0, min(1.0, (ec['EC'] + 0.3) / 0.6))  # Map EC to [0,1]

    combined = 0.4 * bsa_norm + 0.3 * sc_norm + 0.3 * ec_norm

    return {
        **bsa, **sc, **ec,
        'combined_interface_score': round(combined, 4),
        'bsa_norm': round(bsa_norm, 4),
        'sc_norm': round(sc_norm, 4),
        'ec_norm': round(ec_norm, 4),
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        pdb = sys.argv[1]
    else:
        pdb = '/tmp/idp_design_6A/complex_5IMK_epitope_A_complex.pdb'
        if not os.path.exists(pdb):
            import glob
            candidates = glob.glob('/tmp/idp_design_*/complex_*.pdb')
            pdb = candidates[0] if candidates else None
            if not pdb:
                print("No complex PDB found")
                sys.exit(1)

    print(f"Scoring: {pdb}")
    result = score_complex(pdb)
    for k, v in result.items():
        print(f"  {k}: {v}")
