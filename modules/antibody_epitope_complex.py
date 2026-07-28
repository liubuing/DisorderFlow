#!/usr/bin/env python
"""Antibody-Epitope Complex Builder — position antibody scaffold facing epitope.

Creates a composite PDB where the antibody scaffold and epitope are placed
at a design-relevant distance (15-20 Å), suitable for BFN antibody CDR design.

The scaffold's CDR face is oriented toward the epitope center, simulating
the antibody-antigen docking geometry.
"""

import os
import tempfile
import numpy as np


def _read_ca_positions(pdb_path, chain_id=None):
    """Read CA atom coordinates from a PDB file.

    Returns:
        positions: (N, 3) array of CA coordinates
        residue_ids: list of (chain, resseq) tuples
        atom_lines: list of (line, chain, resseq) for all atoms
    """
    positions = []
    residue_ids = []
    atom_lines = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                atom_lines.append((line, None, None))
                continue
            cid = line[21:22].strip()
            if chain_id and cid != chain_id:
                atom_lines.append((line, cid, None))
                continue
            atom_name = line[12:16].strip()
            try:
                resid = int(line[22:26])
            except ValueError:
                atom_lines.append((line, cid, None))
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_lines.append((line, cid, resid))
            if atom_name == 'CA':
                positions.append([x, y, z])
                residue_ids.append((cid, resid))
    return np.array(positions), residue_ids, atom_lines


def position_epitope(scaffold_pdb, epitope_pdb, scaffold_chain='A',
                     epitope_chain='A', distance=18.0, output_dir=None):
    """Create combined PDB with epitope positioned in front of scaffold CDRs.

    Strategy:
      1. Compute epitope center of mass (CA atoms)
      2. Compute scaffold CDR center (residues 26-33, 51-58, 97-113 for nanobody)
      3. Compute scaffold global center
      4. Place epitope along the vector from scaffold center toward CDR face
         at the specified distance from the CDR center.

    Args:
        scaffold_pdb: path to antibody scaffold PDB
        epitope_pdb: path to epitope PDB
        scaffold_chain: chain ID of the antibody
        epitope_chain: chain ID for the epitope in output
        distance: target distance from CDR center to epitope center (Å)
        output_dir: output directory

    Returns:
        dict with keys: pdb_path, combined_pdb, distance, epitope_center, cdr_center
    """
    # Read CA positions
    scaff_ca, scaff_ids, scaff_atoms = _read_ca_positions(
        scaffold_pdb, scaffold_chain)
    epitope_ca, epitope_ids, epitope_atoms = _read_ca_positions(
        epitope_pdb, epitope_chain)

    if len(scaff_ca) == 0:
        raise ValueError(f"No CA atoms found in scaffold {scaffold_pdb}")
    if len(epitope_ca) == 0:
        raise ValueError(f"No CA atoms found in epitope {epitope_pdb}")

    # CDR positions (nanobody CDR definitions: H1~26-33, H2~51-58, H3~97-113)
    scaffold_offset = scaff_ids[0][1] - 1  # residue numbering offset
    cdr_ranges = [
        (26 - scaffold_offset, 33 - scaffold_offset),
        (51 - scaffold_offset, 58 - scaffold_offset),
        (97 - scaffold_offset, 113 - scaffold_offset),
    ]

    cdr_indices = []
    for cdr_start, cdr_end in cdr_ranges:
        cdr_start = max(0, cdr_start)
        cdr_end = min(len(scaff_ca), cdr_end)
        cdr_indices.extend(range(cdr_start, cdr_end))

    if cdr_indices:
        cdr_center = scaff_ca[cdr_indices].mean(axis=0)
    else:
        # Fallback: use geometric center of scaffold
        cdr_center = scaff_ca.mean(axis=0)

    scaffold_center = scaff_ca.mean(axis=0)
    epitope_center = epitope_ca.mean(axis=0)

    # Direction: from scaffold center through CDR center
    direction = cdr_center - scaffold_center
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 0.5:
        # CDR center coincides with scaffold center — pick z-axis
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = direction / dir_norm

    # Target position: epitope center placed at CDR_center + distance * direction
    target_epitope_center = cdr_center + distance * direction
    translation = target_epitope_center - epitope_center

    # Build combined PDB
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='complex_')
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(scaffold_pdb))[0]
    epi_base = os.path.splitext(os.path.basename(epitope_pdb))[0]
    out_path = os.path.join(output_dir, f'{base}_{epi_base}_complex.pdb')

    TER_LINE = f"{'TER':6s}{'':4s}{'':3s} {scaffold_chain:1s}{scaff_ids[-1][1]:4d}\n"

    with open(out_path, 'w') as f:
        # Scaffold atoms — keep ONLY the requested scaffold chain. _read_ca_positions
        # passes through atoms of other chains (cid in the tuple), which for a
        # multi-chain scaffold PDB (e.g. 5IMK has chain A = a 118-residue antigen)
        # would otherwise be written into the complex and collide with the
        # epitope's chain, producing a wrong antigen. Drop every non-scaffold atom.
        for line, cid, resid in scaff_atoms:
            if (line.startswith('ATOM') or line.startswith('HETATM')) and cid == scaffold_chain:
                f.write(line)
        f.write(TER_LINE)

        # Epitope atoms (translated)
        for line, cid, resid in epitope_atoms:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = float(line[30:38]) + translation[0]
                y = float(line[38:46]) + translation[1]
                z = float(line[46:54]) + translation[2]
                # Use epitope_chain for output
                new_line = (f"{line[:21]}{epitope_chain}{line[22:30]}"
                           f"{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                f.write(new_line)
            elif line.startswith('TER'):
                f.write(f"{'TER':6s}{'':4s}{'':3s} {epitope_chain:1s}{epitope_ids[-1][1]:4d}\n")
        f.write('END\n')

    actual_distance = np.linalg.norm(cdr_center - target_epitope_center)

    return {
        'pdb_path': out_path,
        'epitope_center': target_epitope_center.tolist(),
        'cdr_center': cdr_center.tolist(),
        'distance': float(actual_distance),
        'translation': translation.tolist(),
    }


def analyze_contact_potential(complex_pdb, epitope_chain='B',
                              distance_cutoff=20.0):
    """Quick check: are any scaffold atoms within contact distance of epitope?

    Returns counts of scaffold residues within the cutoff of epitope CA atoms.
    Used as a sanity check before running BFN design.
    """
    epitope_ca, _, _ = _read_ca_positions(complex_pdb, epitope_chain)
    scaffold_ca, scaff_ids, _ = _read_ca_positions(
        complex_pdb, 'A')  # assume scaffold is chain A

    if len(epitope_ca) == 0 or len(scaffold_ca) == 0:
        return {'nearby_residues': 0, 'min_distance': float('inf')}

    # Compute all pairwise distances
    dists = np.linalg.norm(
        scaffold_ca[:, None, :] - epitope_ca[None, :, :], axis=-1
    )
    min_per_scaff = dists.min(axis=1)
    nearby = int((min_per_scaff < distance_cutoff).sum())

    return {
        'nearby_residues': nearby,
        'min_distance': float(min_per_scaff.min()),
        'mean_min_distance': float(min_per_scaff.mean()),
    }
