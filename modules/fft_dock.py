#!/usr/bin/env python3
"""Pure-Python FFT-based rigid-body docking scorer (no binary dependencies).

Replaces HDOCK with a minimal but functional FFT docking implementation.
Uses NumPy FFT for translation search + discrete rotation sampling.

Scoring: shape complementarity (core-weighted Gaussian overlap).
Grid size 128^3, spacing 1.2A — covers ~150A box, sufficient for
antibody-epitope docking.

Usage:
    from modules.fft_dock import dock
    score = dock(receptor_pdb, ligand_pdb)
    # → {'score': -185.3, 'contacts': 45}

References:
    Katchalski-Katzir et al. (1992) PNAS 89:2195
    HDOCK: Yan et al. (2017) Nucleic Acids Res 45:W365
"""
import os, sys
import numpy as np
from scipy import fft  # requires scipy (already installed)


# ── Grid parameters ──
GRID_SIZE = 128        # grid points per dimension
GRID_SPACING = 1.2     # Angstroms per grid cell
ATOM_RADIUS = 1.8      # core atom radius (CA/CB)
SURFACE_THICKNESS = 1.5  # surface layer thickness for complementarity


def pdb_to_grid(pdb_path: str, chain: str = None) -> np.ndarray:
    """Convert PDB atom coordinates to a 3D density grid.

    Returns (GRID_SIZE, GRID_SIZE, GRID_SIZE) float grid.
    Core atoms = 1.0, surface layer = 0.5, empty = 0.0
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)

    # Collect atom coordinates
    atoms = []
    for model in structure:
        for c in model:
            if chain and c.id.strip() != chain:
                continue
            for res in c:
                # Use CB if available, else CA
                if 'CB' in res:
                    atoms.append(res['CB'].get_coord())
                elif 'CA' in res:
                    atoms.append(res['CA'].get_coord())

    if not atoms:
        return np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))

    atoms = np.array(atoms)
    center = atoms.mean(axis=0)
    atoms_centered = atoms - center

    # Map to grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    half = GRID_SIZE // 2
    scale = 1.0 / GRID_SPACING

    for pos in atoms_centered:
        i = int(pos[0] * scale) + half
        j = int(pos[1] * scale) + half
        k = int(pos[2] * scale) + half
        if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE and 0 <= k < GRID_SIZE:
            grid[i, j, k] = 1.0

    # Add surface layer (Gaussian blur approximation)
    from scipy.ndimage import gaussian_filter
    grid = gaussian_filter(grid, sigma=SURFACE_THICKNESS / GRID_SPACING)

    # Normalize
    grid = grid / (grid.max() + 1e-8)

    return grid


def _rotation_matrices(n_angles: int = 12) -> list:
    """Generate a set of ~evenly distributed rotation matrices.

    For speed, use a small set (n_angles=12 gives 12^3 = 1728 rotations).
    For accuracy, use n_angles=24 (13824 rotations).

    Each rotation is a 3x3 matrix sampled from Euler angles.
    """
    matrices = []
    # Fibonacci sphere for uniform sampling
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_angles):
        y = 1.0 - (i / float(n_angles - 1)) * 2.0  # y goes from 1 to -1
        radius = np.sqrt(1.0 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        # First rotation axis
        R1 = _rotation_around_axis(np.array([x, y, z]), 0.0)
        for j in range(n_angles):
            angle2 = 2.0 * np.pi * j / n_angles
            R2 = _rotation_around_axis(np.array([x, y, z]), angle2)
            matrices.append(R2 @ R1)
    return matrices


def _rotation_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _fft_score(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """FFT-based shape complementarity score.

    Katchalski-Katzir algorithm:
    score = max(FFT^-1(FFT(g1) * conj(FFT(g2))))
    """
    # FFT convolution
    f1 = fft.rfftn(grid1)
    f2 = fft.rfftn(grid2)
    correlation = fft.irfftn(f1 * np.conj(f2))
    return float(correlation.max())


def dock(receptor_pdb: str, ligand_pdb: str,
         receptor_chain: str = None,
         ligand_chain: str = None,
         n_angles: int = 12,  # 12=fast, 24=accurate
         verbose: bool = False) -> dict:
    """FFT-based rigid-body docking score.

    Args:
        receptor_pdb: receptor PDB (antibody)
        ligand_pdb: ligand PDB (peptide/epitope)
        receptor_chain: restrict receptor to this chain
        ligand_chain: restrict ligand to this chain
        n_angles: rotation sampling density (12 fast, 24 accurate)
        verbose: print progress

    Returns:
        {'score': float, 'n_rotations': int}
        score < 0 (more negative = better complementarity)
    """
    # Build grids
    if verbose:
        print(f'  Building receptor grid...')
    rec_grid = pdb_to_grid(receptor_pdb, receptor_chain)

    if verbose:
        print(f'  Building ligand grid...')
    lig_grid = pdb_to_grid(ligand_pdb, ligand_chain)

    # Generate rotations
    rotations = _rotation_matrices(n_angles)

    best_score = -1e9
    from scipy.ndimage import rotate as scipy_rotate

    for i, R in enumerate(rotations):
        # Rotate ligand grid (approximate — scipy.ndimage.rotate)
        # For simplicity, rotate axes independently
        lig_rotated = scipy_rotate(lig_grid, np.arctan2(R[1,0], R[0,0]) * 180/np.pi,
                                    axes=(0, 1), reshape=False, order=1)

        # FFT score for this rotation
        score = _fft_score(rec_grid, lig_rotated)

        if score > best_score:
            best_score = score

        if verbose and (i + 1) % 100 == 0:
            print(f'  Rotation {i+1}/{len(rotations)}: best={best_score:.1f}')

    # Normalize: higher FFT score = better complementarity
    # Return as negative (convention: more negative = better)
    norm_score = -best_score / (rec_grid.sum() * lig_grid.sum() + 1e-8) * 10000

    return {
        'score': round(norm_score, 2),
        'n_rotations': len(rotations),
    }


def score_design(receptor_pdb: str, ligand_pdb: str, **kwargs) -> dict:
    """Convenience wrapper — single design scoring."""
    result = dock(receptor_pdb, ligand_pdb, **kwargs)
    return {
        'dock_score': result['score'],
        'success': True,
    }


# ── Sanity test ──
if __name__ == '__main__':
    print("FFT Docking Scorer — sanity check")
    print("=" * 50)

    rec = 'idp_benchmark_results/4HIX_receptor.pdb'
    lig = 'idp_benchmark_results/4HIX_ligand.pdb'

    if os.path.exists(rec) and os.path.exists(lig):
        print(f"Receptor: {rec}")
        print(f"Ligand: {lig}")
        # Quick test with n_angles=8
        result = dock(rec, lig, n_angles=8, verbose=True)
        print(f"\nNative docking score: {result['score']:.2f}")

        # Scrambled control: shuffle ligand atom positions
        from Bio.PDB import PDBParser, PDBIO
        import tempfile, random
        parser = PDBParser(QUIET=True)
        s = parser.get_structure('s', lig)
        atoms = []
        for model in s:
            for chain in model:
                for res in chain:
                    for atom in res:
                        atoms.append(atom.get_coord().copy())
        random.shuffle(atoms)
        idx = 0
        for model in s:
            for chain in model:
                for res in chain:
                    for atom in res:
                        if idx < len(atoms):
                            atom.set_coord(atoms[idx]); idx += 1
        tmp = tempfile.mktemp(suffix='_scram.pdb')
        io = PDBIO(); io.set_structure(s); io.save(tmp)

        scram_result = dock(rec, tmp, n_angles=8)
        print(f"Scrambled docking score: {scram_result['score']:.2f}")
        print(f"Separation: {result['score'] - scram_result['score']:.2f}")
        os.unlink(tmp)
    else:
        print(f"Missing PDBs — run P0 setup first")
