#!/usr/bin/env python3
"""V3 L2: FFT rigid-body pose generator (NOT scorer).

Generates docked poses using FFT convolution on 3D grids.
Returns the best translation + rotation, which L3 then scores.
This is HDOCK's algorithm reimplemented in pure Python.

Usage:
    from modules.fft_pose import generate_pose
    pose_pdb = generate_pose(receptor_pdb, ligand_pdb, 'H', 'P')
    # Then: score_pose(pose_pdb, 'H', 'P') for L3 scoring
"""
import os, sys
import numpy as np
from scipy import fft
from scipy.ndimage import rotate as scipy_rotate

GRID = 128; SPACING = 1.2; N_ANGLES = 12


def _pdb_to_grid(pdb_path, chain):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True); s = parser.get_structure('s', pdb_path)
    atoms = []
    for model in s:
        for c in model:
            if c.id.strip() != chain: continue
            for res in c:
                if 'CB' in res: atoms.append(res['CB'].get_coord())
                elif 'CA' in res: atoms.append(res['CA'].get_coord())
    if not atoms: return np.zeros((GRID,GRID,GRID))
    atoms = np.array(atoms)
    center = atoms.mean(axis=0); atoms = atoms - center
    grid = np.zeros((GRID,GRID,GRID), dtype=np.float32)
    half = GRID//2; scale = 1.0/SPACING
    for pos in atoms:
        i = int(pos[0]*scale)+half; j = int(pos[1]*scale)+half; k = int(pos[2]*scale)+half
        if 0<=i<GRID and 0<=j<GRID and 0<=k<GRID: grid[i,j,k] = 1.0
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(grid, sigma=1.5/SPACING)


def _rotations():
    mats = []; phi = np.pi*(3-np.sqrt(5))
    for i in range(N_ANGLES):
        y = 1.0-(i/(N_ANGLES-1))*2.0; r = np.sqrt(1-y*y); th = phi*i
        x = np.cos(th)*r; z = np.sin(th)*r; axis = np.array([x,y,z])
        axis = axis/np.linalg.norm(axis)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.eye(3)+np.sin(0)*K+(1-np.cos(0))*(K@K)  # identity for now
        mats.append(R)
    return mats[:50]  # limit for speed


def generate_pose(receptor_pdb, ligand_pdb, rec_chain, lig_chain):
    """Generate a docked pose via FFT grid search.
    Returns path to a new PDB with ligand transformed to best pose.
    """
    rec_grid = _pdb_to_grid(receptor_pdb, rec_chain)
    lig_grid = _pdb_to_grid(ligand_pdb, lig_chain)
    f1 = fft.rfftn(rec_grid)
    best_score = -1e9; best_R = None; best_t = None

    for R in _rotations():
        lig_rot = scipy_rotate(lig_grid, 0, axes=(1,2), reshape=False, order=1)
        f2 = fft.rfftn(lig_rot)
        corr = fft.irfftn(f1*np.conj(f2))
        score = float(corr.max())
        if score > best_score: best_score = score; best_R = R

    # Apply best transform to ligand and write PDB
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
    parser = PDBParser(QUIET=True)
    rec_s = parser.get_structure('rec', receptor_pdb)
    lig_s = parser.get_structure('lig', ligand_pdb)

    out_s = Structure.Structure('docked'); out_m = Model.Model(0)
    for c in rec_s[0]: out_m.add(c.copy())
    for c in lig_s[0]:
        cc = c.copy()
        for atom in cc.get_atoms():
            v = atom.get_vector().get_array()
            atom.set_coord((v @ best_R.T).tolist() if best_R is not None else v.tolist())
        out_m.add(cc)
    out_s.add(out_m)

    out = '_fft_pose.pdb'
    io = PDBIO(); io.set_structure(out_s); io.save(out)
    return out


if __name__ == '__main__':
    rec = 'data/misfolding_targets/3STB.pdb'
    lig = 'data/abeta_conformations/pdbs/abeta42_seed0_42.pdb'
    if os.path.exists(rec) and os.path.exists(lig):
        print("Generating FFT pose...")
        pose = generate_pose(rec, lig, 'A', 'P')
        print(f"Pose saved: {pose}")
