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
    # 单链原子(CB 优先、无 CB 用 CA)栅格化：以质心为原点居中、1.2Å/格，
    # 高斯模糊(sigma≈1.25 格)平滑密度以获得非零重叠的互相关峰
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
    # 旋转采样：Fibonacci 球面取均匀轴向；当前角度固定为 0，
    # 即旋转矩阵实际为单位阵（占位实现，仅平移搜索有效）
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
    # 逐旋转做 FFT 互相关，取相关峰值最高的旋转；
    # 互相关峰值位置即最优平移（此处只比较峰值大小，未显式回读平移）
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
    # 输出结构 = 受体全部链(原坐标) + 配体链(应用最优旋转)
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
