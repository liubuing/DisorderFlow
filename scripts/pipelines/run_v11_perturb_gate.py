#!/usr/bin/env python3
"""V11 PERTURBATION GATE: Test if HDOCK is sensitive to CDR backbone changes.

Instead of AF2 folding (GPU OOM / slow CPU), we apply controlled random
perturbations to CDR loop backbone atoms and check if HDOCK scores change.
This validates the fold-then-dock hypothesis: "different backbone -> different score".

If HDOCK CAN distinguish perturbed backbones, fold-then-dock WOULD work.
If HDOCK CANNOT (even with large perturbations), it's fundamentally limited.
"""
import sys, os, pickle, time, json, subprocess, shutil, random
import numpy as np
from Bio.PDB import PDBParser, PDBIO

SCAFFOLD = 'data/anti_abeta_refs/4HIX.pdb'
OUT_DIR = 'fold_dock_results'
os.makedirs(f'{OUT_DIR}/receptors', exist_ok=True)

print("=" * 60)
print("V11 Perturbation Gate: CDR Backbone -> HDOCK Sensitivity")
print("=" * 60)

# Parse scaffold and identify CDR atoms
from disorderflow.utils.protein import parsers, constants
from disorderflow.datasets.sabdab import _label_heavy_chain_cdr, _label_light_chain_cdr

parser = PDBParser(QUIET=True)
s = parser.get_structure('4HIX', SCAFFOLD)[0]
hd, hm = parsers.parse_biopython_structure(s['H'], max_resseq=113)
hd, hm = _label_heavy_chain_cdr(hd, hm)
ld, lm = parsers.parse_biopython_structure(s['L'], max_resseq=106)
ld, lm = _label_light_chain_cdr(ld, lm)

# Identify CDR atom coordinates
def get_cdr_atoms(structure, chain_id, data, cdr_types):
    """Get all N,CA,C,O atom coordinates for CDR residues."""
    if chain_id not in structure:
        return {}
    chain = structure[chain_id]
    residues = list(chain.get_residues())
    atoms = {}
    for ct in cdr_types:
        m = (data['cdr_flag'] == ct)
        if m.any():
            for idx in m.nonzero(as_tuple=True)[0]:
                i = idx.item()
                if i < len(residues):
                    res = residues[i]
                    for an in ['N', 'CA', 'C', 'O']:
                        if an in res:
                            atoms[(chain_id, i, an)] = res[an].get_coord().copy()
    return atoms

h_atoms = get_cdr_atoms(s, 'H', hd, [1, 2, 3])
l_atoms = get_cdr_atoms(s, 'L', ld, [4, 5, 6])
print(f"Native CDR atoms: H={len(h_atoms)}, L={len(l_atoms)}")

# Generate perturbed receptors
def perturb_and_save(scaffold_pdb, out_path, h_atoms, l_atoms, sigma):
    """Apply Gaussian noise to CDR backbone atoms and save."""
    ps = PDBParser(QUIET=True)
    st = ps.get_structure('ref', scaffold_pdb)[0]

    for atoms, chain_id in [(h_atoms, 'H'), (l_atoms, 'L')]:
        if chain_id not in st:
            continue
        chain = st[chain_id]
        residues = list(chain.get_residues())
        for (cid, i, an), coord in atoms.items():
            if cid == chain_id and i < len(residues):
                res = residues[i]
                if an in res:
                    noise = np.random.randn(3) * sigma
                    res[an].set_coord(coord + noise)

    io = PDBIO()
    io.set_structure(st)
    io.save(out_path)

sigmas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]  # Angstroms
random.seed(42)
np.random.seed(42)

print(f"\nGenerating {len(sigmas)} perturbed receptors (sigma={sigmas})...")
for sigma in sigmas:
    out = f'{OUT_DIR}/receptors/perturb_{sigma:.1f}A.pdb'
    perturb_and_save(SCAFFOLD, out, h_atoms, l_atoms, sigma)

    # Verify perturbation
    ps = PDBParser(QUIET=True)
    st = ps.get_structure('test', out)[0]
    distances = []
    for (cid, i, an), coord in h_atoms.items():
        if cid in st:
            residues = list(st[cid].get_residues())
            if i < len(residues) and an in residues[i]:
                d = np.linalg.norm(residues[i][an].get_coord() - coord)
                distances.append(d)
    for (cid, i, an), coord in l_atoms.items():
        if cid in st:
            residues = list(st[cid].get_residues())
            if i < len(residues) and an in residues[i]:
                d = np.linalg.norm(residues[i][an].get_coord() - coord)
                distances.append(d)
    rmsd = np.sqrt(np.mean(np.array(distances)**2))
    print(f"  sigma={sigma:.1f}A: actual CDR RMSD={rmsd:.3f}A ({len(distances)} atoms)")

# Run HDOCK
print(f"\nRunning HDOCK on {len(sigmas)} perturbed receptors...")
ligand = f'{OUT_DIR}/ligand.pdb'
if not os.path.exists(ligand):
    shutil.copy('HDOCKlite-v1.1/4HIX_ligand_native.pdb', ligand)

def hdock_score(receptor_rel, out_name):
    cmd = (f'cd HDOCKlite-v1.1 && LD_LIBRARY_PATH=. ./hdock '
           f'{receptor_rel} ../{ligand} -out {out_name} 2>&1')
    try:
        r = subprocess.run(['wsl', 'bash', '-c', cmd],
                          capture_output=True, text=True, timeout=180)
    except:
        return None
    op = f'HDOCKlite-v1.1/{out_name}'
    if not os.path.exists(op):
        return None
    scores = []
    with open(op) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 9:
                try: scores.append(float(p[6]))
                except: pass
    if not scores: return None
    return {'top1': min(scores), 'top5': sorted(scores)[:5],
            'mean': np.mean(scores), 'n': len(scores)}

results = {}
for sigma in sigmas:
    rec = f'../{OUT_DIR}/receptors/perturb_{sigma:.1f}A.pdb'
    out_name = f'v11_p{sigma:.1f}.out'
    print(f"  sigma={sigma:.1f}A...", end=' ', flush=True)
    r = hdock_score(rec, out_name)
    if r:
        results[f'sigma_{sigma:.1f}'] = r
        print(f"top1={r['top1']:.1f}")
    else:
        print("FAILED")

# Analysis
print(f"\n{'='*60}")
print("V11 Perturbation Gate: CDR Backbone RMSD vs HDOCK Score")
print(f"{'='*60}")
print(f"{'Sigma':>8} {'RMSD(A)':>10} {'HDOCK top1':>12} {'Delta':>10}")
print("-" * 45)
base_score = results.get('sigma_0.0', {}).get('top1', 0)
for sigma in sigmas:
    key = f'sigma_{sigma:.1f}'
    if key in results:
        s = results[key]['top1']
        delta = s - base_score
        print(f"{sigma:>8.1f} {'?':>10} {s:>12.1f} {delta:>+10.1f}")

# Gate verdict
if len(results) >= 3:
    scores = [results[k]['top1'] for k in sorted(results.keys())]
    score_range = max(scores) - min(scores)
    print(f"\nScore range across perturbations: {score_range:.1f}")
    if score_range > 5:
        print("Gate: PASS - HDOCK IS sensitive to CDR backbone changes")
        print("fold-then-dock WOULD work (if we had AF2 folding)")
    elif score_range > 1:
        print("Gate: MARGINAL - weak sensitivity")
    else:
        print("Gate: FAIL - HDOCK insensitive to CDR backbone changes")
        print("fold-then-dock would NOT help even with AF2")

ts = time.strftime('%Y%m%d_%H%M%S')
out_path = f'idp_benchmark_results/fold_dock_gate_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({'probe': 'V11 perturbation gate', 'sigmas': sigmas, 'results': results}, f, indent=2)
print(f"Saved: {out_path}")
print("Done.")
