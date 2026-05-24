#!/usr/bin/env python
"""AQP4 Advanced Antibody Design Pipeline.

Four-direction analysis:
  1. Rigid-body docking refinement (shape + electrostatic complementarity)
  2. Interface chemistry (hydrophobicity, charge, H-bond)
  3. Statistical potential binding free energy (DFIRE-like)
  4. Interface-guided CDR optimization

Usage:
  python aqp4_advanced_pipeline.py --pdb complex.pdb --designs designs.fasta
  python aqp4_advanced_pipeline.py --results_dir ./results --output_dir ./out
  python aqp4_advanced_pipeline.py --pdb complex.pdb --skip-docking
"""
import sys, os, io, json, math, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / 'results' / 'aqp4'

TARGET_SEQ = "RRFKEAFSKAAQQTKGSYMEVEDNRSQVETDDLILKPGVVHVIDVDRGEEKKGKDQSGEVLSSV"

HYDROPATHY = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
    'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
    'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3,
}
CHARGE = {
    'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,
    'M':0,'N':0,'P':0,'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0,
}
HBOND = {
    'A':0,'C':1,'D':-1,'E':-1,'F':0,'G':0,'H':2,'I':0,'K':1,'L':0,
    'M':0,'N':2,'P':0,'Q':2,'R':1,'S':2,'T':2,'V':0,'W':1,'Y':2,
}
RES_3TO1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}

DFIRE_RC = 15.0
DFIRE_BIN = 0.5
DFIRE_NBINS = int(DFIRE_RC / DFIRE_BIN)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='AQP4 Advanced Antibody Design Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python aqp4_advanced_pipeline.py --pdb complex.pdb
  python aqp4_advanced_pipeline.py --results_dir ./results --output_dir ./out
  python aqp4_advanced_pipeline.py --pdb complex.pdb --skip-docking --skip-energy
        ''')
    p.add_argument('--pdb', help='Path to input PDB complex (receptor=chain A, ligand=chain B)')
    p.add_argument('--designs', help='Path to FASTA file with nanobody designs')
    p.add_argument('--results_dir', default=str(DEFAULT_RESULTS_DIR),
                   help='Directory with AF2 multimer results JSON and designs FASTA')
    p.add_argument('--output_dir', help='Output directory (default: ./aqp4_advanced_results)')
    p.add_argument('--rec_chain', default='A', help='Receptor chain ID (default: A)')
    p.add_argument('--lig_chain', default='B', help='Ligand chain ID (default: B)')
    p.add_argument('--skip-docking', action='store_true', help='Skip docking refinement')
    p.add_argument('--skip-chemistry', action='store_true', help='Skip interface chemistry')
    p.add_argument('--skip-energy', action='store_true', help='Skip binding energy calculation')
    p.add_argument('--skip-optimization', action='store_true', help='Skip CDR optimization')
    p.add_argument('--n-rotations', type=int, default=16, help='Docking rotation samples')
    p.add_argument('--n-translations', type=int, default=8, help='Docking translation samples per rotation')
    p.add_argument('--contact-cutoff', type=float, default=5.0, help='Interface contact cutoff (A)')
    return p.parse_args()


# ============================================================
# PDB parsing
# ============================================================
def parse_pdb_atoms(pdb_path):
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            try:
                atoms.append({
                    'res': (line[17:20].strip(), int(line[22:26].strip()), line[21].strip()),
                    'atom': line[12:16].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'element': line[76:78].strip() or line[12:14].strip(),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def get_residue_centroids(atoms, chain=None):
    groups = defaultdict(list)
    for a in atoms:
        if chain and a['res'][2] != chain:
            continue
        groups[a['res']].append(a)
    centroids = {}
    for res, alist in groups.items():
        coords = np.array([[a['x'], a['y'], a['z']] for a in alist])
        centroids[res] = coords.mean(axis=0)
    return centroids


def get_surface_atoms(atoms, probe_radius=1.4):
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    if len(coords) == 0:
        return []
    tree = KDTree(coords)
    surface_mask = np.zeros(len(atoms), dtype=bool)
    for i, c in enumerate(coords):
        neighbors = tree.query_ball_point(c, 5.0)
        if len(neighbors) < 25:
            surface_mask[i] = True
    return [atoms[i] for i in range(len(atoms)) if surface_mask[i]]


def load_fasta_sequences(fasta_path):
    """Load sequences from a FASTA file."""
    seqs = []
    if not os.path.exists(fasta_path):
        return seqs
    with open(fasta_path) as f:
        current_seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    seqs.append(current_seq)
                current_seq = ''
            elif line:
                current_seq += line
        if current_seq:
            seqs.append(current_seq)
    return seqs


# ============================================================
# Direction 1: Docking refinement
# ============================================================
def generate_rotation_samples(n_angles=24):
    rotations = []
    phi = math.pi * (3 - math.sqrt(5))
    for i in range(n_angles):
        y = 1 - (i / float(n_angles - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        axis = np.array([math.cos(theta) * radius, y, math.sin(theta) * radius])
        for angle in np.linspace(0, 2*math.pi, 6, endpoint=False):
            r = Rotation.from_rotvec(axis * angle)
            rotations.append(r)
    return rotations


def shape_complementarity_score(coords_A, coords_B, grid_spacing=1.0, padding=10):
    all_coords = np.vstack([coords_A, coords_B])
    min_c = all_coords.min(axis=0) - padding
    max_c = all_coords.max(axis=0) + padding
    dims = np.ceil((max_c - min_c) / grid_spacing).astype(int) + 1

    def points_to_grid(coords):
        idx = np.floor((coords - min_c) / grid_spacing).astype(int)
        grid = np.zeros(dims)
        valid = (idx >= 0).all(axis=1) & (idx < dims).all(axis=1)
        for p in idx[valid]:
            grid[tuple(p)] = 1.0
        return gaussian_filter(grid, sigma=1.5)

    grid_A = points_to_grid(coords_A)
    grid_B = points_to_grid(coords_B)
    fA = np.fft.rfftn(grid_A)
    fB = np.fft.rfftn(grid_B)
    score_map = np.fft.irfftn(fA * fB.conj())
    overlap_penalty = np.fft.irfftn(np.abs(fA) * np.abs(fB))
    composite = score_map - 0.3 * overlap_penalty
    return composite.max(), composite


def docking_score(atoms_rec, atoms_lig, rotation, translation, grid_spacing=1.2):
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_lig])
    lig_centroid = lig_coords.mean(axis=0)
    rotated = rotation.apply(lig_coords - lig_centroid) + lig_centroid + translation

    rec_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_rec])
    shape_score, _ = shape_complementarity_score(rec_coords, rotated, grid_spacing)

    rec_tree = KDTree(rec_coords)
    clashes, contacts = 0, 0
    for rc in rotated:
        dist, _ = rec_tree.query(rc)
        if dist < 2.0:
            clashes += 1
        elif dist < 5.0:
            contacts += 1

    return shape_score + contacts * 0.5 - clashes * 10.0, rotated


def local_docking(pdb_path, rec_chain='A', lig_chain='B',
                  epitope_centroid=None, n_rotations=24, translations_per_rot=10):
    print("\n" + "=" * 60)
    print("  Direction 1: Rigid-Body Docking Refinement")
    print("=" * 60)

    all_atoms = parse_pdb_atoms(pdb_path)
    atoms_rec = [a for a in all_atoms if a['res'][2] == rec_chain]
    atoms_lig = [a for a in all_atoms if a['res'][2] == lig_chain]

    surf_rec = get_surface_atoms(atoms_rec)
    surf_lig = get_surface_atoms(atoms_lig)

    if not surf_rec or not surf_lig:
        print("  WARNING: empty surface atoms, using all atoms")
        surf_rec = atoms_rec
        surf_lig = atoms_lig

    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_lig])
    lig_center = lig_coords.mean(axis=0)

    if epitope_centroid is None:
        rec_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_rec])
        epitope_centroid = rec_coords.mean(axis=0)

    print(f"  Rec atoms: {len(atoms_rec)}, Lig atoms: {len(atoms_lig)}")
    print(f"  Surface: rec={len(surf_rec)}, lig={len(surf_lig)}")

    rotations = generate_rotation_samples(n_rotations)
    distances = np.linspace(2, 15, translations_per_rot // 2)
    directions = [np.array(d) for d in [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]]

    results = []
    total = len(rotations) * len(distances) * len(directions)
    count = 0
    print(f"  Evaluating {total} poses...")

    for rot in rotations:
        for d in distances:
            for direc in directions:
                trans = epitope_centroid + direc * d - lig_center
                score, _ = docking_score(surf_rec, surf_lig, rot, trans)
                if score > -1e6:
                    results.append({'rotation': rot, 'translation': trans, 'score': float(score)})
                count += 1
                if count % 1000 == 0:
                    print(f"    {count}/{total} poses...")

    results.sort(key=lambda x: x['score'], reverse=True)
    top_scores = [f"{r['score']:.0f}" for r in results[:10]]
    print(f"  Evaluated: {total} poses")
    print(f"  Top 10 scores: {top_scores}")
    return results, atoms_rec, atoms_lig


# ============================================================
# Direction 2: Interface chemistry analysis
# ============================================================
def analyze_interface_chemistry(atoms_rec, atoms_lig, contact_cutoff=5.0):
    print("\n" + "=" * 60)
    print("  Direction 2: Interface Chemistry Analysis")
    print("=" * 60)

    rec_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_rec])
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_lig])
    rec_tree = KDTree(rec_coords)
    lig_tree = KDTree(lig_coords)

    interface_rec = set()
    interface_lig = set()
    for i, a in enumerate(atoms_rec):
        dist, _ = lig_tree.query(rec_coords[i])
        if dist < contact_cutoff:
            interface_rec.add(a['res'])
    for i, a in enumerate(atoms_lig):
        dist, _ = rec_tree.query(lig_coords[i])
        if dist < contact_cutoff:
            interface_lig.add(a['res'])

    # Filter out non-amino-acid residues (solvent, ligands, etc.)
    interface_rec = {r for r in interface_rec if r[0] in RES_3TO1}
    interface_lig = {r for r in interface_lig if r[0] in RES_3TO1}

    def get_interface_seq(residues):
        seen = set()
        seq = ''
        for res in sorted(residues, key=lambda r: r[1]):
            if res not in seen:
                seen.add(res)
                seq += RES_3TO1[res[0]]
        return seq

    rec_iface_seq = get_interface_seq(interface_rec)
    lig_iface_seq = get_interface_seq(interface_lig)

    n_rec = len(set(r[1] for r in interface_rec))
    n_lig = len(set(r[1] for r in interface_lig))
    print(f"  Interface residues: receptor={n_rec}, ligand={n_lig}")

    def score_interface(seq, label):
        hydro = [HYDROPATHY.get(aa, 0) for aa in seq]
        charge = [CHARGE.get(aa, 0) for aa in seq]
        hbond_n = sum(1 for aa in seq if HBOND.get(aa, 0) != 0)
        print(f"\n  {label} ({len(seq)} res): {seq}")
        print(f"    Hydrophobicity: mean={np.mean(hydro):.2f} [{min(hydro):.1f},{max(hydro):.1f}]")
        print(f"    Net charge: {sum(charge):.1f}")
        print(f"    H-bond capable: {hbond_n}")
        return {'seq': seq, 'hydro_mean': float(np.mean(hydro)),
                'net_charge': float(sum(charge)), 'n_hbond': hbond_n}

    rec_props = score_interface(rec_iface_seq, 'Receptor (AQP4)')
    lig_props = score_interface(lig_iface_seq, 'Ligand (Nanobody)')

    hydro_comp = -abs(rec_props['hydro_mean'] + lig_props['hydro_mean'])
    charge_comp = -abs(rec_props['net_charge'] + lig_props['net_charge'])
    hbond_density = (rec_props['n_hbond'] + lig_props['n_hbond']) / max(n_rec + n_lig, 1)

    print(f"\n  Complementarity:")
    print(f"    Hydrophobic match: {hydro_comp:.2f}")
    print(f"    Charge complement: {charge_comp:.2f}")
    print(f"    H-bond density: {hbond_density:.3f}")

    return {
        'receptor': rec_props, 'ligand': lig_props,
        'hydro_complement': hydro_comp,
        'charge_complement': charge_comp,
        'hbond_density': hbond_density,
        'interface_rec_n': n_rec, 'interface_lig_n': n_lig,
    }


# ============================================================
# Direction 3: Binding free energy (DFIRE statistical potential)
# ============================================================
def build_dfire_potential():
    aa_list = sorted(HYDROPATHY.keys())
    n_aa = len(aa_list)
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

    potential = np.zeros((n_aa, n_aa, DFIRE_NBINS))
    for i, aa1 in enumerate(aa_list):
        for j, aa2 in enumerate(aa_list):
            h1, h2 = HYDROPATHY[aa1], HYDROPATHY[aa2]
            c1, c2 = CHARGE[aa1], CHARGE[aa2]
            for k in range(DFIRE_NBINS):
                r = (k + 0.5) * DFIRE_BIN
                hp = -0.3 * max(h1, 0) * max(h2, 0) * math.exp(-((r - 5.0) ** 2) / 4.0)
                elec = -2.0 * c1 * c2 * math.exp(-r / 6.0) / max(r, 2.0)
                sigma = 3.5
                lj = 4.0 * ((sigma / max(r, 1.5)) ** 12 - (sigma / max(r, 1.5)) ** 6)
                hb = -2.0 if (HBOND.get(aa1,0) and HBOND.get(aa2,0) and 2.5 < r < 3.5) else 0.0
                potential[i, j, k] = hp + elec + lj + hb

    for k in range(DFIRE_NBINS):
        potential[:, :, k] -= potential[:, :, k].mean()

    return potential, aa_to_idx


def compute_binding_energy(atoms_rec, atoms_lig):
    print("\n" + "=" * 60)
    print("  Direction 3: Binding Free Energy (DFIRE Statistical Potential)")
    print("=" * 60)

    potential, aa_to_idx = build_dfire_potential()

    rec_centroids = get_residue_centroids(atoms_rec)
    lig_centroids = get_residue_centroids(atoms_lig)

    rec_reslist = sorted(rec_centroids.items(), key=lambda x: x[0][1])
    lig_reslist = sorted(lig_centroids.items(), key=lambda x: x[0][1])

    rec_pts = np.array([c for _, c in rec_reslist])
    lig_pts = np.array([c for _, c in lig_reslist])
    rec_aas = [RES_3TO1.get(r[0], 'X') for (r, _) in rec_reslist]
    lig_aas = [RES_3TO1.get(r[0], 'X') for (r, _) in lig_reslist]

    rec_tree = KDTree(rec_pts)

    total_energy = 0.0
    n_pairs = 0
    components = defaultdict(float)

    for i, (lc, la) in enumerate(zip(lig_pts, lig_aas)):
        if la not in aa_to_idx:
            continue
        dists, idxs = rec_tree.query(lc, k=min(10, len(rec_pts)))
        if not hasattr(dists, '__iter__'):
            dists, idxs = [dists], [idxs]

        for d, j in zip(dists, idxs):
            if d < 2.0 or d > DFIRE_RC:
                continue
            ra = rec_aas[j]
            if ra not in aa_to_idx:
                continue
            bin_idx = int(d / DFIRE_BIN)
            if bin_idx >= DFIRE_NBINS:
                continue
            e = potential[aa_to_idx[la], aa_to_idx[ra], bin_idx]
            total_energy += e
            n_pairs += 1

            h1, h2 = HYDROPATHY.get(la, 0), HYDROPATHY.get(ra, 0)
            c1, c2 = CHARGE.get(la, 0), CHARGE.get(ra, 0)
            if h1 > 1.0 and h2 > 1.0:
                components['hydrophobic'] += e
            if c1 * c2 < 0:
                components['salt_bridge'] += e
            if HBOND.get(la, 0) and HBOND.get(ra, 0):
                components['hbond'] += e

    avg_energy = total_energy / max(n_pairs, 1)
    dG_estimate = total_energy * 0.15

    print(f"  Residue pairs evaluated: {n_pairs}")
    print(f"  Total binding score: {total_energy:.1f}")
    print(f"  Per-pair average: {avg_energy:.3f}")
    for comp, e in sorted(components.items(), key=lambda x: x[1]):
        print(f"    {comp}: {e:.1f}")
    qual = "favorable" if dG_estimate < -5 else ("weak" if dG_estimate < 0 else "unfavorable")
    print(f"  Estimated dG_bind: {dG_estimate:.1f} kcal/mol ({qual})")

    return {
        'total_score': float(total_energy),
        'per_pair_avg': float(avg_energy),
        'dG_estimate': float(dG_estimate),
        'n_pairs': n_pairs,
        'quality': qual,
        'components': dict(components),
    }


# ============================================================
# Direction 4: CDR optimization via interface constraints
# ============================================================
def optimize_cdr_for_interface(interface, nanobody_seq):
    rec_props = interface['receptor']
    target_hydro = rec_props['hydro_mean']
    target_charge = rec_props['net_charge']

    optimized = []
    for i, aa in enumerate(nanobody_seq):
        if aa not in HYDROPATHY:
            optimized.append(aa)
            continue
        scores = {}
        for mut_aa in HYDROPATHY:
            hydro_score = -abs(HYDROPATHY[mut_aa] - target_hydro) / 4.5
            charge_score = CHARGE[mut_aa] * (-target_charge) * 0.5
            conservation = 2.0 if mut_aa == aa else 0.0
            diversity = 0.3 if abs(CHARGE[mut_aa]) > 0 or HBOND[mut_aa] != 0 else 0.0
            scores[mut_aa] = hydro_score + charge_score + conservation + diversity
        optimized.append(max(scores, key=scores.get))

    opt_seq = ''.join(optimized)
    mutations = [(i, nanobody_seq[i], opt_seq[i]) for i in range(len(nanobody_seq))
                 if nanobody_seq[i] != opt_seq[i]]
    return opt_seq, mutations


# ============================================================
# Main pipeline
# ============================================================
def run_pipeline(pdb_path, designs=None, output_dir=None,
                 rec_chain='A', lig_chain='B',
                 skip_docking=False, skip_chemistry=False,
                 skip_energy=False, skip_optimization=False,
                 n_rotations=16, n_translations=8, contact_cutoff=5.0):
    """Run the full AQP4 advanced design pipeline."""
    output_dir = Path(output_dir) if output_dir else PROJECT_DIR / 'alphafold_results' / 'aqp4_advanced'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  AQP4 Advanced Design Pipeline")
    print(f"  PDB: {pdb_path}")
    print(f"  Output: {output_dir}")
    steps = []
    if not skip_docking: steps.append("1: Docking")
    if not skip_chemistry: steps.append("2: Interface Chemistry")
    if not skip_energy: steps.append("3: Binding Energy")
    if not skip_optimization: steps.append("4: CDR Optimization")
    print(f"  Steps: {', '.join(steps) if steps else '(none selected)'}")
    print("=" * 70)

    if not os.path.exists(pdb_path):
        print(f"[ERROR] PDB not found: {pdb_path}")
        return None

    # Parse complex
    all_atoms = parse_pdb_atoms(str(pdb_path))
    atoms_rec = [a for a in all_atoms if a['res'][2] == rec_chain]
    atoms_lig = [a for a in all_atoms if a['res'][2] == lig_chain]
    print(f"\n  Chain {rec_chain} (receptor): {len(atoms_rec)} atoms, {len(set(a['res'] for a in atoms_rec))} residues")
    print(f"  Chain {lig_chain} (ligand):   {len(atoms_lig)} atoms, {len(set(a['res'] for a in atoms_lig))} residues")

    if not atoms_rec or not atoms_lig:
        print("[ERROR] Missing receptor or ligand chains in PDB.")
        return None

    output = {'pdb_path': str(pdb_path)}

    # Direction 1: Docking refinement
    if not skip_docking:
        rec_centroids = get_residue_centroids(atoms_rec)
        epitope_centroid = np.array(list(rec_centroids.values())).mean(axis=0)
        docking_results, _, _ = local_docking(
            str(pdb_path), rec_chain=rec_chain, lig_chain=lig_chain,
            epitope_centroid=epitope_centroid,
            n_rotations=n_rotations, translations_per_rot=n_translations)
        output['docking'] = {
            'top_score': float(docking_results[0]['score']) if docking_results else None,
            'n_poses': len(docking_results),
            'top_10_scores': [float(r['score']) for r in docking_results[:10]],
        }

    # Direction 2: Interface chemistry
    if not skip_chemistry:
        interface = analyze_interface_chemistry(atoms_rec, atoms_lig, contact_cutoff)
        output['interface_chemistry'] = {
            'receptor': interface['receptor'],
            'ligand': interface['ligand'],
            'complementarity': {
                'hydrophobic_match': interface['hydro_complement'],
                'charge_complement': interface['charge_complement'],
                'hbond_density': interface['hbond_density'],
            },
            'interface_residues': {
                'receptor_n': interface['interface_rec_n'],
                'ligand_n': interface['interface_lig_n'],
            },
        }
    else:
        interface = None

    # Direction 3: Binding energy
    if not skip_energy:
        energy = compute_binding_energy(atoms_rec, atoms_lig)
        output['binding_energy'] = energy

    # Direction 4: CDR optimization
    if not skip_optimization and interface is not None and designs:
        print("\n" + "=" * 60)
        print("  Direction 4: Interface-Guided CDR Optimization")
        print("=" * 60)
        print(f"\n  Target interface: hydro={interface['receptor']['hydro_mean']:.2f}, charge={interface['receptor']['net_charge']:.1f}")

        optimized_designs = []
        for i, seq in enumerate(designs):
            opt_seq, mutations = optimize_cdr_for_interface(interface, seq)
            optimized_designs.append({
                'idx': i, 'original': seq, 'optimized': opt_seq,
                'n_mutations': len(mutations), 'mutations': mutations,
            })
            print(f"  [{i}] {len(mutations)} mutations: {seq[:20]}... -> {opt_seq[:20]}...")
        output['optimized_designs'] = optimized_designs

        # Save optimized FASTA
        fasta_out = output_dir / 'optimized_designs.fasta'
        with open(fasta_out, 'w') as f:
            for d in optimized_designs:
                f.write(f">design_{d['idx']}_mut{d['n_mutations']}\n{d['optimized']}\n")
        print(f"\n  Saved optimized designs: {fasta_out}")

    # Save results
    results_path = output_dir / 'advanced_pipeline_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  [OK] Results saved: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    if output.get('docking'):
        print(f"  Best docking score: {output['docking']['top_score']:.1f}")
    if output.get('interface_chemistry'):
        ic = output['interface_chemistry']
        print(f"  H-bond density: {ic['complementarity']['hbond_density']:.3f}")
        print(f"  Charge complement: {ic['complementarity']['charge_complement']:.2f}")
    if output.get('binding_energy'):
        be = output['binding_energy']
        print(f"  dG_bind estimate: {be['dG_estimate']:.1f} kcal/mol ({be['quality']})")
    if output.get('optimized_designs'):
        print(f"  Optimized: {len(output['optimized_designs'])} designs")

    return output


def main():
    args = parse_args()

    # Resolve PDB path
    pdb_path = None
    designs = None

    if args.pdb:
        pdb_path = args.pdb
    else:
        # Search in results_dir
        results_dir = Path(args.results_dir)
        results_json = results_dir / 'AF2_multimer_v2_results.json'
        if results_json.exists():
            with open(results_json) as f:
                prev_results = json.load(f)
            success = [r for r in prev_results.get('results', []) if r.get('success') and r.get('pdb_path')]
            if success:
                success.sort(key=lambda r: r.get('iptm') or -1, reverse=True)
                best = success[0]
                pdb_path = best['pdb_path']
                print(f"  Found best PDB: {pdb_path}")
                print(f"  ipTM={best.get('iptm'):.3f}  pTM={best.get('ptm'):.3f}")
                if 'plddt' in best:
                    print(f"  pLDDT={best['plddt']:.1f}")
            else:
                print("[ERROR] No successful AF2 multimer predictions in results.")
                return
        else:
            print(f"[ERROR] No --pdb specified and no results JSON found at {results_json}")
            return

    if not pdb_path or not os.path.exists(pdb_path):
        print(f"[ERROR] PDB not found: {pdb_path}")
        return

    # Load designs
    if args.designs:
        designs = load_fasta_sequences(args.designs)
    else:
        results_dir = Path(args.results_dir)
        for fasta_name in ['3GD8_constrained_nanobody_designs.fasta', 'nanobody_designs.fasta', 'designs.fasta']:
            fasta_path = results_dir / fasta_name
            if fasta_path.exists():
                designs = load_fasta_sequences(str(fasta_path))
                if designs:
                    print(f"  Loaded {len(designs)} designs from {fasta_path}")
                    break
        if not designs:
            print("  No designs FASTA found, skipping CDR optimization.")
            args.skip_optimization = True

    run_pipeline(
        pdb_path=pdb_path,
        designs=designs,
        output_dir=args.output_dir,
        rec_chain=args.rec_chain,
        lig_chain=args.lig_chain,
        skip_docking=args.skip_docking,
        skip_chemistry=args.skip_chemistry,
        skip_energy=args.skip_energy,
        skip_optimization=args.skip_optimization,
        n_rotations=args.n_rotations,
        n_translations=args.n_translations,
        contact_cutoff=args.contact_cutoff,
    )


if __name__ == '__main__':
    main()
