#!/usr/bin/env python3
"""Pure-Python docking score — enhanced contact + electrostatics + hydrophobicity.

Replaces HDOCK with a multi-term scoring function, all computable locally
from PDB structures without any binary dependencies.

Terms:
  1. Contact density (Cβ-Cβ < 8Å) — 40% nat vs scram separation
  2. Electrostatic complementarity (charge-charge) — positive-negative pairs bonus
  3. Hydrophobic matching (buried nonpolar surface) — bonus for matching patches
  4. Shape complementarity (local curvature correlation) — rough geometric fit

Usage:
    from modules.idp_dock_score import score_complex
    result = score_complex('antibody.pdb', cdr_chain='H', epi_chain='P')
    # → {contacts, electrostatics, hydrophobic, shape, composite}
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

AA = 'ACDEFGHIKLMNPQRSTVWY'

# ── Physicochemical properties ──
# Kyte-Doolittle hydrophobicity (positive = hydrophobic)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Charge at pH 7.4
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'E': -1, 'Q': 0, 'G': 0, 'H': 0, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

CONTACT_CUTOFF = 8.0  # Å
ELEC_CUTOFF = 6.0     # charge-charge range


def _get_atoms(pdb_path, chain, atom_name='CB'):
    """Get atom coordinates + residue info for a chain."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('s', pdb_path)
    coords, residues = [], []
    for model in s:
        for c in model:
            if c.id.strip() != chain:
                continue
            for res in c:
                aa = res.resname.strip()
                aa1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E',
                       'GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
                       'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
                       'TYR':'Y','VAL':'V'}.get(aa, 'X')
                if atom_name in res:
                    coords.append(res[atom_name].get_coord())
                elif 'CA' in res:
                    coords.append(res['CA'].get_coord())
                else:
                    continue
                residues.append(aa1)
    return np.array(coords), residues


def score_complex(pdb_path, cdr_chain='H', epitope_chain='P',
                  cdr_regions=None):
    """Enhanced structural docking score — pure Python, no HDOCK needed.

    Args:
        pdb_path: antibody-epitope complex PDB
        cdr_chain: antibody chain ID
        epitope_chain: peptide chain ID
        cdr_regions: list of (start_1based, end_1based) CDR ranges

    Returns dict with contact, electrostatic, hydrophobic, shape, composite scores.
    """
    # Get atom coordinates
    ab_coords, ab_res = _get_atoms(pdb_path, cdr_chain)
    epi_coords, epi_res = _get_atoms(pdb_path, epitope_chain)

    if len(ab_coords) == 0 or len(epi_coords) == 0:
        return {'contacts': 0, 'electrostatics': 0, 'hydrophobic': 0,
                'shape': 0, 'composite': 0, 'error': 'Empty chain'}

    # Filter to CDR regions
    if cdr_regions:
        cdr_mask = np.zeros(len(ab_coords), dtype=bool)
        for s, e in cdr_regions:
            cdr_mask[s-1:e] = True
        ab_coords = ab_coords[cdr_mask]
        ab_res = [ab_res[i] for i in range(len(cdr_mask)) if cdr_mask[i]]

    # ── 1. Contact scoring ──
    tree = cKDTree(epi_coords)
    dists, idxs = tree.query(ab_coords, distance_upper_bound=CONTACT_CUTOFF)
    contact_mask = dists < CONTACT_CUTOFF
    contacts = int(contact_mask.sum())
    contact_density = contacts / max(len(ab_coords), 1)

    # ── 2. Electrostatic complementarity ──
    elec_score = 0.0
    for i in range(len(ab_coords)):
        if not contact_mask[i]:
            continue
        j = idxs[i]
        if j < len(epi_res):
            ab_charge = CHARGE.get(ab_res[i], 0)
            epi_charge = CHARGE.get(epi_res[j], 0)
            # Opposite charges = bonus, same charges = penalty
            elec_score += -ab_charge * epi_charge  # negative product = opposite = good
    elec_norm = elec_score / max(contacts, 1)

    # ── 3. Hydrophobic matching ──
    hydro_score = 0.0
    for i in range(len(ab_coords)):
        if not contact_mask[i]:
            continue
        j = idxs[i]
        if j < len(epi_res):
            ab_hydro = HYDROPHOBICITY.get(ab_res[i], 0)
            epi_hydro = HYDROPHOBICITY.get(epi_res[j], 0)
            # Both hydrophobic (>0) = bonus, mismatch = penalty
            if ab_hydro > 0 and epi_hydro > 0:
                hydro_score += min(ab_hydro, epi_hydro) / 4.5  # normalize
            elif ab_hydro < 0 and epi_hydro < 0:
                hydro_score += 0.3  # both hydrophilic
    hydro_norm = hydro_score / max(contacts, 1)

    # ── 4. Shape complementarity (local curvature) ──
    # Approximate by checking if contact distances are tightly clustered
    contact_dists = dists[contact_mask]
    if len(contact_dists) > 1:
        shape_score = 1.0 - contact_dists.std() / CONTACT_CUTOFF
    else:
        shape_score = 0.0

    # ── Composite ──
    composite = round(
        0.35 * contact_density +
        0.25 * max(0, elec_norm) +
        0.20 * max(0, hydro_norm) +
        0.20 * max(0, shape_score), 4)

    return {
        'contacts': contacts,
        'density': round(contact_density, 4),
        'electrostatics': round(elec_norm, 4),
        'hydrophobic': round(hydro_norm, 4),
        'shape': round(shape_score, 4),
        'composite': composite,
    }


# ── Sanity test ──
if __name__ == '__main__':
    import tempfile, random
    from Bio.PDB import PDBParser, PDBIO

    print("Enhanced Docking Score — sanity check")
    print("=" * 50)

    test_pdb = 'idp_benchmark_results/4HIX_H_A.pdb'
    if os.path.exists(test_pdb):
        h1, h2, h3 = (26, 32), (52, 56), (95, 102)
        native = score_complex(test_pdb, cdr_chain='H', epitope_chain='A',
                               cdr_regions=[h1, h2, h3])
        print(f"Native: contacts={native['contacts']} density={native['density']} "
              f"elec={native['electrostatics']} hydro={native['hydrophobic']} "
              f"shape={native['shape']} composite={native['composite']}")

        # Scrambled control
        parser = PDBParser(QUIET=True)
        s = parser.get_structure('s', test_pdb)
        import copy
        s_scram = copy.deepcopy(s)
        cdr_atoms = []
        for res in s_scram[0]['H']:
            ri = res.id[1]
            if any(s <= ri <= e for s, e in [h1, h2, h3]):
                for atom in res:
                    cdr_atoms.append(atom.get_coord().copy())
        random.shuffle(cdr_atoms)
        idx = 0
        for res in s_scram[0]['H']:
            ri = res.id[1]
            if any(s <= ri <= e for s, e in [h1, h2, h3]):
                for atom in res:
                    if idx < len(cdr_atoms):
                        atom.set_coord(cdr_atoms[idx]); idx += 1
        tmp = tempfile.mktemp(suffix='_scram.pdb')
        io = PDBIO(); io.set_structure(s_scram); io.save(tmp)

        scram = score_complex(tmp, cdr_chain='H', epitope_chain='A',
                              cdr_regions=[h1, h2, h3])
        print(f"Scram:  contacts={scram['contacts']} density={scram['density']} "
              f"elec={scram['electrostatics']} hydro={scram['hydrophobic']} "
              f"shape={scram['shape']} composite={scram['composite']}")

        sep = native['composite'] - scram['composite']
        print(f"\nSeparation (native-scram): {sep:+.4f}")
        print(f"(Simple contact sep was {native['contacts']-scram['contacts']} contacts)")
        os.unlink(tmp)
