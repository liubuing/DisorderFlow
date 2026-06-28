#!/usr/bin/env python3
"""P2: Multi-conformation robust scoring for IDP targets.

IDP_DESIGN_SCHEME_V2 §P2-2:
  Score each design against K conformations, use min/percentile (not mean).
  "A design that only scores well in one conformation is not robust."

Uses existing 5-seed Aβ42 conformations from data/abeta_conformations/pdbs/.
Contact-based scoring (no AF2/HDOCK dependency) — works immediately.

Usage:
    from modules.idp_multiconf_scorer import score_robust
    result = score_robust('design_antibody.pdb', epitope_spec)
    # → {min: 3, median: 5, max: 7, robust: 3, n_failed: 0}
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.idp_validation_triplet import interface_contacts, _get_cb_positions

# Paths to 5-seed conformations
ABETA_CONF_DIR = 'data/abeta_conformations/pdbs'
ABETA_PKL = 'data/abeta_conformations/abeta42_5seed.pkl'
ABETA_SEQUENCE = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'

# Epitope segments from P1 library
DEFAULT_EPITOPES = {
    'mid_16_24':  (16, 24),   # KLVFFAED — aggregation core
    'core_16_21': (16, 21),   # KLVFFA — minimal core
    'n_term_1_10': (1, 10),   # DAEFRHDSGY — plaque clearance
    'c_term_33_42': (33, 42), # GLMVGGVVIA — fibril cap
}


def _load_conformations():
    """Load 5-seed CA positions and RMSF from pickle."""
    with open(ABETA_PKL, 'rb') as f:
        data = pickle.load(f)
    return data['ca_positions'], data['rmsf'], data['n_conformations']


def score_robust(antibody_pdb: str,
                 ab_chain: str = 'H',
                 epitope_spec: str = 'mid_16_24',
                 cdr_regions: list = None,
                 contact_cutoff: float = 8.0) -> dict:
    """Score an antibody design against all 5 Aβ42 conformations.

    Strategy: superpose each Aβ conformation CA positions to a reference
    (seed 0), keep antibody fixed, compute contacts. The antibody should
    be pre-positioned near the antigen (e.g. from a docked or crystal pose).

    Args:
        antibody_pdb: PDB with antibody (scaffold + grafted CDR)
        ab_chain: antibody chain ID
        epitope_spec: key into DEFAULT_EPITOPES or 'full'
        cdr_regions: CDR residue ranges [(start, end), ...]
        contact_cutoff: Cb-Cb distance cutoff in A

    Returns:
        {min, max, median, p25, robust, n_below_threshold, per_conf}
        robust = p25 (25th percentile) — at least 75% of conformations
                 must have this many contacts
    """
    # Load conformations
    ca_positions, rmsf, n_conf = _load_conformations()

    # Get epitope residue range
    if epitope_spec == 'full':
        epi_range = (1, 42)
    else:
        epi_range = DEFAULT_EPITOPES.get(epitope_spec, (16, 24))
    epi_start, epi_end = epi_range

    # Load antibody CA positions once
    ab_coords_dict = _get_cb_positions(antibody_pdb)
    if ab_chain not in ab_coords_dict:
        return {'error': f'Chain {ab_chain} not found in {antibody_pdb}'}
    ab_coords = ab_coords_dict[ab_chain]

    # Filter to CDR regions if specified
    if cdr_regions:
        cdr_mask = np.zeros(len(ab_coords), dtype=bool)
        for s, e in cdr_regions:
            cdr_mask[s-1:e] = True
        ab_coords = ab_coords[cdr_mask]

    # Superpose reference: use seed 0 as anchor
    # Align each conformation's epitope CA to seed 0's epitope CA
    ref_ca = ca_positions[0]  # (42, 3)
    ref_epi = ref_ca[epi_start-1:epi_end]

    per_conf = []
    for seed_idx in range(n_conf):
        conf_ca = ca_positions[seed_idx]  # (42, 3)
        conf_epi = conf_ca[epi_start-1:epi_end]

        # Optimal superposition: align conf_epi to ref_epi
        # Kabsch algorithm (simple version)
        ref_center = ref_epi.mean(axis=0)
        conf_center = conf_epi.mean(axis=0)
        ref_centered = ref_epi - ref_center
        conf_centered = conf_epi - conf_center

        # Covariance matrix
        H = conf_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T  # rotation matrix
        t = ref_center - conf_center @ R

        # Apply rotation to all antigen atoms
        conf_aligned = conf_ca @ R + t

        # Compute contacts between ab_coords and aligned epitope
        dists = np.linalg.norm(
            ab_coords[:, None, :] - conf_aligned[epi_start-1:epi_end][None, :, :],
            axis=-1)
        contacts = int((dists < contact_cutoff).sum())
        per_conf.append(contacts)

    per_conf = np.array(per_conf)
    robust = float(np.percentile(per_conf, 25))  # 25th percentile = at least 75% of confs
    n_below = int((per_conf < 2).sum())  # confs with <2 contacts

    return {
        'min': int(per_conf.min()),
        'max': int(per_conf.max()),
        'median': float(np.median(per_conf)),
        'p25': robust,
        'robust': round(robust, 1),
        'n_below_threshold': n_below,
        'n_conformations': n_conf,
        'per_conf': per_conf.tolist(),
        'rmsf_mean': float(rmsf[epi_start-1:epi_end].mean()),
        'rmsf_max': float(rmsf[epi_start-1:epi_end].max()),
    }


def score_designs(design_pdbs: list, **kwargs) -> list:
    """Batch-score multiple design PDBs. Returns sorted by robust descending."""
    results = []
    for pdb in design_pdbs:
        if os.path.exists(pdb):
            r = score_robust(pdb, **kwargs)
            r['pdb'] = pdb
            results.append(r)
    results.sort(key=lambda r: r['robust'], reverse=True)
    return results


# ── Quick test ──
if __name__ == '__main__':
    print("P2 Multi-conformation Robust Scorer — sanity check")
    print("=" * 60)

    ca_positions, rmsf, n_conf = _load_conformations()
    print(f"Loaded {n_conf} conformations, RMSF range: {rmsf.min():.2f}-{rmsf.max():.2f}")

    # Test on 4HIX: use the antibody+peptide complex built earlier
    test_pdb = 'idp_benchmark_results/4HIX_H_A.pdb'
    if os.path.exists(test_pdb):
        # CDR regions for 4HIX VH (Chothia)
        h1, h2, h3 = (26, 32), (52, 56), (95, 102)
        result = score_robust(test_pdb, ab_chain='H', epitope_spec='mid_16_24',
                              cdr_regions=[h1, h2, h3])
        print(f"\n4HIX solanezumab (native CDR) vs 5 Aβ42 conformations:")
        print(f"  Epitope: mid_16_24 (KLVFFAED)")
        print(f"  Per-conformation contacts: {result['per_conf']}")
        print(f"  Robust (p25): {result['robust']}")
        print(f"  Range: {result['min']}-{result['max']} (median={result['median']})")
        print(f"  Confs with <2 contacts: {result['n_below_threshold']}/{result['n_conformations']}")
        print(f"  RMSF of epitope: {result['rmsf_mean']:.2f} (max={result['rmsf_max']:.2f})")

        # Now test a scrambled CDR
        import tempfile, random
        from Bio.PDB import PDBParser, PDBIO
        parser = PDBParser(QUIET=True)
        s = parser.get_structure('s', test_pdb)

        # Scramble CDR atom positions
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

        scram_result = score_robust(tmp, ab_chain='H', epitope_spec='mid_16_24',
                                    cdr_regions=[h1, h2, h3])
        print(f"\nScrambled CDR:")
        print(f"  Per-conformation contacts: {scram_result['per_conf']}")
        print(f"  Robust (p25): {scram_result['robust']}")
        print(f"  Range: {scram_result['min']}-{scram_result['max']}")

        sep = result['robust'] - scram_result['robust']
        print(f"\nNative vs Scrambled robust separation: {sep:.1f} contacts")
        print(f"(Single-conformation separation was 2 contacts)")
        os.unlink(tmp)
