#!/usr/bin/env python3
"""P0-2: IDP Triplet Validator — docking-free interface quality metrics.

M0 gate proved AF2 ipTM cannot distinguish native from scrambled CDR
(4HIX: 0.118 vs 0.116). This module replaces AF2 as primary metric for
IDP targets with structure-based contact analysis.

Triplet scores (all computed from antibody-peptide PDB):
  1. Interface contacts — Cβ pairs ≤8Å between CDR and epitope
  2. Contact density — contacts / CDR_length (normalized)
  3. BFN confidence — disorder-gated (disorder>0.5 → marked unreliable)

Usage:
    from modules.idp_validation_triplet import score_complex
    scores = score_complex('antibody_peptide.pdb', cdr_regions=[(26,32),...])
    # → {'contacts': 45, 'density': 1.8, 'bfn_iptm': 0.12, 'bfn_reliable': False}
"""
import os, sys
import numpy as np
from typing import List, Tuple, Dict, Optional

AA3_TO_1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E',
            'GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
            'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
            'TYR':'Y','VAL':'V'}

CONTACT_CUTOFF = 8.0  # Å, Cβ-Cβ distance


def _get_cb_positions(pdb_path: str) -> Dict[str, np.ndarray]:
    """Extract Cβ positions (fallback to CA for GLY) per chain from PDB.

    Returns {chain_id: (N, 3) array of Cβ/CA coordinates}.
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    coords = {}
    for model in structure:
        for chain in model:
            cid = chain.id.strip()
            if cid not in coords:
                coords[cid] = []
            for res in chain:
                if 'CB' in res:
                    coords[cid].append(res['CB'].get_coord())
                elif 'CA' in res:
                    coords[cid].append(res['CA'].get_coord())
    return {k: np.array(v) for k, v in coords.items()}


def interface_contacts(pdb_path: str, cdr_chain: str = 'H',
                       epitope_chain: str = 'P',
                       cdr_regions: Optional[List[Tuple[int, int]]] = None,
                       cutoff: float = CONTACT_CUTOFF) -> Dict:
    """Count interface contacts between CDR residues and epitope.

    Args:
        pdb_path: antibody-peptide complex PDB
        cdr_chain: antibody chain ID (default 'H')
        epitope_chain: peptide chain ID (default 'P')
        cdr_regions: list of (start_1based, end_1based) CDR residue ranges.
                     If None, uses all cdr_chain residues.
        cutoff: Cβ-Cβ distance cutoff in Å

    Returns:
        dict with contacts, density, cdr_residues, epitope_residues
    """
    coords = _get_cb_positions(pdb_path)

    ab_coords = coords.get(cdr_chain)
    epi_coords = coords.get(epitope_chain)

    if ab_coords is None or epi_coords is None:
        return {'contacts': 0, 'density': 0.0, 'error': 'Missing chain'}

    # Filter to CDR regions if specified
    if cdr_regions:
        # cdr_regions are 1-based residue indices
        cdr_mask = np.zeros(len(ab_coords), dtype=bool)
        for start, end in cdr_regions:
            cdr_mask[start-1:end] = True
        ab_coords = ab_coords[cdr_mask]

    if len(ab_coords) == 0 or len(epi_coords) == 0:
        return {'contacts': 0, 'density': 0.0, 'cdr_len': len(ab_coords),
                'epi_len': len(epi_coords)}

    # Pairwise distances
    dists = np.linalg.norm(
        ab_coords[:, None, :] - epi_coords[None, :, :], axis=-1)
    contacts = int((dists < cutoff).sum())
    density = contacts / max(len(ab_coords), 1)

    return {
        'contacts': contacts,
        'density': round(density, 4),
        'cdr_len': len(ab_coords),
        'epi_len': len(epi_coords),
        'mean_distance': round(float(dists.min(axis=1).mean()), 2),
    }


def score_complex(pdb_path: str,
                  cdr_chain: str = 'H',
                  epitope_chain: str = 'P',
                  cdr_regions: Optional[List[Tuple[int, int]]] = None,
                  disorder_mean: Optional[float] = None,
                  bfn_iptm: Optional[float] = None) -> Dict:
    """Compute IDP triplet score for antibody-peptide complex.

    M0 gate result: AF2 ipTM is noise-level for IDP targets.
    Primary metric = interface contacts (structure-based, no AF2 dependency).

    Returns:
        {contacts, density, mean_dist, bfn_iptm, bfn_reliable, triplet_summary}
    """
    contact_score = interface_contacts(pdb_path, cdr_chain, epitope_chain,
                                       cdr_regions)

    # BFN confidence with disorder gate (I-2)
    bfn_reliable = True
    if disorder_mean is not None and disorder_mean > 0.5:
        bfn_reliable = False

    triplet = {
        'contacts': contact_score['contacts'],
        'density': contact_score['density'],
        'mean_distance': contact_score.get('mean_distance', 99),
        'cdr_len': contact_score.get('cdr_len', 0),
        'epi_len': contact_score.get('epi_len', 0),
        'bfn_iptm': bfn_iptm,
        'bfn_reliable': bfn_reliable,
    }

    # Simple composite: contacts weighted by density, penalized by distance
    triplet['composite'] = round(
        triplet['density'] * max(0, 1.0 - triplet['mean_distance'] / 15.0), 4)

    return triplet


def score_designs(design_pdbs: List[str],
                  cdr_chain: str = 'H',
                  epitope_chain: str = 'P',
                  cdr_regions: Optional[List[Tuple[int, int]]] = None) -> List[Dict]:
    """Batch-score multiple design PDBs. Returns sorted by composite descending."""
    results = []
    for pdb in design_pdbs:
        if os.path.exists(pdb):
            results.append({'pdb': pdb, **score_complex(pdb, cdr_chain, epitope_chain, cdr_regions)})
    results.sort(key=lambda r: r['composite'], reverse=True)
    return results


# ── Quick sanity test ──
if __name__ == '__main__':
    # Test on 4HIX: extract antibody+peptide as separate chains
    print("IDP Triplet Validator — sanity check")
    print("=" * 50)

    # Use the 4HIX PDB to test contact scoring
    pdb_4hix = 'data/anti_abeta_refs/4HIX.pdb'
    if os.path.exists(pdb_4hix):
        # Antibody = chain H+L, peptide = find short chain
        coords = _get_cb_positions(pdb_4hix)
        chains = sorted(coords.keys(), key=lambda c: len(coords[c]))
        print(f"  Chains: {[(c, len(v)) for c, v in coords.items()]}")
        # Smallest chain is likely the peptide
        if len(chains) >= 3:
            ab_chain = chains[-1]  # longest = antibody
            pep_chain = chains[0]  # shortest = peptide
            print(f"  Antibody: chain {ab_chain} ({len(coords[ab_chain])}aa)")
            print(f"  Peptide:  chain {pep_chain} ({len(coords[pep_chain])}aa)")

            # CDR regions for 4HIX VH (Chothia)
            h1, h2, h3 = (26, 32), (52, 56), (95, 102)
            cs = interface_contacts(pdb_4hix, ab_chain, pep_chain,
                                    [h1, h2, h3])
            print(f"\n  Native CDR contacts: {cs['contacts']} (density={cs['density']:.3f})")
            print(f"  Mean CDR-peptide distance: {cs['mean_distance']}Å")
            print(f"  This is the baseline — scrambled CDR should score lower.")
    else:
        print(f"  {pdb_4hix} not found")
