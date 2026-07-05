#!/usr/bin/env python3
"""S0 Observational Pre-check: Collect anti-IDP antibody complexes and extract flexibility metrics.

Downloads PDB structures for known anti-Aβ/tau/αSyn antibodies and folded-antigen controls.
Extracts CDR-H3 B-factors and epitope sequences for disorder prediction.
This is the Go/NoGo gate for the entire Way4 project.

Output: way4_s0_observational.json — per-antibody flexibility metrics for statistical testing.

Usage:
    python way4_s0/collect_anti_idp.py [--control-n N] [--out way4_s0_observational.json]
"""

import sys, os, json, time, argparse, re, pickle, gzip
import urllib.request
import numpy as np
from collections import defaultdict
from io import StringIO, BytesIO

# --- Anti-IDP Antibody List (known anti-Aβ42 antibodies) ---
ANTI_IDP_PDBS = [
    # Anti-Aβ42 antibodies with known PDB structures
    ('4HIX', 'Aducanumab (anti-Aβ, N-term epitope)', 'anti_abeta'),
    ('5CSZ', 'Crenezumab (anti-Aβ, mid-domain)', 'anti_abeta'),
    ('3UOT', 'Anti-Aβ42 scFv (oligomer-specific)', 'anti_abeta'),
    ('6CGZ', 'Anti-Aβ42 Fab (protofibril)', 'anti_abeta'),
    ('6D9B', 'Anti-Aβ42 antibody (fibril)', 'anti_abeta'),
    ('6H3R', 'Anti-Aβ42 Fab (pyroglutamate-Aβ)', 'anti_abeta'),
    ('7K4V', 'Anti-Aβ42 antibody (N3pE)', 'anti_abeta'),
    # Anti-tau antibodies
    ('5V7V', 'Anti-tau Fab (PHF)', 'anti_tau'),
    ('6CBV', 'Anti-tau Fab (microtubule-binding region)', 'anti_tau'),
    ('7PFE', 'Anti-tau nanobody', 'anti_tau'),
    # Anti-αSynuclein antibodies
    ('6H1F', 'Anti-αSynuclein Fab', 'anti_asyn'),
    ('6HUV', 'Anti-αSynuclein Fab (C-term)', 'anti_asyn'),
    ('4BKL', 'Anti-αSynuclein scFv', 'anti_asyn'),
]

# Known chain mappings for anti-IDP complexes (H=heavy, L=light, A=antigen)
# Format: {pdb_id: {'H': chain_id, 'L': chain_id, 'Ag': chain_id}}
# If L is empty, it's a nanobody/heavy-chain-only antibody
CHAIN_MAP = {
    '4HIX':  {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '5CSZ':  {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '3UOT':  {'H': 'A', 'L': 'B', 'Ag': 'D'},  # scFv(A,B) + Aβ peptide(D)
    '6CGZ':  {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '6D9B':  None,   # cryo-EM single chain, not an antibody-antigen complex
    '6H3R':  {'H': 'A', 'L': 'B', 'Ag': 'C'},  # VH(A), VL(B)
    '7K4V':  None,   # single chain, not clearly antibody
    '5V7V':  None,   # large EM complex, not useful
    '6CBV':  {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '7PFE':  None,   # designed protein, no conserved Cys
    '6H1F':  {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '6HUV':  None,   # proteasome-like multi-chain, not antibody
    '4BKL':  {'H': 'A', 'L': 'B', 'Ag': 'C'},
}

# --- Folded-antigen antibody controls (well-characterized, epitope disorder ≈ 0) ---
FOLDED_CONTROLS = [
    # Well-characterized antibody-antigen complexes with folded antigens
    ('1N8Z', 'Anti-hen egg lysozyme (HEL) Fab', 'hel'),
    ('1BJ1', 'Anti-VEGF Fab', 'vegf'),
    ('3PP4', 'Anti-IL-13 Fab', 'il13'),
    ('4FQI', 'Anti-Factor D Fab', 'factor_d'),
    ('5JDS', 'Anti-PCSK9 Fab', 'pcsk9'),
    ('3BKJ', 'Anti-TNFα Fab', 'tnfa'),
    ('4RFO', 'Anti-GFP nanobody', 'gfp'),
    ('4KRL', 'Anti-BRIL Fab', 'bril'),
    ('1YY9', 'Anti-EGFR Fab', 'egfr'),
    ('3WD5', 'Anti-HER2 Fab', 'her2'),
    ('4NZU', 'Anti-RSV F Fab', 'rsv_f'),
    ('5FUC', 'Anti-HIV gp120 Fab', 'hiv_gp120'),
    ('3U1S', 'Anti-influenza HA Fab', 'ha'),
    ('5JQ3', 'Anti-DLL4 Fab', 'dll4'),
    ('7BWJ', 'Anti-SARS-CoV-2 RBD Fab', 'covid_rbd'),
    ('7CDI', 'Anti-SARS-CoV-2 RBD Fab', 'covid_rbd'),
    ('7CH5', 'Anti-SARS-CoV-2 RBD Fab', 'covid_rbd'),
    ('7CH4', 'Anti-SARS-CoV-2 RBD Fab', 'covid_rbd'),
    ('6APC', 'Anti-HIV Env Fab', 'hiv_env'),
    ('4HSA', 'Anti-IL-17A Fab', 'il17'),
]

FOLDED_CHAIN_MAP = {
    '1N8Z': {'H': 'B', 'L': 'A', 'Ag': 'C'},  # Heavy=B, Light=A, Antigen=C (lysozyme)
    '1BJ1': {'H': 'H', 'L': 'L', 'Ag': 'V'},
    '3PP4': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '4FQI': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '5JDS': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '3BKJ': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '4RFO': {'H': 'H', 'L': 'L', 'Ag': 'G'},
    '4KRL': {'H': 'B', 'L': '',  'Ag': 'A'},
    '1YY9': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '3WD5': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '4NZU': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '5FUC': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '3U1S': {'H': 'H', 'L': 'L', 'Ag': 'G'},
    '5JQ3': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '7BWJ': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '7CDI': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '7CH5': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '7CH4': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '6APC': {'H': 'H', 'L': 'L', 'Ag': 'A'},
    '4HSA': {'H': 'H', 'L': 'L', 'Ag': 'A'},
}

OUT_DIR = 'way4_s0/data'
PDB_DIR = os.path.join(OUT_DIR, 'pdbs')
SABDAB_DIR = os.path.join(OUT_DIR, 'sabdab_chothia')


# =============================================================================
# PDB Download — Primary: SAbDab Chothia-numbered; Fallback: RCSB
# =============================================================================

def download_pdb_sabdab(pdb_id, out_dir=SABDAB_DIR):
    """Download Chothia-numbered PDB from SAbDab.

    SAbDab provides pre-numbered PDBs where residue numbering follows the
    Chothia scheme, allowing direct CDR identification via ChothiaCDRRange.
    URL: http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{pdb_id}/
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f'{pdb_id}.pdb')

    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 1000:
        return pdb_path

    # SAbDab Chothia-numbered download
    # Try multiple possible URL patterns
    urls = [
        f'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{pdb_id.lower()}/',
        f'https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{pdb_id.lower()}/',
        f'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{pdb_id.upper()}/',
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                # if resp.status != 200: continue -> handled by urlopen
                content = resp.read()
                # Check if it's a valid PDB (starts with ATOM/HEADER/etc.)
                text = content.decode('utf-8', errors='ignore')[:200]
                if 'ATOM' in text or 'HEADER' in text:
                    with open(pdb_path, 'wb') as f:
                        f.write(content)
                    print(f'Chothia-numbered OK ({len(content)} bytes) from SAbDab')
                    return pdb_path
                else:
                    print(f'SAbDab returned non-PDB content ({len(content)} bytes)')
        except Exception as e:
            continue

    return None


def download_pdb_rcsb(pdb_id, out_dir=PDB_DIR):
    """Download PDB file from RCSB (fallback, no Chothia numbering)."""
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f'{pdb_id}.pdb')

    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 1000:
        return pdb_path, False  # is_chothia = False

    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    try:
        urllib.request.urlretrieve(url, pdb_path)
        size = os.path.getsize(pdb_path)
        if size < 1000:
            os.unlink(pdb_path)
            raise ValueError(f'File too small ({size} bytes)')
        return pdb_path, False
    except Exception as e:
        # Try biological assembly
        try:
            url2 = f'https://files.rcsb.org/download/{pdb_id}.pdb1'
            urllib.request.urlretrieve(url2, pdb_path)
            if os.path.getsize(pdb_path) > 1000:
                return pdb_path, False
        except Exception:
            pass
        return None, False


def download_pdb(pdb_id, prefer_chothia=True):
    """Download PDB from RCSB. (SAbDab2 no longer serves raw PDB files.)

    Returns: (pdb_path, is_chothia) or (None, False)
    """
    # SAbDab2 is a React SPA — no direct PDB download. Use RCSB directly.
    rcsb_path, _ = download_pdb_rcsb(pdb_id)
    return rcsb_path, False


# =============================================================================
# PDB Parsing & B-factor Extraction
# =============================================================================

def parse_pdb_chains(pdb_path):
    """Parse a PDB file and return per-chain residue data with B-factors.

    Uses Bio.PDB for robust structure parsing.
    Returns: {chain_id: [{'resn': str, 'resi': int, 'icode': str,
                          'ca_bfactor': float, 'all_bfactor': float,
                          'atoms': {name: (x,y,z,bfactor)}}, ...]}
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    model = structure[0]

    chains = {}
    for chain in model:
        cid = chain.id.strip()
        residues = []
        for res in chain:
            # Only standard amino acids
            try:
                hetfield, resseq, icode = res.get_id()
            except Exception:
                continue
            if hetfield.strip() != '' and hetfield != 'H_MSE':
                continue  # Skip HETATM except selenomethionine

            resname = res.get_resname().strip()
            # Skip non-standard residues
            if resname not in AA3_STANDARD and resname != 'MSE':
                continue

            atoms = {}
            ca_bfactor = None
            all_bfactors = []

            for atom in res:
                aname = atom.get_name().strip()
                x, y, z = atom.get_coord()
                bfactor = atom.get_bfactor()
                atoms[aname] = (float(x), float(y), float(z), float(bfactor))
                all_bfactors.append(float(bfactor))
                if aname == 'CA':
                    ca_bfactor = float(bfactor)

            if ca_bfactor is None and all_bfactors:
                ca_bfactor = np.mean(all_bfactors)

            residues.append({
                'resn': resname if resname != 'MSE' else 'MET',
                'resi': resseq,
                'icode': icode.strip(),
                'ca_bfactor': ca_bfactor,
                'all_bfactor': np.mean(all_bfactors) if all_bfactors else None,
                'atoms': atoms,
            })

        if residues:
            chains[cid] = residues

    return chains


AA3_STANDARD = {
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
}

AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


# =============================================================================
# CDR-H3 Identification
# Primary: Chothia numbering (from SAbDab pre-numbered PDBs)
# Fallback: Pattern-based (conserved Cys/WGXG motifs)
# =============================================================================

# Import ChothiaCDRRange from the existing codebase
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from disorderflow.utils.protein.constants import ChothiaCDRRange, CDR
    HAS_CHOTHIA_RANGE = True
except ImportError:
    HAS_CHOTHIA_RANGE = False


def detect_chothia_numbering(chain_residues):
    """Auto-detect if a chain uses Chothia antibody numbering.

    Chothia signature: Cys at resseq ~22 (FR1) and ~92 (FR3) in VH domain,
    with VH domain being ~120 residues.

    Returns True if Chothia numbering is detected.
    """
    # Check if there are Cys at Chothia positions 22 and 92
    cys_resseqs = []
    for res in chain_residues[:130]:  # Only look in VH domain (~first 125 res)
        if res['resn'] == 'CYS':
            cys_resseqs.append(res.get('resi', 0))

    has_c22 = any(20 <= r <= 26 for r in cys_resseqs)
    has_c92 = any(88 <= r <= 99 for r in cys_resseqs)

    return has_c22 and has_c92


def find_cdr_h3_chothia(chain_residues):
    """Find CDR-H3 using Chothia residue numbering (resseq 95-102).

    Handles insertion codes (e.g., 100A, 100B) common in long CDR-H3s.

    Returns: (cdr_h3_start_idx, cdr_h3_end_idx, cdr_h3_seq) or (None, None, None)
    """
    # Collect all residues in the Chothia CDR-H3 range (95-102)
    # including insertion codes
    cdr_indices = []
    for i, res in enumerate(chain_residues[:130]):  # Only VH domain
        resseq = res.get('resi', 0)
        if 95 <= resseq <= 102:
            cdr_indices.append(i)

    if not cdr_indices:
        return None, None, None

    start = cdr_indices[0]
    end = cdr_indices[-1] + 1
    seq = ''.join(AA3_TO_1.get(chain_residues[i]['resn'], 'X') for i in range(start, end))
    return start, end, seq


def find_cdr_h3_pattern(chain_residues, heavy_chain=True):
    """Find CDR-H3 using conserved sequence patterns in VH domain only.

    CDR-H3 is bounded by:
      - N-terminal: Cys at ~Chothia 92 (conserved Cys in FR3)
      - C-terminal: WGXG motif (FR4 start)

    Searches only in the VH domain (~first 125 residues of heavy chain).

    Returns: (cdr_h3_start_idx, cdr_h3_end_idx, cdr_h3_sequence)
             or (None, None, None) if not found
    """
    # Restrict to VH domain: ~first 125 residues
    vh_residues = chain_residues[:125]
    seq_1 = ''.join(AA3_TO_1.get(r['resn'], 'X') for r in vh_residues)

    if len(seq_1) < 80:
        return None, None, None

    # Find Cys residues in VH
    cys_positions = [i for i, r in enumerate(vh_residues) if r['resn'] == 'CYS']
    if not cys_positions:
        return None, None, None

    # Find WGXG or WG pattern (FR4 start)
    # WGXG where X is typically Q, but also R, K, or in some cases G
    wg_positions = []
    for i in range(len(seq_1) - 3):
        if seq_1[i] == 'W' and seq_1[i+1] == 'G' and seq_1[i+3] == 'G':
            wg_positions.append(i)  # WGxG pattern

    if not wg_positions:
        # Relaxed: just WG
        for i in range(len(seq_1) - 1):
            if seq_1[i:i+2] == 'WG':
                wg_positions.append(i)

    if not wg_positions:
        return None, None, None

    # Find the WG that's closest to a preceding Cys (within 5-25 residues)
    best_cys = None
    best_wg = None
    best_gap = 999

    for cys_pos in cys_positions:
        for wg_pos in wg_positions:
            gap = wg_pos - cys_pos
            if 5 <= gap <= 25 and gap < best_gap:
                best_gap = gap
                best_cys = cys_pos
                best_wg = wg_pos

    if best_cys is None:
        # Try the last Cys before the first WG
        for wg_pos in wg_positions:
            preceding_cys = [c for c in cys_positions if c < wg_pos]
            if preceding_cys:
                best_cys = preceding_cys[-1]
                best_wg = wg_pos
                best_gap = best_wg - best_cys
                break

    if best_cys is None or best_wg is None or best_gap < 3:
        return None, None, None

    # CDR-H3 starts 3 residues after C92 (skip AR/AK framework dipeptide)
    cdr_start = best_cys + 3
    cdr_end = best_wg

    if cdr_start >= cdr_end:
        return None, None, None

    # Validate length: CDR-H3 should be 2-30 residues
    if (cdr_end - cdr_start) > 30:
        # Try cys+2 instead
        cdr_start = best_cys + 2
        if cdr_start >= cdr_end or (cdr_end - cdr_start) > 30:
            return None, None, None

    cdr_h3_seq = seq_1[cdr_start:cdr_end]
    return cdr_start, cdr_end, cdr_h3_seq


# =============================================================================
# ANARCI-based CDR identification (preferred, if available)
# =============================================================================

def find_cdr_h3_anarci(heavy_sequence):
    """Use ANARCI to number antibody and find CDR-H3 boundaries.

    Args:
        heavy_sequence: 1-letter amino acid sequence of VH domain

    Returns:
        (cdr_h3_start, cdr_h3_end, cdr_h3_seq) or None
    """
    try:
        import anarci
        results = anarci.anarci([('heavy', heavy_sequence)], scheme='chothia',
                                output=False)
    except ImportError:
        return None, None, None

    if not results or not results[0]:
        return None, None, None

    numbering = results[0][0][0]  # (seq_id, [(residue_number, aa), ...])
    if not numbering:
        return None, None, None

    residues = numbering[1]

    # Find CDR-H3 in Chothia scheme: residues 95-102
    cdr_h3_residues = []
    cdr_h3_start_idx = None
    cdr_h3_end_idx = None

    for i, (resseq, aa) in enumerate(residues):
        if resseq >= 95 and resseq <= 102 and aa != '-':
            if cdr_h3_start_idx is None:
                cdr_h3_start_idx = i
            cdr_h3_end_idx = i
            cdr_h3_residues.append(aa)

    if not cdr_h3_residues:
        return None, None, None

    cdr_h3_seq = ''.join(cdr_h3_residues).replace('-', '')
    return cdr_h3_start_idx, cdr_h3_end_idx + 1, cdr_h3_seq


# =============================================================================
# Epitope Identification
# =============================================================================

def identify_epitope_interface(chains, antigen_chain_id, heavy_chain_id,
                               light_chain_id=None, cutoff=8.0):
    """Identify epitope residues on antigen within contact distance of antibody.

    Args:
        chains: {chain_id: [residue_dicts]}
        antigen_chain_id: chain ID of antigen
        heavy_chain_id: chain ID of antibody heavy chain
        light_chain_id: chain ID of antibody light chain (or None for nanobody)
        cutoff: distance cutoff in Å for contact

    Returns:
        List of epitope residue indices (0-based in antigen chain)
    """
    if antigen_chain_id not in chains:
        return []

    # Collect antibody CA atoms
    ab_ca_coords = []
    for cid in [heavy_chain_id, light_chain_id]:
        if not cid or cid not in chains:
            continue
        for res in chains[cid]:
            if 'CA' in res['atoms']:
                ab_ca_coords.append(res['atoms']['CA'][:3])

    if not ab_ca_coords:
        return []

    ab_ca = np.array(ab_ca_coords)  # (N, 3)

    # Collect antigen CA atoms
    ag_chain = chains[antigen_chain_id]
    epitope_indices = []
    for i, res in enumerate(ag_chain):
        if 'CA' not in res['atoms']:
            continue
        ag_ca = np.array(res['atoms']['CA'][:3])
        min_dist = np.min(np.sqrt(np.sum((ab_ca - ag_ca)**2, axis=1)))
        if min_dist <= cutoff:
            epitope_indices.append(i)

    return epitope_indices


# =============================================================================
# Disorder/Flexibility Prediction
# =============================================================================

# IUPred3-like disorder propensity: uses amino-acid-specific energy potentials
# Simplified version based on the physical principles of IUPred
# (pairwise interaction energy deficit for disordered regions)

# Disorder propensity scale (from literature: Campen et al., 2008;
# TOP-IDP scale, normalized to [0, 1], higher = more disorder-prone)
DISORDER_PROPENSITY = {
    'A': 0.30, 'C': 0.25, 'D': 0.55, 'E': 0.60, 'F': 0.35,
    'G': 0.55, 'H': 0.45, 'I': 0.30, 'K': 0.65, 'L': 0.30,
    'M': 0.35, 'N': 0.50, 'P': 0.70, 'Q': 0.55, 'R': 0.60,
    'S': 0.50, 'T': 0.45, 'V': 0.28, 'W': 0.35, 'Y': 0.40,
}


def compute_disorder_aa_propensity(sequence_1letter):
    """Compute per-residue disorder propensity based on AA scale.

    Simple windowed average of DISORDER_PROPENSITY values.
    Returns: list of [0,1] scores, one per residue
    """
    scores = []
    for aa in sequence_1letter:
        scores.append(DISORDER_PROPENSITY.get(aa, 0.5))
    # Smooth with window=5
    smoothed = []
    window = 5
    for i in range(len(scores)):
        start = max(0, i - window // 2)
        end = min(len(scores), i + window // 2 + 1)
        smoothed.append(float(np.mean(scores[start:end])))
    return smoothed


def compute_disorder_iupred_web(sequence_1letter):
    """Attempt to compute disorder using IUPred3 web server.

    Falls back to AA propensity if web service unavailable.
    """
    # IUPred3 REST API (if available) or local installation
    # For now, use the AA propensity as a proxy
    # The plan says to use IUPred3/pLDDT/MD, and we'll validate consistency
    return None  # Placeholder for web/API integration


def compute_b_factor_zscore(chain_residues):
    """Compute per-residue B-factor z-scores normalized within the chain.

    Returns: list of z-scores, one per residue (higher = more flexible)
    """
    bfactors = []
    for res in chain_residues:
        bf = res.get('ca_bfactor')
        if bf is not None:
            bfactors.append(bf)

    if not bfactors:
        return [0.0] * len(chain_residues)

    mean_bf = np.mean(bfactors)
    std_bf = np.std(bfactors)
    if std_bf == 0:
        std_bf = 1.0

    zscores = []
    for res in chain_residues:
        bf = res.get('ca_bfactor')
        if bf is not None:
            zscores.append((bf - mean_bf) / std_bf)
        else:
            zscores.append(0.0)

    return zscores


# =============================================================================
# Main Collection Routine
# =============================================================================

def collect_antibody_data(pdb_id, description, group, chain_map):
    """Collect all metrics for a single antibody-antigen complex.

    Returns: dict with CDR-H3 flexibility, epitope flexibility, and metadata.
    """
    result = {
        'pdb_id': pdb_id,
        'description': description,
        'group': group,
        'status': 'FAILED',
        'error': None,
    }

    # Skip PDBs with no valid chain map (explicitly excluded)
    if chain_map is None:
        result['error'] = 'Excluded: not a well-characterized antibody-antigen complex'
        return result

    # Download PDB (prefer Chothia-numbered from SAbDab)
    pdb_path, is_chothia = download_pdb(pdb_id, prefer_chothia=True)
    if pdb_path is None:
        result['error'] = 'PDB download failed (both SAbDab and RCSB)'
        return result

    result['pdb_source'] = 'sabdab_chothia' if is_chothia else 'rcsb'

    # Parse chains
    try:
        chains = parse_pdb_chains(pdb_path)
    except Exception as e:
        result['error'] = f'PDB parse error: {e}'
        return result

    # Get chain mapping
    h_chain = chain_map.get('H', 'H')
    l_chain = chain_map.get('L', '')
    ag_chain = chain_map.get('Ag', 'A')

    # Verify chains exist
    available_chains = list(chains.keys())
    result['available_chains'] = available_chains

    if h_chain not in chains:
        # Try to auto-detect heavy chain (longest chain)
        chain_lengths = {c: len(chains[c]) for c in chains}
        sorted_chains = sorted(chain_lengths, key=chain_lengths.get, reverse=True)
        if len(sorted_chains) >= 2:
            h_chain = sorted_chains[0]
            if not l_chain or l_chain not in chains:
                l_chain = sorted_chains[1] if len(sorted_chains) > 2 else ''
            ag_chain = sorted_chains[-1]
            result['auto_detected_chains'] = True
            result['detected_H'] = h_chain
            result['detected_L'] = l_chain
            result['detected_Ag'] = ag_chain
        else:
            result['error'] = f'Heavy chain {h_chain} not found in {available_chains}'
            return result

    heavy_residues = chains[h_chain]
    heavy_seq_1 = ''.join(AA3_TO_1.get(r['resn'], 'X') for r in heavy_residues)

    # --- CDR-H3 Identification ---
    cdr_h3_start, cdr_h3_end, cdr_h3_seq = None, None, None
    method = 'unknown'

    # Auto-detect Chothia numbering (works for many RCSB PDBs)
    is_chothia = detect_chothia_numbering(heavy_residues)
    result['chothia_detected'] = is_chothia

    # Try pattern-based first (works regardless of numbering, restricted to VH)
    cdr_h3_start, cdr_h3_end, cdr_h3_seq = find_cdr_h3_pattern(heavy_residues)
    method = 'pattern'

    # Fall back to Chothia numbering if pattern fails
    if cdr_h3_start is None and is_chothia:
        cdr_h3_start, cdr_h3_end, cdr_h3_seq = find_cdr_h3_chothia(heavy_residues)
        method = 'chothia_numbering'

    # Last resort: ANARCI
    if cdr_h3_start is None:
        cdr_h3_start, cdr_h3_end, cdr_h3_seq = find_cdr_h3_anarci(heavy_seq_1)
        method = 'ANARCI'

    if cdr_h3_start is None:
        result['error'] = 'CDR-H3 not identified'
        return result

    # --- CDR-H3 B-factors ---
    cdr_h3_bfactors = []
    for i in range(cdr_h3_start, cdr_h3_end):
        if i < len(heavy_residues):
            bf = heavy_residues[i].get('ca_bfactor')
            if bf is not None:
                cdr_h3_bfactors.append(bf)

    # --- Framework B-factors (for normalization) ---
    fw_bfactors = []
    for i, res in enumerate(heavy_residues):
        if i < cdr_h3_start or i >= cdr_h3_end:
            bf = res.get('ca_bfactor')
            if bf is not None:
                fw_bfactors.append(bf)

    # --- Global B-factor normalization ---
    all_bfactors = cdr_h3_bfactors + fw_bfactors
    if all_bfactors:
        global_mean = np.mean(all_bfactors)
        global_std = np.std(all_bfactors)
        if global_std == 0:
            global_std = 1.0

        # CDR-H3 relative B-factor (how much more flexible than framework)
        cdr_h3_mean_bf = np.mean(cdr_h3_bfactors) if cdr_h3_bfactors else None
        fw_mean_bf = np.mean(fw_bfactors) if fw_bfactors else None

        cdr_h3_bf_zscore = (cdr_h3_mean_bf - global_mean) / global_std if cdr_h3_mean_bf else None
        cdr_h3_bf_vs_fw = (cdr_h3_mean_bf / fw_mean_bf) if (cdr_h3_mean_bf and fw_mean_bf and fw_mean_bf > 0) else None

        # Per-residue B-factor z-scores
        bf_zscores = []
        for i, res in enumerate(heavy_residues):
            bf = res.get('ca_bfactor')
            if bf is not None:
                bf_zscores.append((bf - global_mean) / global_std)
            else:
                bf_zscores.append(0.0)
    else:
        cdr_h3_mean_bf = None
        fw_mean_bf = None
        cdr_h3_bf_zscore = None
        cdr_h3_bf_vs_fw = None
        bf_zscores = []

    result['cdr_h3'] = {
        'method': method,
        'start_idx': cdr_h3_start,
        'end_idx': cdr_h3_end,
        'length': len(cdr_h3_seq),
        'sequence': cdr_h3_seq,
        'mean_bfactor': cdr_h3_mean_bf,
        'bfactors': cdr_h3_bfactors,
        'bf_zscore': cdr_h3_bf_zscore,
        'bf_vs_framework': cdr_h3_bf_vs_fw,
    }

    result['framework'] = {
        'mean_bfactor': fw_mean_bf,
        'bfactors': fw_bfactors,
    }

    result['heavy_chain'] = {
        'chain_id': h_chain,
        'length': len(heavy_residues),
        'sequence': heavy_seq_1,
        'bfactor_zscores': bf_zscores,
    }

    # --- Epitope Analysis ---
    if ag_chain not in chains:
        result['epitope'] = {'status': 'antigen_chain_not_found', 'chain_id_expected': ag_chain}
        result['status'] = 'PARTIAL'  # Antibody data OK, but no epitope
        return result

    ag_residues = chains[ag_chain]
    ag_seq_1 = ''.join(AA3_TO_1.get(r['resn'], 'X') for r in ag_residues)

    # Identify interface epitope residues
    epitope_indices = identify_epitope_interface(
        chains, ag_chain, h_chain, l_chain if l_chain else None)

    # Epitope disorder prediction (AA propensity scale)
    ag_disorder = compute_disorder_aa_propensity(ag_seq_1)

    # Epitope B-factors
    ag_bf_zscores = []
    ag_bfactors_all = []
    for res in ag_residues:
        bf = res.get('ca_bfactor')
        if bf is not None:
            ag_bfactors_all.append(bf)

    if ag_bfactors_all:
        ag_bf_mean = np.mean(ag_bfactors_all)
        ag_bf_std = np.std(ag_bfactors_all)
        if ag_bf_std == 0:
            ag_bf_std = 1.0
        for res in ag_residues:
            bf = res.get('ca_bfactor')
            if bf is not None:
                ag_bf_zscores.append((bf - ag_bf_mean) / ag_bf_std)
            else:
                ag_bf_zscores.append(0.0)
    else:
        ag_bf_zscores = [0.0] * len(ag_residues)

    # Epitope-specific metrics
    epitope_disorder = [ag_disorder[i] for i in epitope_indices] if epitope_indices else []
    epitope_bf = [ag_bf_zscores[i] for i in epitope_indices] if epitope_indices else []

    result['epitope'] = {
        'chain_id': ag_chain,
        'antigen_length': len(ag_residues),
        'antigen_sequence': ag_seq_1,
        'epitope_residue_count': len(epitope_indices),
        'epitope_indices': epitope_indices,
        'epitope_disorder_aa_scale': epitope_disorder,
        'epitope_bf_zscores': epitope_bf,
        'mean_disorder': float(np.mean(epitope_disorder)) if epitope_disorder else None,
        'mean_bf_zscore': float(np.mean(epitope_bf)) if epitope_bf else None,
        'all_residue_disorder': ag_disorder,
        'all_residue_bf_zscores': ag_bf_zscores,
    }

    # --- Compute key metrics for statistical test ---
    result['metrics'] = {
        'cdr_h3_mean_bfactor': cdr_h3_mean_bf,
        'cdr_h3_bf_zscore': cdr_h3_bf_zscore,
        'cdr_h3_bf_vs_framework': cdr_h3_bf_vs_fw,
        'cdr_h3_length': len(cdr_h3_seq),
        'epitope_mean_disorder': float(np.mean(epitope_disorder)) if epitope_disorder else None,
        'epitope_mean_bf': float(np.mean(epitope_bf)) if epitope_bf else None,
    }

    result['status'] = 'OK'
    return result


# =============================================================================
# Statistical Analysis
# =============================================================================

def run_statistical_test(all_results):
    """Test the Way4 thesis: CDR-H3 flexibility correlates with epitope flexibility.

    Null hypothesis: No correlation between epitope flexibility and CDR-H3 flexibility.
    Alternative: Anti-IDP antibodies have more flexible CDR-H3 (higher B-factor)
                 compared to anti-folded antibodies, and CDR-H3 flexibility correlates
                 with epitope flexibility.

    Tests:
      1. Mann-Whitney U: CDR-H3 B-factor (IDP vs folded antigen groups)
      2. Spearman correlation: epitope disorder vs CDR-H3 B-factor
      3. Within-group correlation (IDP group only)
    """
    from scipy import stats

    results_ok = [r for r in all_results if r['status'] in ('OK', 'PARTIAL')
                   and 'metrics' in r and r['metrics'].get('cdr_h3_mean_bfactor') is not None]

    idp_results = [r for r in results_ok if r['group'] in ('anti_abeta', 'anti_tau', 'anti_asyn')]
    folded_results = [r for r in results_ok if r['group'] not in ('anti_abeta', 'anti_tau', 'anti_asyn')]

    print(f'\n{"="*60}')
    print(f'S0 Statistical Analysis')
    print(f'{"="*60}')
    print(f'Total PDBs processed: {len(all_results)}')
    print(f'OK/Partial: {len(results_ok)}')
    print(f'Anti-IDP group: {len(idp_results)}')
    print(f'Folded-antigen group: {len(folded_results)}')

    analysis = {
        'total_processed': len(all_results),
        'ok_count': len(results_ok),
        'idp_count': len(idp_results),
        'folded_count': len(folded_results),
        'tests': {},
    }

    # Test 1: Group comparison (IDP vs Folded)
    idp_cdr_bf = [r['metrics']['cdr_h3_mean_bfactor'] for r in idp_results
                   if r['metrics']['cdr_h3_mean_bfactor'] is not None]
    folded_cdr_bf = [r['metrics']['cdr_h3_mean_bfactor'] for r in folded_results
                      if r['metrics']['cdr_h3_mean_bfactor'] is not None]

    if len(idp_cdr_bf) >= 2 and len(folded_cdr_bf) >= 2:
        u_stat, u_pval = stats.mannwhitneyu(idp_cdr_bf, folded_cdr_bf,
                                             alternative='two-sided')

        print(f'\n--- Test 1: CDR-H3 B-factor (IDP vs Folded) ---')
        print(f'  IDP CDR-H3 B-factor:    {np.mean(idp_cdr_bf):.2f} ± {np.std(idp_cdr_bf):.2f} (n={len(idp_cdr_bf)})')
        print(f'  Folded CDR-H3 B-factor: {np.mean(folded_cdr_bf):.2f} ± {np.std(folded_cdr_bf):.2f} (n={len(folded_cdr_bf)})')
        print(f'  Mann-Whitney U: {u_stat:.2f}, p={u_pval:.4f}')
        print(f'  Significant (p<0.05): {u_pval < 0.05}')

        analysis['tests']['mann_whitney_bfactor'] = {
            'test': 'Mann-Whitney U (IDP vs Folded CDR-H3 B-factor)',
            'idp_mean': float(np.mean(idp_cdr_bf)),
            'idp_std': float(np.std(idp_cdr_bf)),
            'idp_n': len(idp_cdr_bf),
            'folded_mean': float(np.mean(folded_cdr_bf)),
            'folded_std': float(np.std(folded_cdr_bf)),
            'folded_n': len(folded_cdr_bf),
            'u_statistic': float(u_stat),
            'p_value': float(u_pval),
            'significant_p05': bool(u_pval < 0.05),
        }

    # Test 1b: CDR-H3 B-factor Z-score comparison
    idp_cdr_bf_z = [r['metrics']['cdr_h3_bf_zscore'] for r in idp_results
                     if r['metrics']['cdr_h3_bf_zscore'] is not None]
    folded_cdr_bf_z = [r['metrics']['cdr_h3_bf_zscore'] for r in folded_results
                        if r['metrics']['cdr_h3_bf_zscore'] is not None]

    if len(idp_cdr_bf_z) >= 2 and len(folded_cdr_bf_z) >= 2:
        u_stat_z, u_pval_z = stats.mannwhitneyu(idp_cdr_bf_z, folded_cdr_bf_z,
                                                  alternative='greater')  # one-sided: IDP > folded
        print(f'\n--- Test 1b: CDR-H3 B-factor Z-score (IDP > Folded, one-sided) ---')
        print(f'  IDP CDR-H3 B-factor Z:    {np.mean(idp_cdr_bf_z):.3f} ± {np.std(idp_cdr_bf_z):.3f}')
        print(f'  Folded CDR-H3 B-factor Z: {np.mean(folded_cdr_bf_z):.3f} ± {np.std(folded_cdr_bf_z):.3f}')
        print(f'  Mann-Whitney U (greater): p={u_pval_z:.4f}')
        print(f'  Significant (p<0.05): {u_pval_z < 0.05}')

        analysis['tests']['mann_whitney_bfactor_zscore'] = {
            'test': 'Mann-Whitney U (IDP > Folded CDR-H3 B-factor Z-score, one-sided)',
            'idp_mean_z': float(np.mean(idp_cdr_bf_z)),
            'idp_std_z': float(np.std(idp_cdr_bf_z)),
            'folded_mean_z': float(np.mean(folded_cdr_bf_z)),
            'folded_std_z': float(np.std(folded_cdr_bf_z)),
            'p_value': float(u_pval_z),
            'significant_p05': bool(u_pval_z < 0.05),
        }

    # Test 2: Spearman correlation — epitope disorder vs CDR-H3 B-factor (all)
    epi_disorder_all = []
    cdr_bf_all = []
    labels_all = []
    for r in results_ok:
        ed = r['metrics'].get('epitope_mean_disorder')
        cb = r['metrics'].get('cdr_h3_mean_bfactor')
        if ed is not None and cb is not None:
            epi_disorder_all.append(ed)
            cdr_bf_all.append(cb)
            labels_all.append(r['pdb_id'])

    if len(epi_disorder_all) >= 5:
        spearman_r, spearman_p = stats.spearmanr(epi_disorder_all, cdr_bf_all)

        print(f'\n--- Test 2: Epitope disorder vs CDR-H3 B-factor (all groups) ---')
        print(f'  N={len(epi_disorder_all)}')
        print(f'  Spearman r={spearman_r:.4f}, p={spearman_p:.4f}')
        print(f'  Significant (p<0.05): {spearman_p < 0.05}')

        analysis['tests']['spearman_all'] = {
            'test': 'Spearman: epitope_disorder vs CDR-H3 B-factor (all)',
            'n': len(epi_disorder_all),
            'spearman_r': float(spearman_r),
            'p_value': float(spearman_p),
            'significant_p05': bool(spearman_p < 0.05),
        }

    # Test 3: Spearman within IDP group only
    idp_epi_disorder = []
    idp_cdr_bf = []
    for r in idp_results:
        ed = r['metrics'].get('epitope_mean_disorder')
        cb = r['metrics'].get('cdr_h3_mean_bfactor')
        if ed is not None and cb is not None:
            idp_epi_disorder.append(ed)
            idp_cdr_bf.append(cb)

    if len(idp_epi_disorder) >= 5:
        spr_idp, spp_idp = stats.spearmanr(idp_epi_disorder, idp_cdr_bf)

        print(f'\n--- Test 3: Epitope disorder vs CDR-H3 B-factor (IDP group only) ---')
        print(f'  N={len(idp_epi_disorder)}')
        print(f'  Spearman r={spr_idp:.4f}, p={spp_idp:.4f}')
        print(f'  Significant (p<0.05): {spp_idp < 0.05}')

        analysis['tests']['spearman_idp_only'] = {
            'test': 'Spearman: epitope_disorder vs CDR-H3 B-factor (IDP only)',
            'n': len(idp_epi_disorder),
            'spearman_r': float(spr_idp),
            'p_value': float(spp_idp),
            'significant_p05': bool(spp_idp < 0.05),
        }

    # Test 4: B-factor ratio (CDR-H3 / framework) comparison
    idp_bf_ratio = [r['metrics']['cdr_h3_bf_vs_framework'] for r in idp_results
                     if r['metrics']['cdr_h3_bf_vs_framework'] is not None]
    folded_bf_ratio = [r['metrics']['cdr_h3_bf_vs_framework'] for r in folded_results
                        if r['metrics']['cdr_h3_bf_vs_framework'] is not None]

    if len(idp_bf_ratio) >= 2 and len(folded_bf_ratio) >= 2:
        u_ratio, p_ratio = stats.mannwhitneyu(idp_bf_ratio, folded_bf_ratio,
                                                alternative='greater')

        print(f'\n--- Test 4: CDR-H3/Framework B-factor ratio (IDP > Folded) ---')
        print(f'  IDP B-factor ratio:    {np.mean(idp_bf_ratio):.3f} ± {np.std(idp_bf_ratio):.3f}')
        print(f'  Folded B-factor ratio: {np.mean(folded_bf_ratio):.3f} ± {np.std(folded_bf_ratio):.3f}')
        print(f'  Mann-Whitney U (greater): p={p_ratio:.4f}')

        analysis['tests']['mann_whitney_bf_ratio'] = {
            'test': 'Mann-Whitney U (IDP > Folded CDR-H3/FW B-factor ratio)',
            'idp_mean_ratio': float(np.mean(idp_bf_ratio)),
            'folded_mean_ratio': float(np.mean(folded_bf_ratio)),
            'p_value': float(p_ratio),
            'significant_p05': bool(p_ratio < 0.05),
        }

    # Go/NoGo Decision
    print(f'\n{"="*60}')
    print(f'S0 Go/NoGo Decision')
    print(f'{"="*60}')

    go_conditions = []
    nogo_conditions = []

    # Condition 1: IDP CDR-H3 more flexible than folded (Test 1b)
    if 'mann_whitney_bfactor_zscore' in analysis['tests']:
        t = analysis['tests']['mann_whitney_bfactor_zscore']
        if t['significant_p05'] and t['idp_mean_z'] > t['folded_mean_z']:
            go_conditions.append('Test 1b PASSED: IDP CDR-H3 B-factor Z > Folded (p<0.05)')
        else:
            nogo_conditions.append(
                f'Test 1b NOT passed: IDP Z={t["idp_mean_z"]:.3f}, '
                f'Folded Z={t["folded_mean_z"]:.3f}, p={t["p_value"]:.4f}')

    # Condition 2: Positive correlation epitope disorder ↔ CDR-H3 B-factor (all)
    if 'spearman_all' in analysis['tests']:
        t = analysis['tests']['spearman_all']
        if t['significant_p05'] and t['spearman_r'] > 0:
            go_conditions.append(
                f'Test 2 PASSED: Epitope disorder ↔ CDR-H3 B-factor r={t["spearman_r"]:.3f}, p={t["p_value"]:.4f}')
        else:
            nogo_conditions.append(
                f'Test 2 NOT passed: r={t["spearman_r"]:.3f}, p={t["p_value"]:.4f}')

    # Decision
    if go_conditions:
        verdict = 'GO'
        verdict_reason = '; '.join(go_conditions)
    elif len(go_conditions) == 0 and len(nogo_conditions) >= 2:
        verdict = 'NoGo'
        verdict_reason = 'All tests fail: ' + '; '.join(nogo_conditions)
    else:
        verdict = 'WEAK_GO'
        verdict_reason = f'Mixed: GO={len(go_conditions)}, NoGo={len(nogo_conditions)}. '
        verdict_reason += 'Proceed with caution, validate with additional data.'

    print(f'\n  VERDICT: {verdict}')
    print(f'  Reason: {verdict_reason}')
    print(f'  GO conditions met: {len(go_conditions)}')
    print(f'  NoGo conditions:   {len(nogo_conditions)}')

    analysis['verdict'] = verdict
    analysis['verdict_reason'] = verdict_reason
    analysis['go_conditions'] = go_conditions
    analysis['nogo_conditions'] = nogo_conditions

    return analysis


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='S0: Way4 Thesis Observational Pre-check')
    parser.add_argument('--control-n', type=int, default=12,
                        help='Number of folded-antigen controls to include')
    parser.add_argument('--out', type=str, default='way4_s0_observational.json',
                        help='Output JSON path')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip PDB download if files exist')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PDB_DIR, exist_ok=True)

    all_results = []

    # --- Process Anti-IDP antibodies ---
    print('='*60)
    print('S0 Phase 1: Anti-IDP Antibody Complexes')
    print('='*60)

    for pdb_id, desc, group in ANTI_IDP_PDBS:
        print(f'\n[{pdb_id}] {desc}')
        cm = CHAIN_MAP.get(pdb_id, {'H': 'H', 'L': 'L', 'Ag': 'A'})
        result = collect_antibody_data(pdb_id, desc, group, cm)
        all_results.append(result)
        print(f'  Status: {result["status"]}')
        if result['status'] in ('OK', 'PARTIAL'):
            cdr = result.get('cdr_h3', {})
            epi = result.get('epitope', {})
            met = result.get('metrics', {})
            print(f'  CDR-H3 ({cdr.get("method", "?")}): len={cdr.get("length", "?")}, '
                  f'seq={cdr.get("sequence", "?")}')
            print(f'  CDR-H3 B-factor: {cdr.get("mean_bfactor", "?"):.2f}' if cdr.get('mean_bfactor') else f'  CDR-H3 B-factor: N/A')
            print(f'  Epitope: {epi.get("epitope_residue_count", 0)} contact residues, '
                  f'mean disorder={met.get("epitope_mean_disorder", "?"):.3f}' if met.get('epitope_mean_disorder') else f'  Epitope: {epi.get("epitope_residue_count", 0)} contact residues')
        else:
            print(f'  Error: {result.get("error", "unknown")}')
        time.sleep(0.3)  # Rate limit for RCSB

    # --- Process Folded-antigen Controls ---
    print(f'\n{"="*60}')
    print(f'S0 Phase 2: Folded-Antigen Control Antibodies')
    print(f'{"="*60}')

    controls = FOLDED_CONTROLS[:args.control_n]
    for pdb_id, desc, group in controls:
        print(f'\n[{pdb_id}] {desc}')
        cm = FOLDED_CHAIN_MAP.get(pdb_id, {'H': 'H', 'L': 'L', 'Ag': 'A'})
        result = collect_antibody_data(pdb_id, desc, group, cm)
        all_results.append(result)
        print(f'  Status: {result["status"]}')
        if result['status'] in ('OK', 'PARTIAL'):
            cdr = result.get('cdr_h3', {})
            met = result.get('metrics', {})
            print(f'  CDR-H3 ({cdr.get("method", "?")}): len={cdr.get("length", "?")}, '
                  f'seq={cdr.get("sequence", "?")}')
            print(f'  CDR-H3 B-factor: {cdr.get("mean_bfactor", "?"):.2f}' if cdr.get('mean_bfactor') else '  CDR-H3 B-factor: N/A')
        else:
            print(f'  Error: {result.get("error", "unknown")}')
        time.sleep(0.3)

    # --- Statistical Analysis ---
    analysis = run_statistical_test(all_results)

    # --- Save Output ---
    output = {
        'metadata': {
            'plan': 'WAY4_REALIZE_PLAN_V14',
            'phase': 'S0',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'thesis': 'Anti-IDP antibody CDR-H3 flexibility correlates with epitope flexibility',
            'verification_label': '[PENDING]',  # Always start as PENDING per asset discipline
        },
        'results': all_results,
        'analysis': analysis,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f'\n{"="*60}')
    print(f'Output saved: {args.out}')
    print(f'Verification label: [PENDING] — requires manual review before [VERIFIED]')
    print(f'{"="*60}')

    return output


if __name__ == '__main__':
    main()
