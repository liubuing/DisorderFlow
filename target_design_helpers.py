"""
Target design helper functions — epitope prediction, contact analysis, interface scoring.

All functions work CPU-only using BioPython, numpy, and scipy.
"""

import os
import numpy as np
from scipy.spatial import KDTree

# ── Residue properties ──

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

# Theoretical max SASA values (Tien et al. 2013, standard tripeptide reference)
MAX_SASA = {
    'A': 113.0, 'R': 241.0, 'N': 158.0, 'D': 151.0, 'C': 140.0,
    'Q': 189.0, 'E': 183.0, 'G': 85.0,  'H': 194.0, 'I': 182.0,
    'L': 180.0, 'K': 211.0, 'M': 204.0, 'F': 218.0, 'P': 143.0,
    'S': 122.0, 'T': 146.0, 'W': 259.0, 'Y': 229.0, 'V': 160.0,
}

# Kyte-Doolittle hydropathy (negated: positive = hydrophilic)
KYTE_DOOLITTLE = {
    'A': -1.8, 'C': -2.5, 'D': 3.5,  'E': 3.5,  'F': -2.8,
    'G': 0.4,  'H': 3.2,  'I': -4.5, 'K': 3.9,  'L': -3.8,
    'M': -1.9, 'N': 3.5,  'P': 1.6,  'Q': 3.5,  'R': 4.5,
    'S': 0.8,  'T': 0.7,  'V': -4.2, 'W': 0.9,  'Y': 1.3,
}

# Eisenberg consensus hydrophobicity (for interface analysis)
EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08,
}

POSITIVE_AA = {'K', 'R', 'H'}
NEGATIVE_AA = {'D', 'E'}


# ── PDB parsing helpers ──

def _parse_structure(pdb_path):
    """Parse PDB with BioPython, return (structure, header info)."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    pid = os.path.basename(pdb_path).replace('.pdb', '')
    structure = parser.get_structure(pid, pdb_path)
    return structure


def _get_chain_residues(structure, chain_id):
    """Get CA-only residues for a chain. Returns list of (residue, resseq, resname_1)."""
    results = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if 'CA' in res:
                        resname = res.get_resname().strip()
                        one = AA_3TO1.get(resname, 'X')
                        results.append((res, res.get_id()[1], one))
    return results


def _extract_sequence(structure, chain_id):
    """Extract amino acid sequence string from a chain."""
    residues = _get_chain_residues(structure, chain_id)
    return ''.join(r[2] for r in residues)


# ── SASA computation ──

def compute_residue_sasa(pdb_path, chain_id=None):
    """
    Compute per-residue Solvent Accessible Surface Area via Shrake-Rupley.

    Args:
        pdb_path: path to PDB file
        chain_id: chain to analyze, or None for first chain

    Returns:
        List of dicts: [{chain_id, resseq, resname, sasa, rel_sasa}, ...]
        Sorted by sasa descending.
    """
    from Bio.PDB.SASA import ShrakeRupley
    structure = _parse_structure(pdb_path)

    sr = ShrakeRupley()
    sr.compute(structure[0], level="R")

    if chain_id is None:
        for model in structure:
            for chain in model:
                chain_id = chain.id
                break
            break

    results = []
    residues = _get_chain_residues(structure, chain_id)
    for res, resseq, resname in residues:
        sasa = getattr(res, 'sasa', 0.0) or 0.0
        max_s = MAX_SASA.get(resname, 200.0)
        rel_sasa = sasa / max_s if max_s > 0 else 0.0
        results.append({
            'chain_id': chain_id,
            'resseq': resseq,
            'resname': resname,
            'sasa': round(sasa, 2),
            'rel_sasa': round(rel_sasa, 4),
        })
    results.sort(key=lambda x: x['sasa'], reverse=True)
    return results


# ── Hydrophilicity ──

def compute_hydrophilicity(sequence):
    """Compute per-residue Kyte-Doolittle hydrophilicity (positive = surface-exposed)."""
    return [KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence]


# ── Protrusion index ──

def compute_protrusion_index(pdb_path, chain_id=None, neighborhood_radius=10.0):
    """
    Compute per-residue protrusion index: distance of CA from centroid of
    neighboring CAs within neighborhood_radius.

    Higher values = residue sticks out more = more likely antigenic.
    """
    structure = _parse_structure(pdb_path)
    residues = _get_chain_residues(structure, chain_id or 'A')

    if not residues:
        return []

    # Collect CA coordinates
    coords = np.array([res['CA'].get_coord() for res, _, _ in residues])
    tree = KDTree(coords)

    protrusions = []
    for i, (res, resseq, resname) in enumerate(residues):
        neighbors = tree.query_ball_point(coords[i], neighborhood_radius)
        if len(neighbors) > 1:
            centroid = coords[neighbors].mean(axis=0)
            protrusion = np.linalg.norm(coords[i] - centroid)
        else:
            protrusion = 0.0
        protrusions.append(protrusion)

    if protrusions:
        pmin, pmax = min(protrusions), max(protrusions)
        if pmax > pmin:
            protrusions = [(p - pmin) / (pmax - pmin) for p in protrusions]
        else:
            protrusions = [0.0 for _ in protrusions]

    return protrusions


# ── Epitope scoring ──

def score_epitope_residues(pdb_path, chain_id=None, weights=None):
    """
    Combine SASA, hydrophilicity, and protrusion into epitope likelihood score.

    Args:
        pdb_path: path to PDB
        chain_id: chain to analyze
        weights: dict with keys 'sasa', 'hydrophilicity', 'protrusion'

    Returns:
        List of dicts sorted by combined_score descending,
        with keys: chain_id, resseq, resname, sasa, rel_sasa,
                   hydrophilicity, protrusion, combined_score, rank
    """
    if weights is None:
        weights = {'sasa': 0.4, 'hydrophilicity': 0.35, 'protrusion': 0.25}

    # Get SASA data
    sasa_data = compute_residue_sasa(pdb_path, chain_id)
    if not sasa_data:
        return []

    used_chain = sasa_data[0]['chain_id']
    sequence = _extract_sequence(_parse_structure(pdb_path), used_chain)

    # Get hydrophilicity
    hydrophilicity = compute_hydrophilicity(sequence)

    # Get protrusion
    protrusions = compute_protrusion_index(pdb_path, used_chain)

    # Normalize helpers
    def _norm(values):
        vals = np.array(values, dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            return list((vals - vmin) / (vmax - vmin))
        return [0.0] * len(vals)

    sasa_vals = [d['rel_sasa'] for d in sasa_data]
    n_sasa = _norm(sasa_vals)
    n_hydro = _norm(hydrophilicity)
    n_prot = _norm(protrusions)

    w = weights
    for i, d in enumerate(sasa_data):
        idx = i  # SASA data order matches residue order
        combined = (w['sasa'] * n_sasa[idx] + w['hydrophilicity'] * n_hydro[idx] +
                    w['protrusion'] * n_prot[idx])
        d['hydrophilicity'] = round(hydrophilicity[idx], 3)
        d['protrusion'] = round(protrusions[idx], 4)
        d['combined_score'] = round(combined, 4)

    sasa_data.sort(key=lambda x: x['combined_score'], reverse=True)
    for i, d in enumerate(sasa_data):
        d['rank'] = i + 1

    return sasa_data


def format_epitope_table(epitope_data, top_n=20):
    """Format epitope scoring results as a readable text table."""
    if not epitope_data:
        return "无表位数据"

    header = f"{'排名':<5} {'链':<4} {'位点':<6} {'残基':<6} {'SASA(Å²)':<10} {'相对SASA':<10} {'亲水性':<8} {'凸出度':<8} {'综合得分':<10}"
    sep = '-' * len(header)
    lines = [f"表位候选残基 (Top {top_n})", sep, header, sep]

    for d in epitope_data[:top_n]:
        lines.append(
            f"{d['rank']:<5} {d['chain_id']:<4} {d['resseq']:<6} {d['resname']:<6} "
            f"{d['sasa']:<10.1f} {d['rel_sasa']:<10.3f} {d['hydrophilicity']:<8.2f} "
            f"{d['protrusion']:<8.3f} {d['combined_score']:<10.4f}"
        )

    # Summary stats
    n_surface = sum(1 for d in epitope_data if d['rel_sasa'] > 0.15)
    top5_str = ', '.join(d['chain_id'] + str(d['resseq']) + '(' + d['resname'] + ')' for d in epitope_data[:5])
    lines.append(sep)
    lines.append(f"总残基数: {len(epitope_data)}  |  表面残基 (relSASA>0.15): {n_surface}  |  Top5: {top5_str}")
    return '\n'.join(lines)


# ── Contact analysis ──

def analyze_contacts(pdb_path, chain_a, chain_b, distance_cutoff=8.0):
    """
    Analyze inter-chain contacts.

    Args:
        pdb_path: path to PDB
        chain_a, chain_b: chain IDs to compare
        distance_cutoff: distance threshold in Angstroms

    Returns:
        (contacts, interface_a, interface_b, summary)
        contacts: list of (res_a_resseq, res_a_name, res_b_resseq, res_b_name, min_dist)
        interface_a: set of resseq in chain_a at interface
        interface_b: set of resseq in chain_b at interface
        summary: dict with contact_count, interface_size_a, interface_size_b
    """
    structure = _parse_structure(pdb_path)

    atoms_a = []  # (resseq, resname, atom_name, coord)
    atoms_b = []
    for model in structure:
        for chain in model:
            for res in chain:
                cid = chain.id
                resseq = res.get_id()[1]
                resname = res.get_resname().strip()
                one = AA_3TO1.get(resname, 'X')
                for atom in res:
                    if atom.element in ('C', 'N', 'O', 'S'):
                        if cid == chain_a:
                            atoms_a.append((resseq, one, atom.name, atom.get_coord()))
                        elif cid == chain_b:
                            atoms_b.append((resseq, one, atom.name, atom.get_coord()))

    if not atoms_a or not atoms_b:
        return [], set(), set(), {'contact_count': 0, 'interface_size_a': 0, 'interface_size_b': 0}

    coords_b = np.array([a[3] for a in atoms_b])
    tree_b = KDTree(coords_b)

    contacts = []
    interface_a, interface_b = set(), set()
    for resseq_a, resname_a, aname, coord_a in atoms_a:
        dist, idx = tree_b.query(coord_a)
        if dist < distance_cutoff:
            resseq_b, resname_b, bname, _ = atoms_b[idx]
            contacts.append((resseq_a, resname_a, resseq_b, resname_b, round(float(dist), 2)))
            interface_a.add(resseq_a)
            interface_b.add(resseq_b)

    # Deduplicate residue pairs (keep shortest distance per pair)
    pair_dists = {}
    for ra, rna, rb, rnb, d in contacts:
        key = (ra, rb)
        if key not in pair_dists or d < pair_dists[key][4]:
            pair_dists[key] = (ra, rna, rb, rnb, d)
    unique_contacts = sorted(pair_dists.values(), key=lambda x: x[4])

    return unique_contacts, interface_a, interface_b, {
        'contact_count': len(unique_contacts),
        'interface_size_a': len(interface_a),
        'interface_size_b': len(interface_b),
    }


def format_contact_summary(contacts, interface_a, interface_b, summary, chain_a_name='A', chain_b_name='B'):
    """Format contact analysis as readable text."""
    lines = [f"接触分析 ({chain_a_name}-{chain_b_name})", "=" * 50,
             f"距离阈值: 8.0Å", f"接触对: {summary['contact_count']}",
             f"界面残基 [{chain_a_name}]: {summary['interface_size_a']}",
             f"界面残基 [{chain_b_name}]: {summary['interface_size_b']}", "",
             f"残基对 (距离<8Å):"]

    for ra, rna, rb, rnb, d in contacts[:30]:
        lines.append(f"  {chain_a_name}:{rna}{ra} -- {chain_b_name}:{rnb}{rb}  ({d}Å)")
    if len(contacts) > 30:
        lines.append(f"  ... 共 {len(contacts)} 对")
    return '\n'.join(lines)


# ── Residue-facing computation ──

def find_residues_facing_region(pdb_path, target_chain, epitope_resseqs,
                                 query_chain, distance_cutoff=10.0):
    """
    Find query-chain residues that face a set of epitope residues.

    Args:
        pdb_path: path to complex PDB (antigen + antibody)
        target_chain: chain ID of the antigen
        epitope_resseqs: list of residue sequence numbers on antigen
        query_chain: chain ID of antibody to check
        distance_cutoff: CA-CA distance threshold in Angstroms

    Returns:
        List of resseq (ints) on query_chain facing the epitope, sorted.
    """
    structure = _parse_structure(pdb_path)
    epitope_set = set(epitope_resseqs)

    # Collect CA atoms
    epitope_cas = []
    query_cas = {}  # resseq -> coord
    for model in structure:
        for chain in model:
            for res in chain:
                if 'CA' not in res:
                    continue
                resseq = res.get_id()[1]
                coord = res['CA'].get_coord()
                if chain.id == target_chain and resseq in epitope_set:
                    epitope_cas.append(coord)
                elif chain.id == query_chain:
                    query_cas[resseq] = coord

    if not epitope_cas or not query_cas:
        return []

    tree = KDTree(epitope_cas)
    facing = []
    for resseq, coord in query_cas.items():
        dist, _ = tree.query(coord)
        if dist < distance_cutoff:
            facing.append(resseq)

    return sorted(facing)


def residues_to_region_spec(chain_id, resseqs):
    """
    Convert a list of residue numbers to a compact region spec string.
    Example: [25,26,27,30,31,35] -> "A:25-27,30-31,35"
    """
    if not resseqs:
        return ""
    sorted_r = sorted(resseqs)
    ranges = []
    start = sorted_r[0]
    end = sorted_r[0]
    for r in sorted_r[1:]:
        if r == end + 1:
            end = r
        else:
            ranges.append((start, end))
            start = end = r
    ranges.append((start, end))

    parts = []
    for s, e in ranges:
        if s == e:
            parts.append(str(s))
        else:
            parts.append(f"{s}-{e}")
    return f"{chain_id}:{','.join(parts)}"


# ── Interface property scoring ──

def score_interface_properties(pdb_path, chain_a, chain_b,
                                interface_a, interface_b):
    """
    Score interface properties: hydrophobicity, charge, H-bonds.

    Returns dict of metrics and a formatted text report.
    """
    structure = _parse_structure(pdb_path)

    # Collect interface residues
    residues_a = {}  # resseq -> resname_1
    residues_b = {}
    for model in structure:
        for chain in model:
            for res in chain:
                if 'CA' not in res:
                    continue
                resseq = res.get_id()[1]
                resname = res.get_resname().strip()
                one = AA_3TO1.get(resname, 'X')
                if chain.id == chain_a and resseq in interface_a:
                    residues_a[resseq] = one
                elif chain.id == chain_b and resseq in interface_b:
                    residues_b[resseq] = one

    def _avg_hydro(residues):
        if not residues:
            return 0.0
        return np.mean([EISENBERG.get(aa, 0.0) for aa in residues.values()])

    def _charge_counts(residues):
        pos = sum(1 for aa in residues.values() if aa in POSITIVE_AA)
        neg = sum(1 for aa in residues.values() if aa in NEGATIVE_AA)
        return pos, neg

    hydro_a = _avg_hydro(residues_a)
    hydro_b = _avg_hydro(residues_b)
    pos_a, neg_a = _charge_counts(residues_a)
    pos_b, neg_b = _charge_counts(residues_b)

    # Charge complementarity: positive in A with negative in B + vice versa
    charge_complement = (pos_a + pos_b + neg_a + neg_b)
    max_charge = len(residues_a) + len(residues_b)
    charge_score = (pos_a * neg_b + pos_b * neg_a) / max(max_charge, 1)

    # H-bond estimate: count interface donor-acceptor pairs
    hbond_donors = {'S', 'T', 'Y', 'N', 'Q', 'H', 'K', 'R', 'W'}
    hbond_candidates_a = sum(1 for aa in residues_a.values() if aa in hbond_donors)
    hbond_candidates_b = sum(1 for aa in residues_b.values() if aa in hbond_donors)
    hbond_potential = hbond_candidates_a + hbond_candidates_b

    return {
        'hydrophobicity_a': round(hydro_a, 3),
        'hydrophobicity_b': round(hydro_b, 3),
        'charge_pos_a': pos_a, 'charge_neg_a': neg_a,
        'charge_pos_b': pos_b, 'charge_neg_b': neg_b,
        'charge_complementarity': round(charge_score, 4),
        'hbond_potential': hbond_potential,
    }


def format_interface_report(metrics, contacts_summary):
    """Format interface metrics as human-readable report."""
    m = metrics
    lines = [
        "界面特性报告",
        "=" * 50,
        f"界面接触残基数: {contacts_summary['interface_size_a']} + {contacts_summary['interface_size_b']}",
        f"接触对 (<8Å): {contacts_summary['contact_count']}",
        "",
        "疏水性 (Eisenberg):",
        f"  链A界面: {m['hydrophobicity_a']:+.3f}  (正值=疏水,负值=亲水)",
        f"  链B界面: {m['hydrophobicity_b']:+.3f}",
        "",
        "电荷分布:",
        f"  链A: +{m['charge_pos_a']} / -{m['charge_neg_a']}",
        f"  链B: +{m['charge_pos_b']} / -{m['charge_neg_b']}",
        f"  电荷互补性: {m['charge_complementarity']:.3f} (0=无互补, 1=完美互补)",
        "",
        f"潜在氢键供体/受体: {m['hbond_potential']}",
    ]

    # Qualitative assessment
    if m['charge_complementarity'] > 0.3:
        lines.append("✓ 界面电荷互补性良好")
    else:
        lines.append("○ 界面电荷互补性一般")
    if m['hydrophobicity_a'] > 0 and m['hydrophobicity_b'] > 0:
        lines.append("○ 界面偏疏水 (可能为疏水驱动结合)")
    elif m['hydrophobicity_a'] < 0 and m['hydrophobicity_b'] < 0:
        lines.append("○ 界面偏亲水 (可能为极性/静电驱动)")

    return '\n'.join(lines)


# ── Helper: parse region spec string ──

def parse_region_spec(region_str):
    """
    Parse region spec like "A:25-35,40-50" into {chain: [residue_numbers]}.
    """
    import re
    regions = {}
    if not region_str or not region_str.strip():
        return regions
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', region_str):
        indices = []
        for seg in spec.split(','):
            seg = seg.strip()
            if not seg:
                continue
            if '-' in seg:
                a, b = seg.split('-')
                indices.extend(range(int(a.strip()), int(b.strip()) + 1))
            else:
                indices.append(int(seg))
        regions[cid] = sorted(set(indices))
    return regions
