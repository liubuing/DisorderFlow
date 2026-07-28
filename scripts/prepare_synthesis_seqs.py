#!/usr/bin/env python3
"""S0 Sequence Completion — 25 wet-lab candidates → synthesis-ready scFv sequences.

For each candidate:
- Identify framework source (PDB or scaffold)
- Graft CDR-H3 into framework (or use full VH for template-seeded)
- Build scFv: VH-(G4S)3-VL-HHHHHH
- Codon-optimize DNA
- Output FASTA + annotation TSV
"""

import json, os, sys, time
from collections import OrderedDict

# ── Config ──
OUTDIR = 'idp_design_results/s5_wetlab_synthesis_20260705'
LINKER = 'GGGGSGGGGSGGGGS'       # (G4S)3
HIS_TAG = 'HHHHHH'
SIGNAL_PEPTIDE = 'MGWSCIILFLVATATGVHS'  # murine Ig kappa leader (secreted expression)

# Standard codons for E.coli K12 (for simplicity, use E.coli as default synthesis host)
ECOLI_CODONS = {
    'A': 'GCG', 'R': 'CGT', 'N': 'AAC', 'D': 'GAT', 'C': 'TGC',
    'Q': 'CAG', 'E': 'GAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCG',
    'S': 'AGC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTG',
    '*': 'TAA'
}


# ── PDB-based framework extraction ──
def extract_chain_sequence(pdb_path, chain_id):
    """Extract amino acid sequence from a PDB chain using BioPython."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    three_to_one = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
        'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
        'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
        'MSE':'M','ASX':'N','GLX':'Q','UNK':'X',
    }
    try:
        s = parser.get_structure('pdb', pdb_path)
        seq = []
        for res in s[0][chain_id].get_residues():
            if res.id[0] != ' ': continue  # skip heteroatoms
            rn = res.get_resname().strip()
            seq.append(three_to_one.get(rn, 'X'))
        return ''.join(seq)
    except Exception as e:
        return None


def get_framework_from_pdb(pdb_id, vh_chain, vl_chain):
    """Get VH and VL sequences from a PDB file with H/L chain IDs."""
    pdb_path = f'data/anti_abeta_refs/{pdb_id}.pdb'
    if not os.path.exists(pdb_path):
        return None, None
    vh = extract_chain_sequence(pdb_path, vh_chain)
    vl = extract_chain_sequence(pdb_path, vl_chain)
    return vh, vl


# ── Code for handling ANARCI-less antibody domain identification ──
# In VH, look for characteristic patterns to identify domain boundaries
VH_PATTERNS = ['VQLVES', 'VQLVQS', 'VQLQES', 'VKLVES', 'VKLEES', 'EVQLV', 'QVQLV', 'EVKLV',
                'QVQLQ', 'EVQLQ', 'QVQLK', 'QVQLE', 'EVQLL', 'VQLQQ', 'VQLQE',
                'VQLLES', 'VELVES', 'VQLVET', 'VKLVET', 'VQLQET',
                'GGGLVQ', 'GGGLVK', 'GGGLVE']
VL_PATTERNS = ['DIVMT', 'DIQMT', 'DVVMT', 'EIVLT', 'QSVLT', 'DIELT', 'QIVLT',
                'DIVLT', 'DIQLT', 'DVVLT', 'EIVMT', 'QIVLT', 'QSVLT', 'DIALT',
                'YVVMT', 'YIVMT', 'YELTQ', 'YVLTQ',
                'SPLSLP', 'SPASIS', 'SPATLS', 'SPGEPAS']


def find_vh_domain(sequence):
    """Heuristically find VH domain (~120 aa) in a longer sequence."""
    for pat in VH_PATTERNS:
        idx = sequence.find(pat)
        if idx >= 0:
            # VH domain typically 115-130 aa from this start
            end = min(len(sequence), idx + 130)
            return sequence[idx:end]
    # Fallback: return first ~125 aa
    return sequence[:125] if len(sequence) > 125 else sequence


def find_vl_domain(sequence):
    """Heuristically find VL domain (~110 aa) in a longer sequence."""
    for pat in VL_PATTERNS:
        idx = sequence.find(pat)
        if idx >= 0:
            end = min(len(sequence), idx + 115)
            return sequence[idx:end]
    return sequence[:115] if len(sequence) > 115 else sequence


# ── Codon optimization ──
def codon_optimize(aa_seq, host='ecoli'):
    """Simple codon optimization for a given host."""
    if host == 'ecoli':
        codons = ECOLI_CODONS
    else:
        codons = ECOLI_CODONS  # fallback
    return ''.join(codons.get(aa, 'NNN') for aa in aa_seq)


# ── MW / pI estimation ──
def estimate_mw(aa_seq):
    """Estimate molecular weight (Da) from amino acid composition."""
    aa_mw = {
        'A':89.1,'R':174.2,'N':132.1,'D':133.1,'C':121.2,'Q':146.2,'E':147.1,
        'G':75.1,'H':155.2,'I':131.2,'L':131.2,'K':146.2,'M':149.2,'F':165.2,
        'P':115.1,'S':105.1,'T':119.1,'W':204.2,'Y':181.2,'V':117.1
    }
    mw = sum(aa_mw.get(aa, 110.0) for aa in aa_seq)
    return mw - 18.0 * (len(aa_seq) - 1)  # subtract water for peptide bonds


def estimate_pi(aa_seq):
    """Simple pI estimate based on charged residues."""
    n_pos = aa_seq.count('R') + aa_seq.count('K') + aa_seq.count('H')
    n_neg = aa_seq.count('D') + aa_seq.count('E')
    # Crude approximation
    if n_pos + n_neg == 0:
        return 7.0
    return round(7.0 + 3.0 * (n_pos - n_neg) / (n_pos + n_neg), 1)


def estimate_ext_coeff(aa_seq):
    """Estimate extinction coefficient (M-1 cm-1) at 280 nm."""
    n_w = aa_seq.count('W')
    n_y = aa_seq.count('Y')
    n_c = aa_seq.count('C')
    return 5500 * n_w + 1490 * n_y + 125 * n_c


# ── CDR-H3 grafting ──
def graft_cdr_h3(framework_vh, new_cdr_h3):
    """Graft a new CDR-H3 into a VH framework sequence.

    Uses conserved motifs to locate CDR-H3 boundaries:
    - FR3 end: YYC/YFC/YHC (Chothia H92-H94)
    - CDR-H3: YYC+3 to WG-1 (where WG = FR4 start, H103)
    - After CXX → CDR-H3 starts; before WGXG → CDR-H3 ends

    Returns: (grafted_vh, success_bool)
    """
    import re

    # Find FR3 end: Y[YF]C motif (position H92-H94)
    fw = framework_vh
    if len(fw) < 100:
        return fw, False  # not a VH domain

    # Locate the conserved Cys at H92 (in Chothia) — YYC, YFC, YHC, FHC
    # Search in the middle portion of VH (residues ~70-100 in IMGT)
    fr3_match = re.search(r'[YF][YFH]C', fw[80:110])
    if not fr3_match:
        # Fallback: search broader
        fr3_match = re.search(r'[YF][YFH]C', fw[70:120])

    if not fr3_match:
        return fw, False

    # CDR-H3 starts at Y[CYF]C + 3 (after the CAR/CAK/CTR prefix)
    fr3_pos = 80 + fr3_match.start() if fw[80:110].find(fr3_match.group()) >= 0 else 70 + fr3_match.start()
    cdr_start = fr3_pos + 3  # YYC → AR → CDR-H3 starts here

    # Find FR4 start: WGXG motif (WGQG, WGKG, WGRG)
    # Search in last 25 residues
    wg_match = re.search(r'WG', fw[cdr_start:cdr_start + 30])
    if not wg_match:
        # Try broader: search in last 30 residues of VH
        wg_match = re.search(r'WG', fw[-30:])
        if wg_match:
            cdr_end = len(fw) - 30 + wg_match.start()
        else:
            return fw, False
    else:
        cdr_end = cdr_start + wg_match.start()

    # Graft: FR1_to_FR3 + new_CDR_H3 + FR4
    grafted = fw[:cdr_start] + new_cdr_h3 + fw[cdr_end:]
    return grafted, True


# ── Main pipeline ──
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load candidates
    data = json.load(open('idp_design_results/s5_wetlab_candidates.json', encoding='utf-8'))
    candidates = data['top_candidates']
    print(f"Loaded {len(candidates)} candidates")

    # ── Step 1: Extract framework sequences from PDBs ──
    print("\n── Extracting PDB frameworks ──")

    # 4HIX: chains H (VH), L (VL), A (Aβ antigen)
    fw_4HIX_vh, fw_4HIX_vl = get_framework_from_pdb('4HIX', 'H', 'L')

    # 5CSZ: chains H (VH), L (VL)
    fw_5CSZ_vh, fw_5CSZ_vl = get_framework_from_pdb('5CSZ', 'H', 'L')

    # 3UOT: scFv structure (chains A/B), D/E are antigen
    fw_3UOT_a = extract_chain_sequence('data/anti_abeta_refs/3UOT.pdb', 'A')
    fw_3UOT_b = extract_chain_sequence('data/anti_abeta_refs/3UOT.pdb', 'B')

    # 6CGZ: chains A/B/C — large, need domain extraction
    fw_6CGZ_a = extract_chain_sequence('data/anti_abeta_refs/6CGZ.pdb', 'A')
    fw_6CGZ_b = extract_chain_sequence('data/anti_abeta_refs/6CGZ.pdb', 'B')

    # 6H3R: chains A(127), B(128) ≈ scFv, C(19), D(18) ≈ antigen
    fw_6H3R_a = extract_chain_sequence('data/anti_abeta_refs/6H3R.pdb', 'A')
    fw_6H3R_b = extract_chain_sequence('data/anti_abeta_refs/6H3R.pdb', 'B')

    # 6H1F: chains A(183), B(158) — anti-αSyn Fab/scFv
    fw_6H1F_a = extract_chain_sequence('data/anti_abeta_refs/6H1F.pdb', 'A')
    fw_6H1F_b = extract_chain_sequence('data/anti_abeta_refs/6H1F.pdb', 'B')

    # 6CBV: chains H(386), L(348), B(134) — anti-tau Fab
    fw_6CBV_h, fw_6CBV_l = get_framework_from_pdb('6CBV', 'H', 'L')

    # 4BKL: chains A(216), B(216), E/F/G(25-26) — anti-αSyn
    fw_4BKL_a = extract_chain_sequence('data/anti_abeta_refs/4BKL.pdb', 'A')
    fw_4BKL_b = extract_chain_sequence('data/anti_abeta_refs/4BKL.pdb', 'B')

    print(f"  4HIX: VH={len(fw_4HIX_vh) if fw_4HIX_vh else 0}aa, VL={len(fw_4HIX_vl) if fw_4HIX_vl else 0}aa")
    print(f"  5CSZ: VH={len(fw_5CSZ_vh) if fw_5CSZ_vh else 0}aa, VL={len(fw_5CSZ_vl) if fw_5CSZ_vl else 0}aa")
    print(f"  3UOT: A={len(fw_3UOT_a) if fw_3UOT_a else 0}aa, B={len(fw_3UOT_b) if fw_3UOT_b else 0}aa")
    print(f"  6CGZ: A={len(fw_6CGZ_a) if fw_6CGZ_a else 0}aa, B={len(fw_6CGZ_b) if fw_6CGZ_b else 0}aa")
    print(f"  6H3R: A={len(fw_6H3R_a) if fw_6H3R_a else 0}aa, B={len(fw_6H3R_b) if fw_6H3R_b else 0}aa")
    print(f"  6H1F: A={len(fw_6H1F_a) if fw_6H1F_a else 0}aa, B={len(fw_6H1F_b) if fw_6H1F_b else 0}aa")
    print(f"  6CBV: H={len(fw_6CBV_h) if fw_6CBV_h else 0}aa, L={len(fw_6CBV_l) if fw_6CBV_l else 0}aa")
    print(f"  4BKL: A={len(fw_4BKL_a) if fw_4BKL_a else 0}aa, B={len(fw_4BKL_b) if fw_4BKL_b else 0}aa")

    # ── Step 2: Map each candidate to its framework ──
    print("\n── Mapping candidates to frameworks ──")

    results = []
    unpaired = []

    # Framework assignment logic
    # Literature → PDB mapping:
    #   Solanezumab (anti-Aβ mid) → 4HIX
    #   Donanemab (anti-Aβ N-term, pGlu3) → 4HIX
    #   Crenezumab (anti-Aβ mid) → 5CSZ
    #   Lecanemab (anti-Aβ protofibril) → 5CSZ (humanized, 5CSZ is murine but same class)
    #   Gantenerumab (anti-Aβ N-term) → 6H3R
    #   Bapineuzumab (anti-Aβ N-term) → 4HIX
    #   Ponezumab (anti-Aβ C-term) → 4HIX

    LITERATURE_FRAMEWORKS = {
        'Solanezumab':           ('4HIX', fw_4HIX_vh, fw_4HIX_vl),
        'Donanemab':             ('4HIX', fw_4HIX_vh, fw_4HIX_vl),
        'Crenezumab':            ('5CSZ', fw_5CSZ_vh, fw_5CSZ_vl),
        'Lecanemab (BAN2401)':   ('5CSZ', fw_5CSZ_vh, fw_5CSZ_vl),
        'Gantenerumab':          ('4HIX', fw_4HIX_vh, fw_4HIX_vl),  # 6H3R is NOT Gantenerumab's PDB, use 4HIX
        'Bapineuzumab':          ('4HIX', fw_4HIX_vh, fw_4HIX_vl),
        'Ponezumab':             ('4HIX', fw_4HIX_vh, fw_4HIX_vl),
    }

    PDB_FRAMEWORKS = {
        # 6CGZ is NOT an antibody structure (bacterial enzyme) — reassign to 4HIX
        'PDB 6CGZ':  ('4HIX', fw_4HIX_vh, fw_4HIX_vl, 'full_vh', 'full_vl'),
        'PDB 5CSZ':  ('5CSZ', fw_5CSZ_vh, fw_5CSZ_vl, 'full_vh', 'full_vl'),
        'PDB 4HIX':  ('4HIX', fw_4HIX_vh, fw_4HIX_vl, 'full_vh', 'full_vl'),
        'PDB 3UOT':  ('3UOT', fw_3UOT_a, fw_3UOT_b, 'scfv_a', 'scfv_b'),
        'PDB 6H1F':  ('6H1F', fw_6H1F_a, fw_6H1F_b, 'vh_chain', 'non_vl_fallback'),
        'PDB 6CBV':  ('6CBV', fw_6CBV_h, fw_6CBV_l, 'full_vh', 'full_vl'),
        'PDB 4BKL':  ('4BKL', fw_4BKL_a, fw_4BKL_b, 'vh_216aa', 'vl_216aa'),
    }

    TEMPLATE_FRAMEWORKS = {
        'PillarA_4HIX': ('4HIX', fw_4HIX_vh, fw_4HIX_vl),
        'PillarA_5CSZ': ('5CSZ', fw_5CSZ_vh, fw_5CSZ_vl),
    }

    # Cross-IDP: use native framework, no CDR graft
    CROSS_IDP = {'Anti-αSyn Fab', 'Anti-αSyn scFv', 'Anti-tau Fab'}

    for i, c in enumerate(candidates):
        src = c['source_antibody']
        stype = c['source_type']
        cdr_seq = c['cdr_sequence']
        cdr_len = c['cdr_length']

        vl_scfv_pdb = None  # reset per-candidate

        rec = {
            'rank': i + 1,
            'source': src,
            'source_type': stype,
            'cdr_h3': cdr_seq,
            'cdr_length': cdr_len,
            'epitope': c['epitope'],
            'epitope_sequence': c['epitope_sequence'],
            'epitope_disorder': c['epitope_disorder'],
            'composite_score': c['composite_score'],
        }

        fw_vh = None
        fw_vl = None
        fw_source = 'unknown'
        strategy = 'unknown'

        # ── Case 1: Template-seeded (full VH already designed) ──
        if stype == 'template_seeded':
            fw_info = TEMPLATE_FRAMEWORKS.get(src)
            if fw_info:
                fw_source = fw_info[0]
                # Candidate's CDR sequence IS the full VH domain
                fw_vh = cdr_seq  # 80aa full VH
                fw_vl = fw_info[2]  # VL from source scaffold
                strategy = 'template_seeded: VH=candidate_full_seq, VL=source_scaffold'
            else:
                rec['status'] = 'UNPAIRED'
                rec['reason'] = f'Template {src} not in framework map'
                unpaired.append(rec)
                continue

        # ── Case 2: Literature antibody (CDR-H3 only, need framework) ──
        elif stype == 'literature':
            fw_info = LITERATURE_FRAMEWORKS.get(src)
            if fw_info:
                fw_source = fw_info[0]
                fw_vh = fw_info[1]
                fw_vl = fw_info[2]
                strategy = f'literature: CDR-H3 graft into {fw_source} VH framework'
            else:
                # Fallback to 4HIX
                fw_source = '4HIX (fallback)'
                fw_vh = fw_4HIX_vh
                fw_vl = fw_4HIX_vl
                strategy = f'literature: CDR-H3 graft into 4HIX (fallback, no specific framework found)'

        # ── Case 3: PDB structure ──
        elif stype.startswith('PDB '):
            fw_info = PDB_FRAMEWORKS.get(stype)
            if fw_info:
                fw_source = fw_info[0]
                fw_vh = fw_info[1]
                fw_vl = fw_info[2]
                domain_type = fw_info[3]
                strategy = f'PDB: {stype} direct extraction (domain={domain_type})'
            else:
                rec['status'] = 'UNPAIRED'
                rec['reason'] = f'PDB {stype} not in framework map'
                unpaired.append(rec)
                continue

        else:
            rec['status'] = 'UNPAIRED'
            rec['reason'] = f'Unknown source_type: {stype}'
            unpaired.append(rec)
            continue

        # ── Check framework validity ──
        if fw_vh is None or fw_vl is None:
            rec['status'] = 'UNPAIRED'
            rec['reason'] = f'Framework VH or VL is None (source={fw_source})'
            unpaired.append(rec)
            continue

        # ── Build scFv construct ──
        if strategy.startswith('template_seeded'):
            # VH = candidate's full sequence (80aa, already a full VH domain from MPNN)
            vh_scfv = fw_vh
            rec['cdr_graft_note'] = 'Full VH domain from MPNN redesign (Pillar A). No CDR graft needed.'

        elif strategy.startswith('literature'):
            # Extract VH domain and graft candidate's CDR-H3
            vh_domain = find_vh_domain(fw_vh)
            grafted_vh, graft_ok = graft_cdr_h3(vh_domain, cdr_seq)
            if graft_ok:
                vh_scfv = grafted_vh
                rec['cdr_graft_note'] = f'CDR-H3 ({cdr_len}aa) grafted into {fw_source} VH framework. Graft boundaries auto-detected via YYC/WG motifs.'
            else:
                vh_scfv = vh_domain
                rec['cdr_graft_note'] = f'WARNING: CDR-H3 grafting failed (no YYC/WG motifs in {fw_source} VH). Using framework VH as-is. Manual CDR-H3 grafting REQUIRED.'

        elif strategy.startswith('PDB'):
            # PDB candidates: use native framework as-is
            if src in CROSS_IDP:
                # Cross-IDP: use native VH/VL as-is (no CDR graft)
                if fw_source == '6H1F':
                    # Chain A (127aa) = VH, Chain B (102aa) = NOT a VL domain
                    # Use VH from 6H1F, but VL from 4HIX as fallback
                    vh_scfv = fw_vh  # 127aa VH domain
                    vl_scfv_pdb = find_vl_domain(fw_4HIX_vl)  # fallback VL
                    rec['cdr_graft_note'] = 'Cross-IDP (6H1F): VH from 6H1F (camelid/single-domain?). VL from 4HIX fallback (6H1F chain B is not VL).'
                elif fw_source == '6CBV':
                    vh_scfv = find_vh_domain(fw_vh)  # H chain 227aa → extract VH
                    vl_scfv_pdb = find_vl_domain(fw_vl)  # L chain 213aa → extract VL
                    rec['cdr_graft_note'] = 'Cross-IDP (6CBV anti-tau): native VH/VL domains extracted.'
                elif fw_source == '4BKL':
                    vh_scfv = fw_vh[:130] if len(fw_vh) > 130 else fw_vh  # chain A 216aa, extract domain
                    vl_scfv_pdb = fw_vl[:115] if len(fw_vl) > 115 else fw_vl  # chain B 216aa, extract domain
                    rec['cdr_graft_note'] = 'Cross-IDP (4BKL anti-αSyn): native VH/VL domains extracted.'
                else:
                    vh_scfv = find_vh_domain(fw_vh)
                    vl_scfv_pdb = find_vl_domain(fw_vl)
                    rec['cdr_graft_note'] = f'Cross-IDP candidate: native VH/VL from {fw_source} (no CDR graft). Tests cross-reactivity to Abeta42.'
            elif fw_source in ('4HIX', '5CSZ'):
                vh_scfv = fw_vh[:130] if len(fw_vh) > 130 else fw_vh  # VH domain from H chain
                rec['cdr_graft_note'] = f'PDB {fw_source} VH domain as-is ({len(vh_scfv)}aa).'
            elif fw_source in ('3UOT', '6H3R'):
                vh_scfv = fw_vh  # Already domain-sized
                rec['cdr_graft_note'] = f'PDB {fw_source} scFv domain as-is ({len(vh_scfv)}aa).'
            elif fw_source == '6CGZ':
                vh_scfv = find_vh_domain(fw_vh)  # Large chain, extract VH
                rec['cdr_graft_note'] = f'PDB 6CGZ VH domain extracted ({len(vh_scfv)}aa).'
            else:
                vh_scfv = find_vh_domain(fw_vh)
                rec['cdr_graft_note'] = f'PDB {fw_source} VH domain ({len(vh_scfv)}aa).'

        else:
            vh_scfv = fw_vh[:130] if len(fw_vh) > 130 else fw_vh
            rec['cdr_graft_note'] = 'Unknown source type, using framework as-is.'

        # Extract VL domain
        # vl_scfv_pdb may have been set above (for cross-IDP candidates)
        if vl_scfv_pdb is not None:
            vl_scfv = vl_scfv_pdb
        elif fw_source in ('4HIX', '5CSZ', '6CBV'):
            vl_scfv = fw_vl[:115] if len(fw_vl) > 115 else fw_vl
        elif fw_source in ('3UOT', '6H3R'):
            vl_scfv = fw_vl  # Already domain-sized
        elif fw_source in ('6CGZ', '4BKL', '6H1F'):
            vl_scfv = find_vl_domain(fw_vl)
        else:
            vl_scfv = find_vl_domain(fw_vl)

        # Build final scFv (no signal peptide in the synthesized product — add at expression vector level)
        scfv = vh_scfv + LINKER + vl_scfv + HIS_TAG

        # Build with signal peptide (for reference)
        scfv_with_sp = SIGNAL_PEPTIDE + vh_scfv + LINKER + vl_scfv + HIS_TAG

        # Codon-optimize DNA
        dna_seq = codon_optimize(scfv, 'ecoli')

        # Estimate properties
        mw = estimate_mw(scfv)
        pi = estimate_pi(scfv)
        ext = estimate_ext_coeff(scfv)

        # Store
        rec['vh_seq'] = vh_scfv
        rec['vl_seq'] = vl_scfv
        rec['scfv_seq'] = scfv
        rec['scfv_seq_with_sp'] = scfv_with_sp
        rec['dna_seq'] = dna_seq
        rec['framework_source'] = fw_source
        rec['strategy'] = strategy
        rec['mw_kda'] = round(mw / 1000, 1)
        rec['pi'] = pi
        rec['ext_coeff'] = ext
        rec['scfv_length'] = len(scfv)
        rec['status'] = 'COMPLETED'
        results.append(rec)

        fw_tag = fw_source
        print(f"  {i+1:2d}. {src:<25s} → {fw_tag:<8s} | scFv={len(scfv)}aa, {mw/1000:.1f}kDa, pI={pi} | {strategy[:50]}...")

    # ── Step 3: Write outputs ──
    print(f"\n── Writing outputs ({len(results)} completed, {len(unpaired)} unpaired) ──")

    ts = time.strftime('%Y%m%d_%H%M%S')

    # 3a: AA FASTA
    aa_fasta = os.path.join(OUTDIR, 'candidates_aa.fa')
    with open(aa_fasta, 'w', encoding='utf-8') as f:
        for r in results:
            header = f">candidate_{r['rank']:02d} | {r['source']} | {r['epitope']} | {r['framework_source']} | scFv={r['scfv_length']}aa {r['mw_kda']}kDa pI={r['pi']}"
            f.write(f"{header}\n")
            # Write 60 aa per line
            seq = r['scfv_seq']
            for j in range(0, len(seq), 60):
                f.write(seq[j:j+60] + '\n')
    print(f"  {aa_fasta}")

    # 3b: DNA FASTA
    dna_fasta = os.path.join(OUTDIR, 'candidates_dna.fa')
    with open(dna_fasta, 'w', encoding='utf-8') as f:
        for r in results:
            header = f">candidate_{r['rank']:02d}_DNA | {r['source']} | E.coli codon optimized | {len(r['dna_seq'])}bp"
            f.write(f"{header}\n")
            dna = r['dna_seq']
            for j in range(0, len(dna), 60):
                f.write(dna[j:j+60] + '\n')
    print(f"  {dna_fasta}")

    # 3c: Fab FASTA (VH + VL separately)
    fab_fasta = os.path.join(OUTDIR, 'candidates_fab.fa')
    with open(fab_fasta, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f">candidate_{r['rank']:02d}_VH | {r['source']} | {r['epitope']}\n")
            vh = r['vh_seq']
            for j in range(0, len(vh), 60):
                f.write(vh[j:j+60] + '\n')
            f.write(f">candidate_{r['rank']:02d}_VL | {r['source']} | {r['epitope']}\n")
            vl = r['vl_seq']
            for j in range(0, len(vl), 60):
                f.write(vl[j:j+60] + '\n')
    print(f"  {fab_fasta}")

    # 3d: Annotation TSV
    ann_tsv = os.path.join(OUTDIR, 'candidates_annotation.tsv')
    with open(ann_tsv, 'w', encoding='utf-8') as f:
        cols = ['rank', 'source', 'source_type', 'epitope', 'epitope_sequence',
                'epitope_disorder', 'cdr_h3', 'cdr_length',
                'framework_source', 'strategy', 'scfv_length', 'mw_kda', 'pi',
                'ext_coeff', 'vh_length', 'vl_length', 'composite_score', 'status',
                'cdr_graft_note']
        f.write('\t'.join(cols) + '\n')
        for r in results:
            vals = [
                str(r.get(c, '')) for c in cols[:-1]
            ]
            vals.append(r.get('cdr_graft_note', ''))
            f.write('\t'.join(vals) + '\n')
    print(f"  {ann_tsv}")

    # 3e: Unpaired candidates
    if unpaired:
        unp_tsv = os.path.join(OUTDIR, 'unpaired_candidates.tsv')
        with open(unp_tsv, 'w', encoding='utf-8') as f:
            f.write('rank\tsource\tsource_type\treason\n')
            for r in unpaired:
                f.write(f"{r['rank']}\t{r['source']}\t{r['source_type']}\t{r.get('reason','')}\n")
        print(f"  {unp_tsv} ({len(unpaired)} candidates)")

    # 3f: Summary JSON
    summary_json = os.path.join(OUTDIR, 'synthesis_summary.json')
    json.dump({
        'plan': 'WETLAB_VALIDATION_PLAN_V15',
        'step': 'S0 sequence completion',
        'date': ts,
        'total_candidates': len(candidates),
        'completed': len(results),
        'unpaired': len(unpaired),
        'scfv_format': 'VH-(G4S)3-VL-HHHHHH',
        'signal_peptide': SIGNAL_PEPTIDE,
        'codon_host': 'E.coli K12',
        'note': 'For literature antibodies, framework VH/VL from closest PDB used. CDR-H3 sequences are from candidate data (literature/patent origin). Manual verification of CDR-H3 grafting recommended before synthesis.',
    }, open(summary_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"  {summary_json}")

    # ── Step 4: Summary ──
    print(f"\n{'='*60}")
    print(f"S0 Sequence Completion — DONE")
    print(f"{'='*60}")
    print(f"  Completed: {len(results)}/{len(candidates)}")
    print(f"  Unpaired:  {len(unpaired)}/{len(candidates)}")
    print(f"  Output dir: {OUTDIR}/")
    print(f"    - candidates_aa.fa    ({len(results)} scFv AA sequences)")
    print(f"    - candidates_dna.fa   ({len(results)} codon-optimized DNA)")
    print(f"    - candidates_fab.fa   ({len(results)} VH+VL separately)")
    print(f"    - candidates_annotation.tsv")
    print(f"    - synthesis_summary.json")

    if unpaired:
        print(f"\n  ⚠ Unpaired candidates (need manual lookup):")
        for r in unpaired:
            print(f"    {r['rank']}. {r['source']} ({r['source_type']}): {r.get('reason','')}")

    return results, unpaired


if __name__ == '__main__':
    main()
