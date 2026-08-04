"""
M0 Gate: Physical Scoring on 5CSZ Crystal Structure
验证: 物理评分在有真实坐标时能否区分 native vs scrambled CDR?
不依赖 AF2 —— 直接用晶体坐标。

Gate PASS = native composite > scrambled composite，有实质差异。
"""
import sys, os, copy, random
sys.path.insert(0, os.path.dirname(__file__))

from modules.interface_scorer import score_pose
from Bio.PDB import PDBParser, PDBIO
import numpy as np

# ── 5CSZ native CDR 序列 ──
CDR_H1 = 'GFTFSSYAMS'    # Chothia 25-35 on Fv
CDR_H2 = 'SAINASGTRTY'   # Chothia 50-60
CDR_H3 = 'GKGYVRYFDV'    # tight H3, after CA before WG

CDR_RANGES = [
    (26, 35),   # H1 (1-based in 5CSZ chain H)
    (52, 61),   # H2
    (95, 104),  # H3
]

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}
AA1TO3 = {v: k for k, v in AA3TO1.items()}
ALL_AA1 = list(AA1TO3.keys())

def get_chain_sequence(structure, chain_id):
    """Extract 1-letter sequence from a PDB chain."""
    seq = []
    for res in structure[0][chain_id]:
        if 'CA' in res:
            rname = res.resname.strip()
            seq.append(AA3TO1.get(rname, 'X'))
    return ''.join(seq)

def scramble_cdr(seq, cdr_ranges):
    """Scramble residues within CDR regions, preserving non-CDR."""
    seq_list = list(seq)
    for start, end in cdr_ranges:
        cdr_len = end - start + 1
        scrambled = random.choices(ALL_AA1, k=cdr_len)
        for i, aa in zip(range(start-1, end), scrambled):
            seq_list[i] = aa
    return ''.join(seq_list)

def graft_sequence_to_pdb(pdb_path, chain_id, new_seq, out_path):
    """Change residue names in a PDB chain to match new sequence."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)

    seq_idx = 0
    for res in structure[0][chain_id]:
        if 'CA' in res:
            if seq_idx < len(new_seq):
                new_resname = AA1TO3.get(new_seq[seq_idx], 'ALA')
                res.resname = new_resname
                seq_idx += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    return seq_idx

def main():
    random.seed(42)

    # ── 1. Score native 5CSZ ──
    pdb_5csz = 'data/anti_abeta_refs/5CSZ.pdb'
    print(f'[M0] Scoring native 5CSZ: {pdb_5csz}')
    print(f'     CDR ranges (chain H): {CDR_RANGES}')
    print(f'     Native CDRs: H1={CDR_H1}, H2={CDR_H2}, H3={CDR_H3}')

    native_result = score_pose(
        pdb_5csz,
        rec_chain='H',
        lig_chain='D',   # Aβ42 peptide chain
        cdr_ranges=CDR_RANGES
    )

    print(f'\n  Native score:')
    for k, v in native_result.items():
        if isinstance(v, float):
            print(f'    {k}: {v:.4f}')
        else:
            print(f'    {k}: {v}')

    # ── 2. Extract native sequence and create scrambled variants ──
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('5csz', pdb_5csz)
    native_seq = get_chain_sequence(structure, 'H')
    print(f'\n  5CSZ chain H length: {len(native_seq)} aa')

    # Verify CDR extraction
    for name, (s, e) in zip(['H1','H2','H3'], CDR_RANGES):
        extracted = native_seq[s-1:e]
        print(f'    {name} ({s}-{e}): {extracted}')

    n_scrambled = 10
    scrambled_results = []

    print(f'\n[M0] Testing {n_scrambled} scrambled CDR variants...')

    for i in range(n_scrambled):
        scrambled_seq = scramble_cdr(native_seq, CDR_RANGES)
        out_pdb = f'_m0_scrambled_{i}.pdb'
        graft_sequence_to_pdb(pdb_5csz, 'H', scrambled_seq, out_pdb)

        result = score_pose(
            out_pdb,
            rec_chain='H',
            lig_chain='D',
            cdr_ranges=CDR_RANGES
        )
        scrambled_results.append(result)
        print(f'  Scrambled {i}: composite={result["composite"]:.4f}, '
              f'contacts={result["contacts"]}, e_contact={result["e_contact"]:.4f}, '
              f'e_elec={result["e_elec"]:.4f}')

        # Cleanup temp file
        os.remove(out_pdb)

    # ── 3. Statistical comparison ──
    scrambled_composites = [r['composite'] for r in scrambled_results]
    scrambled_contacts = [r['contacts'] for r in scrambled_results]

    print(f'\n{"="*60}')
    print(f'[M0 GATE RESULT]')
    print(f'{"="*60}')
    print(f'  Native composite:        {native_result["composite"]:.4f}')
    print(f'  Scrambled mean ± std:    {np.mean(scrambled_composites):.4f} ± {np.std(scrambled_composites):.4f}')
    print(f'  Scrambled range:         [{np.min(scrambled_composites):.4f}, {np.max(scrambled_composites):.4f}]')
    print(f'  Native contacts:         {native_result["contacts"]}')
    print(f'  Scrambled contacts mean: {np.mean(scrambled_contacts):.1f} ± {np.std(scrambled_contacts):.1f}')

    sep = native_result['composite'] - np.mean(scrambled_composites)
    z_score = sep / max(np.std(scrambled_composites), 0.0001)

    print(f'\n  Separation (native - mean scrambled): {sep:.4f}')
    print(f'  Z-score: {z_score:.1f}')

    # Detailed breakdown
    print(f'\n  --- Per-metric breakdown ---')
    for metric in ['e_contact', 'e_bsa', 'e_elec', 'e_hydro', 'e_shape']:
        native_val = native_result.get(metric, 0)
        scram_vals = [r.get(metric, 0) for r in scrambled_results]
        print(f'  {metric:12s}: native={native_val:.4f}, scrambled={np.mean(scram_vals):.4f} ± {np.std(scram_vals):.4f}')

    # ── 4. Verdict ──
    if native_result.get('invalid', False):
        print(f'\n  ❌ GATE FAIL: Native 5CSZ score is INVALID (contacts<5)')
        print(f'     Problem is in the scoring, not the designs.')
    elif sep > 0.05 and z_score > 2.0:
        print(f'\n  ✅ GATE PASS: Physical scoring distinguishes native from scrambled')
        print(f'     Separation={sep:.4f}, z={z_score:.1f}')
        print(f'     → L3 scoring is trustable when real coordinates are available.')
    elif sep > 0.01:
        print(f'\n  ⚠️  GATE MARGINAL: Weak separation ({sep:.4f}, z={z_score:.1f})')
        print(f'     → Scoring works but needs refinement or ensemble averaging.')
    else:
        print(f'\n  ❌ GATE FAIL: No meaningful separation ({sep:.4f}, z={z_score:.1f})')
        print(f'     → Physical scoring cannot distinguish CDR quality even with crystal coords.')

    return native_result, scrambled_results

if __name__ == '__main__':
    main()
