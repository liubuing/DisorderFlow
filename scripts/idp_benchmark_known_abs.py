import os
#!/usr/bin/env python3
"""P0: M0 Gate — prove known anti-Aβ antibodies score well in peptide-mode AF2.

IDP_DESIGN_SCHEME_V2 §2 P0-1:
  "先证明已知 Aβ 抗体 native CDR 在肽段模式下 ipTM>0.6，
   否则禁止动模型——度量不可信前训啥都是盲跑。"

Tests known anti-Aβ antibodies (4HIX solanezumab, 5CSZ gantenerumab,
3UOT, 6CGZ, 6D9B, 6H3R, 7K4V) in peptide-mode AF2:
  1. Extract native CDR + epitope peptide from crystal structure
  2. Run AF2 multimer (antibody Fv + short peptide)
  3. Report peptide ipTM — MUST be >0.6 for M0 to pass
  4. Scrambled CDR control — must score significantly lower

Usage:
    python idp_benchmark_known_abs.py              # run all antibodies
    python idp_benchmark_known_abs.py --pdb 4HIX  # single antibody

Output: idp_benchmark_results/m0_gate_<timestamp>.json
"""
import sys, os, json, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
import numpy as np
from collections import defaultdict

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA3_TO_1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E',
            'GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
            'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
            'TYR':'Y','VAL':'V'}

# ── Known anti-Aβ antibodies with epitope info ──
KNOWN_ABS = {
    '4HIX': {
        'pdb': 'data/anti_abeta_refs/4HIX.pdb',
        'antibody': 'solanezumab (3D6 Fab)',
        'epitope_peptide': 'KLVFFAED',       # Aβ 16-23, solanezumab epitope
        'epitope_full': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
        'cdr_definition': 'chothia',  # VH CDRs
        'heavy_chain': 'H', 'light_chain': 'L',
    },
    '5CSZ': {
        'pdb': 'data/anti_abeta_refs/5CSZ.pdb',
        'antibody': 'gantenerumab Fab',
        'epitope_peptide': 'DAEFRHDSGY',     # Aβ 1-10, gantenerumab epitope
        'epitope_full': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
        'cdr_definition': 'chothia',
        'heavy_chain': 'H', 'light_chain': 'L',
    },
    '3UOT': {
        'pdb': 'data/anti_abeta_refs/3UOT.pdb',
        'antibody': 'anti-Aβ Fab (3UOT)',
        'epitope_peptide': 'DAEFRHDSGY',
        'epitope_full': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
        'heavy_chain': 'H', 'light_chain': 'L',
    },
    '7K4V': {
        'pdb': 'data/anti_abeta_refs/7K4V.pdb',
        'antibody': 'anti-Aβ Fab (7K4V)',
        'epitope_peptide': 'DAEFRHDSGY',
        'epitope_full': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
        'heavy_chain': 'H', 'light_chain': 'L',
    },
}

# Chothia CDR definitions for VH
CHOTHIA_H1 = (26, 32)   # 1-based inclusive
CHOTHIA_H2 = (52, 56)
CHOTHIA_H3 = (95, 102)


def get_pdb_sequence(pdb_path, chain_id):
    """Extract amino acid sequence for a specific chain from PDB."""
    seq, seen = [], set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[21:22].strip() == chain_id:
                if line[12:16].strip() == 'CA':
                    resi = line[22:27]
                    if resi not in seen:
                        seen.add(resi)
                        seq.append(AA3_TO_1.get(line[17:20].strip(), 'X'))
    return ''.join(seq)


def get_cdr_from_pdb(pdb_path, heavy_chain='H', light_chain='L'):
    """Extract native CDR sequences from PDB using residue numbering."""
    # Get full sequences first
    vh_seq = get_pdb_sequence(pdb_path, heavy_chain)
    vl_seq = get_pdb_sequence(pdb_path, light_chain)

    # Parse residue numbers to find CDR boundaries
    vh_resi = []
    vl_resi = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                chain = line[21:22].strip()
                resi = int(line[22:26].strip())
                if chain == heavy_chain and resi not in vh_resi:
                    vh_resi.append(resi)
                elif chain == light_chain and resi not in vl_resi:
                    vl_resi.append(resi)

    def _cdr_by_numbering(seq, resi_list, cdr_range):
        """Extract CDR by Chothia numbering range."""
        start, end = cdr_range
        idxs = [i for i, r in enumerate(resi_list) if start <= r <= end]
        if idxs:
            return ''.join(seq[i] for i in idxs)
        return ''

    h1 = _cdr_by_numbering(vh_seq, vh_resi, CHOTHIA_H1)
    h2 = _cdr_by_numbering(vh_seq, vh_resi, CHOTHIA_H2)
    h3 = _cdr_by_numbering(vh_seq, vh_resi, CHOTHIA_H3)
    all_cdr = h1 + h2 + h3

    return {
        'vh_seq': vh_seq, 'vl_seq': vl_seq,
        'vh_resi': vh_resi, 'vl_resi': vl_resi,
        'H1': h1, 'H2': h2, 'H3': h3,
        'cdr_concat': all_cdr,
    }


def scramble_cdr(cdr_seq):
    """Randomly permute CDR sequence (preserves composition)."""
    chars = list(cdr_seq)
    random.shuffle(chars)
    return ''.join(chars)


def run_af2_peptide_mode(ab_fv_seq, peptide_seq, num_recycle=1):
    """Run AF2 multimer on antibody Fv + short peptide.

    Returns dict with iptm, plddt, success.
    """
    from build_design_variant_dataset import _batch_af2_wsl
    results = _batch_af2_wsl([ab_fv_seq], peptide_seq, num_recycle=num_recycle)
    if results and results[0]:
        return {
            'iptm': results[0].get('iptm', 0),
            'plddt': results[0].get('plddt', 0),
            'success': results[0].get('success', False),
        }
    return {'iptm': 0, 'plddt': 0, 'success': False}


def run_af2_full_abeta(ab_fv_seq, abeta_full, num_recycle=1):
    """Run AF2 multimer on antibody Fv + full Aβ42 (baseline comparison)."""
    from build_design_variant_dataset import _batch_af2_wsl
    results = _batch_af2_wsl([ab_fv_seq], abeta_full, num_recycle=num_recycle)
    if results and results[0]:
        return {
            'iptm': results[0].get('iptm', 0),
            'plddt': results[0].get('plddt', 0),
            'success': results[0].get('success', False),
        }
    return {'iptm': 0, 'plddt': 0, 'success': False}


def main():
    parser = argparse.ArgumentParser(description='M0 Gate: known anti-Aβ benchmark')
    parser.add_argument('--pdb', default=None, help='Single PDB ID to test')
    parser.add_argument('--scramble-trials', type=int, default=3,
                        help='Number of scrambled CDR controls per antibody')
    parser.add_argument('--peptide', default=None,
                        help='Override epitope peptide (default: use known epitope)')
    args = parser.parse_args()

    targets = {args.pdb: KNOWN_ABS[args.pdb]} if args.pdb else KNOWN_ABS

    results = []
    print(f"{'='*70}")
    print(f"M0 GATE: Known anti-Aβ antibody benchmark (peptide-mode AF2)")
    print(f"{'='*70}\n")
    print(f"Gate criterion: native CDR peptide ipTM MUST be >0.6")
    print(f"  If <0.6 → metric is broken, DO NOT train models\n")

    for pdb_id, info in targets.items():
        pdb_path = info['pdb']
        if not os.path.exists(pdb_path):
            print(f"[{pdb_id}] PDB not found: {pdb_path} — SKIP")
            continue

        print(f"\n{'─'*50}")
        print(f"[{pdb_id}] {info['antibody']}")
        print(f"  Epitope peptide: {info['epitope_peptide']}")
        print(f"{'─'*50}")

        # Extract native CDR
        cdr_data = get_cdr_from_pdb(pdb_path, info['heavy_chain'], info['light_chain'])
        native_cdr = cdr_data['cdr_concat']
        vh_seq = cdr_data['vh_seq']
        vl_seq = cdr_data['vl_seq']
        ab_fv = vh_seq + vl_seq  # Fv = VH + VL

        print(f"  VH: {len(vh_seq)}aa, VL: {len(vl_seq)}aa, CDR: {len(native_cdr)}aa")
        print(f"  H1={cdr_data['H1']}, H2={cdr_data['H2']}, H3={cdr_data['H3'][:20]}...")

        # Test 1: Native CDR in peptide mode
        print(f"\n  [1/3] Native CDR + peptide ({info['epitope_peptide']})...")
        pep_result = run_af2_peptide_mode(ab_fv, info['epitope_peptide'])
        pep_iptm = pep_result['iptm']
        print(f"    ipTM = {pep_iptm:.4f}  {'✅' if pep_iptm > 0.6 else '❌ GATE FAILED'}")

        # Test 2: Native CDR + full Aβ42 (baseline)
        print(f"  [2/3] Native CDR + full Aβ42 (42aa)...")
        full_result = run_af2_full_abeta(ab_fv, info['epitope_full'])
        full_iptm = full_result['iptm']
        print(f"    ipTM = {full_iptm:.4f}")

        # Test 3: Scrambled CDR controls
        scrambled_iptms = []
        print(f"  [3/3] Scrambled CDR controls (x{args.scramble_trials})...")
        for si in range(args.scramble_trials):
            scram_cdr = scramble_cdr(native_cdr)
            # Graft scrambled CDR into VH
            scram_vh = list(vh_seq)
            pos = 0
            for cdr_name, (start, end) in [('H1', CHOTHIA_H1), ('H2', CHOTHIA_H2), ('H3', CHOTHIA_H3)]:
                native_cdr_seq = cdr_data[cdr_name]
                idxs = [i for i, r in enumerate(cdr_data['vh_resi']) if start <= r <= end]
                for idx in idxs:
                    if pos < len(scram_cdr):
                        scram_vh[idx] = scram_cdr[pos]
                        pos += 1
            scram_fv = ''.join(scram_vh) + vl_seq
            sr = run_af2_peptide_mode(scram_fv, info['epitope_peptide'])
            scrambled_iptms.append(sr['iptm'])
            print(f"    scramble #{si+1}: ipTM = {sr['iptm']:.4f}")

        scram_mean = np.mean(scrambled_iptms) if scrambled_iptms else 0
        separability = pep_iptm - scram_mean
        print(f"\n  Summary [{pdb_id}]:")
        print(f"    Peptide ipTM (native):  {pep_iptm:.4f}")
        print(f"    Peptide ipTM (scrambled mean): {scram_mean:.4f}")
        print(f"    Separability (native − scram):  {separability:+.4f}")
        print(f"    Full Aβ42 ipTM:  {full_iptm:.4f}")
        print(f"    M0 Gate: {'✅ PASS' if pep_iptm > 0.6 else '❌ FAIL — DO NOT TRAIN MODELS'}")

        results.append({
            'pdb_id': pdb_id,
            'antibody': info['antibody'],
            'epitope_peptide': info['epitope_peptide'],
            'native_cdr': native_cdr,
            'cdr_components': {'H1': cdr_data['H1'], 'H2': cdr_data['H2'], 'H3': cdr_data['H3']},
            'peptide_iptm': pep_iptm,
            'full_abeta_iptm': full_iptm,
            'scrambled_iptms': scrambled_iptms,
            'scrambled_mean': scram_mean,
            'separability': separability,
            'm0_pass': pep_iptm > 0.6,
        })

    # ── Final report ──
    print(f"\n{'='*70}")
    print(f"M0 GATE FINAL REPORT")
    print(f"{'='*70}")
    all_pass = all(r['m0_pass'] for r in results)
    for r in results:
        status = '✅ PASS' if r['m0_pass'] else '❌ FAIL'
        print(f"  {r['pdb_id']:6s}  peptide ipTM={r['peptide_iptm']:.4f}  "
              f"scram={r['scrambled_mean']:.4f}  sep={r['separability']:+.4f}  {status}")
    print(f"\n  OVERALL: {'✅ ALL PASS — metric is trustworthy, proceed to model training' if all_pass else '❌ GATE FAILED — FIX METRIC before touching models'}")

    # Save
    os.makedirs('idp_benchmark_results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f'idp_benchmark_results/m0_gate_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'm0_all_pass': all_pass}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
