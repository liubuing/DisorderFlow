#!/usr/bin/env python3
"""V3 Unified Pipeline: L2 FFT pose → L3 score → three-way voting.

Re-ranks P3 designs with V3 physical metrics.
L1: contact+charge+hydro on crystal. L2: FFT pose generation.
L3: interface scoring on docked pose. L4/L5: placeholder for GPU.

Usage: python run_v3_pipeline.py --top 20
"""
import sys, os, json, time, argparse, tempfile
sys.path.insert(0,'.'); sys.path.insert(0,'modules')
import numpy as np

from modules.fft_pose import generate_pose
from modules.interface_scorer import score_pose, three_way_vote

P3_LIBRARY = 'idp_design_results/p3_scale_top100_20260629_030111.json'
AA3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
       'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
       'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}


def graft_and_dock(d, epitope_pdb):
    """Graft CDR into scaffold, dock peptide, score."""
    from Bio.PDB import PDBParser, PDBIO
    sp = d.get('scaffold_path','') or f'data/misfolding_targets/{d["scaffold"]}.pdb'
    parser = PDBParser(QUIET=True); s = parser.get_structure('s', sp)
    chain = d['scaffold_chain']; cdr_spec = d['cdr_spec']
    cdr_ranges = [(int(p.split(':')[1].split('-')[0]), int(p.split(':')[1].split('-')[1]))
                   for p in cdr_spec.split(',')]
    cdr = d['sequence']; pos = 0
    for start, end in cdr_ranges:
        for j in range(start, end+1):
            if pos < len(cdr):
                try: s[0][chain][j].resname = AA3.get(cdr[pos], 'GLY')
                except: pass; pos += 1
    tmp = tempfile.mktemp(suffix='_receptor.pdb')
    io = PDBIO(); io.set_structure(s); io.save(tmp)

    # L2: FFT pose generation
    try:
        pose = generate_pose(tmp, epitope_pdb, chain, 'P')
    except Exception:
        os.unlink(tmp)
        return None

    # L3: score the pose
    cdr_ranges_tuples = [(s,e) for s,e in cdr_ranges]
    result = score_pose(pose, chain, 'P', cdr_ranges_tuples)

    os.unlink(tmp)
    # Windows: file may be briefly locked (AV/indexer); clean best-effort, never fail.
    for _f in (pose,):
        if _f and os.path.exists(_f):
            try: os.unlink(_f)
            except OSError: pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    print("V3 Pipeline: L2 FFT pose → L3 interface score")
    print("=" * 50)

    with open(P3_LIBRARY) as f:
        designs = json.load(f)

    top = designs[:args.top]
    epitope_pdb = 'data/abeta_conformations/pdbs/abeta42_seed0_42.pdb'
    print(f"Receptor: {len(top)} designs, Ligand: {epitope_pdb}")

    results = []
    for i, d in enumerate(top):
        r = graft_and_dock(d, epitope_pdb)
        if r:
            vote = three_way_vote(r['composite'])
            results.append({**d, 'v3_score': r['composite'], 'v3_contacts': r['contacts'],
                            'v3_bsa': r['bsa'], 'v3_vote': vote['reason']})
        if (i+1) % 5 == 0:
            print(f'  {i+1}/{args.top} scored')

    results.sort(key=lambda d: d['v3_score'], reverse=True)

    print(f"\n=== V3 Re-Ranked Top-20 ===")
    for i, d in enumerate(results[:20]):
        print(f"#{i+1:2d} V3={d['v3_score']:.4f} contacts={d['v3_contacts']} "
              f"vote={d['v3_vote']:8s} {d['epitope'][:12]}+{d['scaffold']}")

    # ── Kendall τ: V3 physical score vs old (MPNN+diversity) composite ──
    scored_idx = {id(d): i for i, d in enumerate(results)}
    v3 = [d['v3_score'] for d in results]
    old = [d.get('composite', 0) for d in results]
    try:
        from scipy.stats import kendalltau
        tau, p = kendalltau(v3, old)
        print(f"\n=== Agreement: V3 vs old composite ===")
        print(f"  Kendall tau = {tau:+.4f}  (p={p:.2e})  n={len(v3)}")
        if abs(tau) < 0.3:
            print(f"  => LOW agreement: old composite (MPNN+diversity) does NOT reproduce")
            print(f"     physical interface ranking — confirms old metric was diversity, not quality.")
        else:
            print(f"  => Moderate/high agreement: old composite partly reflects physical quality.")
    except Exception as e:
        print(f"  Kendall tau computation failed: {e}")

    if results:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out = f'idp_design_results/v3_rerank_{ts}.json'
        with open(out, 'w') as f: json.dump(results, f, indent=2)
        print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
