import os
#!/usr/bin/env python3
"""Architecture diagnostic: can the BFN sequence decoder RECOVER real anti-Aβ CDRs?

Known anti-Aβ antibodies exist (5CSZ = aducanumab-like Fab + Aβ DAEFRHDSGYE).
If we mask their CDRs, hand BFN the antibody framework + Aβ antigen (complex
mode), and let it DESIGN, does it recover the real binding CDR sequence?

  High recovery  → BFN architecture CAN learn Aβ binding given the right input;
                  the binding failure is a DATA problem (reject-sampling on real
                  binding labels will work).
  Low / random   → BFN architecture itself can't produce Aβ-binding CDRs from
                  this framework+antigen context; needs fundamental change.

Free: BFN inference only, no AF2. Uses V15 (seq_only, well-trained decoder) —
no co-design so the sequence pathway is the only variable.
"""
import sys, os, re, json
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import numpy as np, torch, yaml

# 5CSZ: heavy H (Fab ~213aa incl constant CH1), light L. Antigen Aβ chain D.
# We take only the Fv (~first 120aa of H) so Chothia CDR ranges (26-102) apply.
SCAFFOLD_PDB = 'data/anti_abeta_refs/5CSZ.pdb'
SCAFFOLD_CHAIN = 'H'
FV_LENGTH = 120  # variable domain only
ANTIGEN_PDB = 'data/abeta_conformations/pdbs/abeta42_seed0_42.pdb'  # full Aβ42 (chain P)
AA = 'ACDEFGHIKLMNPQRSTVWY'

CDR_APPROX_RANGES_H = {'H1': (25, 35), 'H2': (50, 60), 'H3': (95, 110)}


def label_cdr_by_geometry(n, chain_type='H'):
    """CDR flags via Chothia-approx ranges scaled to length (mirrors preprocess_sabdab)."""
    flag = np.zeros(n, dtype=np.int32)
    if n < 80:
        return flag
    scale = n / 120.0
    ranges = CDR_APPROX_RANGES_H if chain_type == 'H' else {'L1': (23,37),'L2':(50,58),'L3':(88,100)}
    types = {'H1':1,'H2':2,'H3':3,'L1':4,'L2':5,'L3':6}
    for name,(s,e) in ranges.items():
        ss = max(0, min(int(s*scale), n-1)); ee = max(ss+1, min(int(e*scale), n))
        flag[ss:ee] = types[name]
    return flag


def make_fv_pdb(src_pdb, chain, fv_len, out_path):
    """Write a PDB with only the first fv_len residues of chain (the Fv).
    5CSZ heavy is a Fab (variable + constant). Truncate to variable domain so
    Chothia CDR ranges (26-102) map correctly."""
    with open(src_pdb) as f, open(out_path, 'w') as o:
        for line in f:
            if line.startswith('ATOM') and line[21].strip() == chain:
                try: rid = int(line[22:26])
                except: continue
                if rid <= fv_len:
                    o.write(line)
            elif line.startswith('ATOM'):
                pass  # drop other chains
        o.write('END\n')
    return out_path


def main():
    import bfn_loader
    bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
    from bfn_loader import run_bfn_design, load_bfn
    from disorderflow.utils.misc import seed_all
    from antibody_epitope_complex import position_epitope
    from disorderflow.datasets.protein import preprocess_protein_structure
    seed_all(42)

    APP = 'app_config.yaml'
    app = yaml.safe_load(open(APP, encoding='utf-8')); orig = app['models']['bfn']['checkpoint']
    ckpt = os.environ.get('V14_CKPT', 'logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt')
    app['models']['bfn']['checkpoint'] = ckpt
    open(APP, 'w', encoding='utf-8').write(yaml.dump(app, default_flow_style=False))
    model, _ = load_bfn('cuda')
    print(f"Loaded {ckpt} | seq_only={model.bfn.receiver.seq_only}")

    try:
        # Truncate 5CSZ H to Fv (~120aa)
        fv_pdb = 'data/anti_abeta_refs/5CSZ_H_Fv.pdb'
        make_fv_pdb(SCAFFOLD_PDB, SCAFFOLD_CHAIN, FV_LENGTH, fv_pdb)
        s_real = preprocess_protein_structure(fv_pdb, chain_ids=[SCAFFOLD_CHAIN])
        real_seq = ''.join(AA[a] if a < 20 else 'X' for a in s_real['chains'][0]['data']['aa'].tolist())
        n = len(real_seq)
        cdr_flag = label_cdr_by_geometry(n, 'H')
        print(f"5CSZ H-Fv: {n}aa. CDR regions (Chothia-approx):")
        real_cdr = {}
        for name in ['H1','H2','H3']:
            mask = cdr_flag == {'H1':1,'H2':2,'H3':3}[name]
            if mask.sum() > 0:
                real_cdr[name] = ''.join(c for c,m in zip(real_seq, mask) if m)
                idxs = np.where(mask)[0]
                print(f"  {name}: res {idxs[0]+1}-{idxs[-1]+1} ({mask.sum()}aa) = {real_cdr[name]}")

        # Build Ab-Fv + Aβ42 complex (antigen visible)
        out_dir = os.path.abspath('oc_validation_results/_complexes'); os.makedirs(out_dir, exist_ok=True)
        ecx = position_epitope(fv_pdb, ANTIGEN_PDB,
                               scaffold_chain=SCAFFOLD_CHAIN, epitope_chain='P',
                               distance=18.0, output_dir=out_dir)
        region_spec_parts = []
        for name,t in [('H1',1),('H2',2),('H3',3)]:
            m = cdr_flag == t
            if m.sum()>0:
                idxs = np.where(m)[0]
                region_spec_parts.append(f"{idxs[0]+1}-{idxs[-1]+1}")
        region_spec = f"{SCAFFOLD_CHAIN}:" + ",".join(region_spec_parts)
        print(f"\nDesign task: mask CDRs {region_spec}, complex mode (Aβ42 visible), recover real CDRs.")

        designs = run_bfn_design(ecx['pdb_path'], region_spec,
                                 num_samples=20, stochastic=True,
                                 context_chains=['P'], device='cuda',
                                 sort_by='iptm', descending=True)

        print(f"\n=== Sequence recovery vs real 5CSZ CDRs ({len(designs)} designs) ===")
        lengths = [int((cdr_flag==t).sum()) for t in [1,2,3]]
        for ci,name in enumerate(['H1','H2','H3']):
            if name not in real_cdr: continue
            real_cdr_str = real_cdr[name]
            l = lengths[ci]; offset = sum(lengths[:ci])
            recs = []; best_des=''
            for d in designs:
                des_cdr = d['sequence'][offset:offset+l]
                if len(des_cdr)==len(real_cdr_str):
                    match = sum(a==b for a,b in zip(des_cdr, real_cdr_str))
                    recs.append(match/len(real_cdr_str))
                    if recs and match/len(real_cdr_str)==max(recs): best_des=des_cdr
            if recs:
                print(f"  {name} real={real_cdr_str}")
                print(f"        best_design={best_des}")
                print(f"        recovery: mean={np.mean(recs)*100:.1f}%  max={np.max(recs)*100:.1f}%  (random ~5%)")

        print(f"\n=== best overall design (top by BFN iptm) ===")
        b = designs[0]
        print(f"  BFN iptm={b['iptm']:.4f} ppl={b['ppl']:.1f}")
        off=0
        for ci,name in enumerate(['H1','H2','H3']):
            if name not in real_cdr: continue
            l=lengths[ci]
            print(f"    {name}: real={real_cdr.get(name):18s} designed={b['sequence'][off:off+l]}")
            off+=l
    finally:
        app2 = yaml.safe_load(open(APP, encoding='utf-8')); app2['models']['bfn']['checkpoint']=orig
        open(APP,'w',encoding='utf-8').write(yaml.dump(app2, default_flow_style=False))


if __name__ == '__main__':
    main()
