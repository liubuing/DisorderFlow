import os
#!/usr/bin/env python3
"""Architecture probe: does the BFN sequence decoder USE the antigen signal?

The known-antibody recovery probe (probe_antibody_seq_recovery) showed ~5%
(random) recovery of real anti-Aβ CDRs. This isolates WHY: is the antigen even
conditioning the decoder? We compare seq recovery of REAL CDRs on SAbDab
complexes in two modes on the SAME scaffold:
  complex — antigen visible (context_chains=[antigen])
  fixbb   — antigen invisible (context_chains=[])
If recovery is identical in both → the encoder/decoder is not using the antigen
at all (architecture design flaw: antigen features never reach CDR decoding).
If complex > fixbb → antigen is used but weakly (data/training issue).

Free: BFN inference only, no AF2. Uses V15 (seq_only decoder).
"""
import sys, os, re, json, time
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import numpy as np, torch, lmdb, pickle, random, yaml
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.datasets.protein import preprocess_protein_structure
AA = 'ACDEFGHIKLMNPQRSTVWY'


def write_pdb(entry, out_path, include_antigen=True):
    """Write a merged PDB from a SAbDab entry dict {heavy,light,antigen}.
    Each chain's data has aa + pos_heavyatom. We reconstruct a minimal PDB
    (CA only + ideal N/C/O) so preprocess_protein_structure can re-read it with
    proper chain IDs. Heavy/antigen chain IDs set explicitly."""
    from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO
    st = Structure.Structure('s'); mo = Model.Model(0)
    chain_map = [('H', entry.get('heavy'))]
    if include_antigen and entry.get('antigen') is not None:
        chain_map.append(('P', entry.get('antigen')))
    for cid, d in chain_map:
        if d is None: continue
        ch = Chain.Chain(cid)
        aa = d['aa'].tolist()
        # pos_heavyatom: (L, n_heavy, 3); index 1 = CA
        pos = d['pos_heavyatom']
        for i, a in enumerate(aa):
            if a >= 20: continue
            res = Residue.Residue((' ', i+1, ' '), AA3.get(AA[a],'GLY'), ' ')
            ca = pos[i][1].tolist() if pos[i].shape[0] > 1 else [0,0,0]
            res.add(Atom.Atom('CA', ca, 0.0, 1.0, ' ', ' CA ', i+1, 'C'))
            # ideal N/C/O around CA along axis (placeholder, parser needs them)
            res.add(Atom.Atom('N', [ca[0]-1.46, ca[1], ca[2]], 0.0, 1.0, ' ', ' N  ', i+1, 'N'))
            res.add(Atom.Atom('C', [ca[0]+1.52, ca[1], ca[2]], 0.0, 1.0, ' ', ' C  ', i+1, 'C'))
            res.add(Atom.Atom('O', [ca[0]+2.1, ca[1], ca[2]], 0.0, 1.0, ' ', ' O  ', i+1, 'O'))
            ch.add(res)
        mo.add(ch)
    st.add(mo)
    io = PDBIO(); io.set_structure(st); io.save(out_path)


AA3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN','G':'GLY',
       'H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
       'T':'THR','W':'TRP','Y':'TYR','V':'VAL'}


def get_cdr_real(entry, scaffold_chain='H'):
    """Return {cdr_name: (start_idx, end_idx, real_seq_str)} from the entry's cdr_flag."""
    d = entry.get('heavy') if scaffold_chain=='H' else entry.get('light')
    if d is None: return {}
    cf = d.get('cdr_flag')
    if cf is None: return {}
    cf = cf.tolist()
    aa = d['aa'].tolist()
    seq = ''.join(AA[a] if a<20 else 'X' for a in aa)
    out = {}
    # cdr_flag values: H1=1,H2=2,H3=3,L1=4,L2=5,L3=6 (geometry labeling)
    names = {1:'H1',2:'H2',3:'H3'}
    for t,nm in names.items():
        idxs = [i for i,c in enumerate(cf) if c==t]
        if len(idxs) >= 3:
            out[nm] = (idxs[0], idxs[-1]+1, ''.join(seq[i] for i in idxs))
    return out


def run_recovery(model, complex_pdb, region_spec, context_chains, n_samples, device):
    """Design n_samples, return list of designed CDR-concat sequences."""
    from bfn_loader import run_bfn_design
    designs = run_bfn_design(complex_pdb, region_spec, num_samples=n_samples,
                             stochastic=True, context_chains=context_chains,
                             device=device, sort_by=None)
    return [d['sequence'] for d in designs]


def main():
    APP='app_config.yaml'
    app=yaml.safe_load(open(APP,encoding='utf-8')); orig=app['models']['bfn']['checkpoint']
    ckpt=os.environ.get('V14_CKPT','logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt')
    app['models']['bfn']['checkpoint']=ckpt
    open(APP,'w',encoding='utf-8').write(yaml.dump(app,default_flow_style=False))
    import bfn_loader; bfn_loader._bfn_model=None; bfn_loader._bfn_config=None
    from bfn_loader import load_bfn
    from disorderflow.utils.misc import seed_all
    model,_=load_bfn('cuda')
    print(f"Loaded {ckpt} | seq_only={model.bfn.receiver.seq_only}")

    env=lmdb.open('data/sabdab_phase3_processed/train.lmdb',readonly=True,lock=False,readahead=False,subdir=False)
    ids=pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids','rb'))
    random.seed(5)
    N_COMPLEX=8  # number of scaffolds
    N_SAMPLES=10
    tmpdir='oc_validation_results/_diag'; os.makedirs(tmpdir,exist_ok=True)

    try:
        picked=0
        agg={'complex':[], 'fixbb':[]}
        random.shuffle(ids)
        for sid in ids:
            if picked>=N_COMPLEX: break
            with env.begin() as txn: e=pickle.loads(txn.get(sid.encode()))
            if e.get('heavy') is None or e.get('antigen') is None: continue
            cdrs = get_cdr_real(e,'H')
            if not cdrs: continue
            # build both PDBs
            pdb_c = os.path.join(tmpdir,f'{sid}_c.pdb'); pdb_f=os.path.join(tmpdir,f'{sid}_f.pdb')
            try:
                write_pdb(e, pdb_c, include_antigen=True)
                write_pdb(e, pdb_f, include_antigen=False)
            except Exception as ex:
                print(f"  {sid}: write_pdb failed {ex}"); continue
            # region spec from cdrs (1-based)
            parts=[f"{s+1}-{e_}" for s,e_,_ in cdrs.values()]
            region_spec="H:"+",".join(parts)
            lengths=[e_-s for s,e_,_ in cdrs.values()]
            # validate parse + CDR count
            try:
                sc=preprocess_protein_structure(pdb_c, chain_ids=['H','P'])
            except Exception as ex:
                print(f"  {sid}: parse complex failed {ex}"); continue
            # design both modes
            seed_all(42)
            try:
                des_c=run_recovery(model, pdb_c, region_spec, ['P'], N_SAMPLES, 'cuda')
            except Exception as ex:
                print(f"  {sid}: complex design failed {ex}"); continue
            seed_all(42)
            try:
                des_f=run_recovery(model, pdb_f, region_spec, [], N_SAMPLES, 'cuda')
            except Exception as ex:
                print(f"  {sid}: fixbb design failed {ex}"); continue
            # per-CDR recovery
            for ci,nm in enumerate(['H1','H2','H3']):
                if nm not in cdrs: continue
                s,e_,real=cdrs[nm]; l=e_-s; off=sum(lengths[:ci])
                rc=[]
                for d in des_c:
                    dc=d[off:off+l]
                    if len(dc)==len(real): rc.append(sum(a==b for a,b in zip(dc,real))/len(real))
                rf=[]
                for d in des_f:
                    dc=d[off:off+l]
                    if len(dc)==len(real): rf.append(sum(a==b for a,b in zip(dc,real))/len(real))
                if rc and rf:
                    agg['complex'].append(np.mean(rc)); agg['fixbb'].append(np.mean(rf))
            print(f"  {sid}: OK (antigen {len(e['antigen']['aa'])}aa) — complex/rec per-CDR logged")
            picked+=1

        c=np.array(agg['complex']); f=np.array(agg['fixbb'])
        print(f"\n=== Antigen-signal usage (n={len(c)} CDR-scaffold instances) ===")
        print(f"  complex (antigen visible): mean recovery {c.mean()*100:.1f}%")
        print(f"  fixbb   (antigen hidden):  mean recovery {f.mean()*100:.1f}%")
        print(f"  Δ(complex-fixbb) = {(c.mean()-f.mean())*100:+.2f} pp")
        if abs(c.mean()-f.mean()) < 0.01:
            print("  => IDENTICAL recovery: antigen signal NOT used by decoder (architecture flaw).")
        elif c.mean() > f.mean()+0.01:
            print("  => complex > fixbb: antigen is used (weakly); issue is data/training.")
        else:
            print("  => fixbb > complex: antigen VISIBILITY HURTS (OOD, as seen before).")
    finally:
        env.close()
        app2=yaml.safe_load(open(APP,encoding='utf-8')); app2['models']['bfn']['checkpoint']=orig
        open(APP,'w',encoding='utf-8').write(yaml.dump(app2,default_flow_style=False))


if __name__=='__main__':
    main()
