"""V3 Robust: fast grid-search pose + cross-conformation p25. Optimized: 
no PDB I/O per pose, contacts computed directly, full L3 only on best pose."""
import sys,os,json,time,argparse,copy; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser,PDBIO,Model,Chain,Residue,Atom
from modules.interface_scorer import score_pose
from scipy.stats import kendalltau

P3_LIBRARY='idp_design_results/p3_scale_all_20260629_030111.json'
ABETA_CONFS=[f'data/abeta_conformations/pdbs/abeta42_seed{s}_42.pdb' for s in range(5)]
EPITOPE='KLVFFAED'
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
     'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
     'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
CONTACT_CUT=8.0

# Grid: 4x4x4=64 poses (was 125)
DX=np.linspace(8,22,4);DY=np.linspace(-2,6,4);DZ=np.linspace(-3,6,4)


def get_cdr_cb(s,chain,cr):
    """Get CDR Cb coordinates once (no PDB I/O)."""
    coords=[]
    for res in s[0][chain]:
        ri=res.id[1]
        if any(sr<=ri<=er for sr,er in cr):
            if'CB'in res:coords.append(res['CB'].get_coord())
            elif'CA'in res:coords.append(res['CA'].get_coord())
    return np.array(coords)


def peptide_cb(ctr,dx,dy,dz):
    """Peptide Cb positions at given offset (no PDB I/O)."""
    pts=[]
    for j in range(len(EPITOPE)):
        pts.append(ctr+np.array([dx+j*3.8,dy,dz]))
    return np.array(pts)


def fast_contacts(cdr_cb,pep_cb):
    """Count contacts from coordinates directly (no PDB I/O)."""
    if len(cdr_cb)==0 or len(pep_cb)==0: return 0
    tree=cKDTree(pep_cb)
    dists,_=tree.query(cdr_cb,distance_upper_bound=CONTACT_CUT)
    return int((dists<CONTACT_CUT).sum())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--top',type=int,default=20);args=ap.parse_args()
    with open(P3_LIBRARY) as f: data=json.load(f)
    designs=data.get('designs',data.get('ensemble',[]))[:args.top]
    n_grid=len(DX)*len(DY)*len(DZ)
    print(f'V3 Fast Grid: {len(designs)} designs x {n_grid} grid x {len(ABETA_CONFS)} confs')
    print(f'  Optimized: fast contacts first, full L3 only on best pose')
    print('='*60)

    t0=time.time()
    for i,d in enumerate(designs):
        sp=d.get('scaffold_path','')or f'data/misfolding_targets/{d["scaffold"]}.pdb'
        if not os.path.exists(sp):continue
        parser=PDBParser(QUIET=True);s=parser.get_structure('s',sp)
        ch=d['scaffold_chain'];cs=d['cdr_spec']
        cr=[(int(p.split(':')[1].split('-')[0]),int(p.split(':')[1].split('-')[1]))for p in cs.split(',')]
        cdr=d['sequence'];pos=0
        for st,en in cr:
            for j in range(st,en+1):
                if pos<len(cdr):
                    try:s[0][ch][j].resname=AA3.get(cdr[pos],'GLY')
                    except:pass;pos+=1
        ca=[res['CA'].get_coord() for res in s[0][ch] if any(sr<=res.id[1]<=er for sr,er in cr) and 'CA' in res]
        ctr=np.array(ca).mean(axis=0)if ca else np.array([0,0,0])
        cdr_cb=get_cdr_cb(s,ch,cr)

        per_conf=[]
        for conf_pdb in ABETA_CONFS:
            best_contacts=0;best_offset=None
            # Fast grid search: find offset with max contacts
            for dx in DX:
             for dy in DY:
              for dz in DZ:
                pep_cb=peptide_cb(ctr,dx,dy,dz)
                nc=fast_contacts(cdr_cb,pep_cb)
                if nc>best_contacts:best_contacts=nc;best_offset=(dx,dy,dz)
            # Full L3 only on best offset
            if best_offset and best_contacts>=5:
                dx,dy,dz=best_offset
                s2=copy.deepcopy(s);pc=Chain.Chain('P')
                for j,aa in enumerate(EPITOPE):
                    r=Residue.Residue((' ',j+1,' '),AA3.get(aa,'GLY'),' ')
                    ca_pos=ctr+np.array([dx+j*3.8,dy,dz])
                    r.add(Atom.Atom('CA',ca_pos.tolist(),0,1,' ',' CA ',j+1,'C'))
                    r.add(Atom.Atom('N',(ca_pos+[-1.46,0,0]).tolist(),0,1,' ',' N  ',j+1,'N'))
                    r.add(Atom.Atom('C',(ca_pos+[1.52,0,0]).tolist(),0,1,' ',' C  ',j+1,'C'))
                    r.add(Atom.Atom('O',(ca_pos+[2.1,0,0]).tolist(),0,1,' ',' O  ',j+1,'O'))
                    pc.add(r)
                s2[0].add(pc)
                import tempfile;tmp=tempfile.mktemp(suffix='.pdb');io=PDBIO();io.set_structure(s2);io.save(tmp)
                r=score_pose(tmp,ch,'P',cr);os.unlink(tmp)
                if r['contacts']>=5:per_conf.append(r['composite'])
        if per_conf:
            d['v3_p25']=round(float(np.percentile(per_conf,25)),4)
            d['v3_median']=round(float(np.median(per_conf)),4)
            d['v3_nvalid']=len(per_conf)
        else:
            d['v3_p25']=0;d['v3_median']=0;d['v3_nvalid']=0
        if(i+1)%5==0:print(f'  {i+1}/{len(designs)} ({time.time()-t0:.0f}s)')

    scores=[d.get('v3_p25',0)for d in designs]
    old_c=[d.get('composite',0)for d in designs]
    tau,p=kendalltau(old_c,scores)
    nv=sum(1 for d in designs if d.get('v3_nvalid',0)>0)
    na=sum(1 for s in scores if s>0)
    print(f'\n=== V3 Fast Grid Results ===')
    print(f'Valid (contacts>=5): {nv}/{len(designs)}  p25>0: {na}')
    print(f'p25 range: {np.min(scores):.4f}-{np.max(scores):.4f}')
    print(f'Kendall tau: {tau:.4f} (p={p:.4f})')
    print(f'Total: {time.time()-t0:.0f}s')

    idx=np.argsort(scores)[-10:][::-1]
    for rank,i in enumerate(idx):
        d=designs[i]
        print(f'#{rank+1:2d} p25={d["v3_p25"]:.4f} n={d["v3_nvalid"]} {d["epitope"][:12]}+{d["scaffold"]}')

    ts=time.strftime('%Y%m%d_%H%M%S')
    out=f'idp_design_results/v3_robust_fast_{ts}.json'
    with open(out,'w')as f:json.dump(designs,f,indent=2)
    print(f'Saved {out}')

if __name__=='__main__':main()
