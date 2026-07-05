"""L3 re-rank: score P3 designs with interface_scorer, compare vs old composite."""
import sys,os,json,time,tempfile; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
import numpy as np
from Bio.PDB import PDBParser,PDBIO,Model,Chain,Residue,Atom
from modules.interface_scorer import score_pose
from scipy.stats import kendalltau

with open('idp_design_results/p3_scale_all_20260629_030111.json') as f:
    data=json.load(f)
designs=data.get('designs',data.get('ensemble',[]))
N=min(len(designs),100)
print(f'L3 re-ranking top {N}/{len(designs)} designs')

AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
old_c,new_c=[],[]

for i,d in enumerate(designs[:N]):
    sp=d.get('scaffold_path','')or f'data/misfolding_targets/{d["scaffold"]}.pdb'
    if not os.path.exists(sp): continue
    parser=PDBParser(QUIET=True);s=parser.get_structure('s',sp)
    ch=d['scaffold_chain'];cs=d['cdr_spec']
    cr=[(int(p.split(':')[1].split('-')[0]),int(p.split(':')[1].split('-')[1]))for p in cs.split(',')]
    cdr=d['sequence'];pos=0
    for st,en in cr:
        for j in range(st,en+1):
            if pos<len(cdr):
                try:s[0][ch][j].resname=AA3.get(cdr[pos],'GLY')
                except:pass;pos+=1
    # CDR center for peptide placement
    ca=[]
    for res in s[0][ch]:
        ri=res.id[1]
        if any(sr<=ri<=er for sr,er in cr):
            if'CA'in res:ca.append(res['CA'].get_coord())
    ctr=np.array(ca).mean(axis=0)if ca else np.array([0,0,0])
    epi='KLVFFAED'
    pc=Chain.Chain('P')
    for j,aa in enumerate(epi):
        r=Residue.Residue((' ',j+1,' '),AA3.get(aa,'GLY'),' ')
        ca_pos=ctr+np.array([15+j*3.8,5,0])
        r.add(Atom.Atom('CA',ca_pos.tolist(),0,1,' ',' CA ',j+1,'C'))
        r.add(Atom.Atom('N',(ca_pos+[-1.46,0,0]).tolist(),0,1,' ',' N  ',j+1,'N'))
        r.add(Atom.Atom('C',(ca_pos+[1.52,0,0]).tolist(),0,1,' ',' C  ',j+1,'C'))
        r.add(Atom.Atom('O',(ca_pos+[2.1,0,0]).tolist(),0,1,' ',' O  ',j+1,'O'))
        pc.add(r)
    s[0].add(pc)
    tmp=tempfile.mktemp(suffix='.pdb');io=PDBIO();io.set_structure(s);io.save(tmp)
    r=score_pose(tmp,ch,'P',cr);os.unlink(tmp)
    old_c.append(d.get('composite',0));new_c.append(r['composite'])
    if(i+1)%20==0:print(f'  {i+1}/{N}')

tau,p=kendalltau(old_c,new_c)
print(f'\nL3 Re-rank: N={len(new_c)}')
print(f'Old composite: {np.min(old_c):.4f}-{np.max(old_c):.4f}')
print(f'New L3 score:   {np.min(new_c):.4f}-{np.max(new_c):.4f}')
print(f'Kendall tau: {tau:.4f} (p={p:.4f})')
if abs(tau)<0.2:print('RANKINGS DIFFERENT — V3 sees new signal')
elif tau>0.3:print('RANKINGS CORRELATED — V3 confirms old')
else:print('WEAK correlation')

with open('idp_design_results/v3_kendall.json','w')as f:
    json.dump({'tau':float(tau),'p':float(p),'n':len(new_c),
               'old_range':[float(np.min(old_c)),float(np.max(old_c))],
               'new_range':[float(np.min(new_c)),float(np.max(new_c))]},f,indent=2)
print('Saved v3_kendall.json')
