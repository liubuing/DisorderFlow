"""L5: AF2 single-chain CDR pLDDT — avoids multimer peptide trap."""
import sys,os,json; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
from build_design_variant_dataset import _batch_af2_wsl
from Bio.PDB import PDBParser
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

print("L5 AF2 single-chain: antibody-only pLDDT (NO peptide)")
print("="*55)

# Load top-5 P3 designs
with open('idp_design_results/p3_scale_top100_20260629_030111.json') as f:
    designs=json.load(f)[:5]

# Graft CDRs, get full antibody sequence (no peptide)
seqs=[]
for d in designs:
    sp=d.get('scaffold_path','')or f'data/misfolding_targets/{d["scaffold"]}.pdb'
    parser=PDBParser(QUIET=True);s=parser.get_structure('s',sp)
    ch=d['scaffold_chain'];cs=d['cdr_spec']
    cr=[(int(p.split(':')[1].split('-')[0]),int(p.split(':')[1].split('-')[1]))for p in cs.split(',')]
    cdr=d['sequence'];pos=0
    for st,en in cr:
        for j in range(st,en+1):
            if pos<len(cdr):
                try:s[0][ch][j].resname=AA3.get(cdr[pos],'GLY')
                except:pass;pos+=1
    seq=[];seen=set()
    for res in s[0][ch]:
        if'CA'in res:
            ri=res.id[1]
            if ri not in seen:seen.add(ri);seq.append(AA3.get(res.resname.strip(),'X'))
    seqs.append(''.join(seq))

# AF2 single-chain: antibody only, NO epitope
print(f'Running AF2 single-chain on {len(seqs)} antibodies...')
# Use empty epitope to force single-chain mode
results=_batch_af2_wsl(seqs,'',num_recycle=1)

cdr_plddts=[]
for i,(d,r) in enumerate(zip(designs,results)):
    if r and r.get('success'):
        plddt=r.get('plddt',0)
        iptm=r.get('iptm',0)
        print(f'  #{i}: pLDDT={plddt:.4f} iptm={iptm:.4f} {d["epitope"][:12]}+{d["scaffold"]}')
        cdr_plddts.append(plddt)

if cdr_plddts:
    import numpy as np
    print(f'\nCDR pLDDT range: {np.min(cdr_plddts):.4f}-{np.max(cdr_plddts):.4f}')
    print('L5 AF2 single-chain working. pLDDT values provide E3 evidence.')
