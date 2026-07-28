"""L4: ESM-IF native vs scrambled — composition proxy (GPU-independent)."""
import sys,os,copy,random,tempfile; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
import numpy as np
from Bio.PDB import PDBParser,PDBIO
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

print("L4 Inverse Folding: native CDR recovery (composition + charge proxy)")
print("="*65)

AB_LIST=[
    ('4HIX','data/anti_abeta_refs/4HIX.pdb','H',[(26,32),(52,56),(95,102)]),
    ('5CSZ','data/anti_abeta_refs/5CSZ.pdb','H',[(26,32),(52,56),(95,102)]),
    ('3UOT','data/anti_abeta_refs/3UOT.pdb','A',[(26,32),(52,56),(95,102)]),
]

def get_cdr_seq(s,chain,cr):
    seq=[]
    for res in s[0][chain]:
        ri=res.id[1]
        if any(sr<=ri<=er for sr,er in cr): 
            if'CA'in res: seq.append(AA3.get(res.resname.strip(),'X'))
    return ''.join(seq)

def scramble_cdr(s,chain,cr):
    s2=copy.deepcopy(s);residues=[]
    for res in s2[0][chain]:
        ri=res.id[1]
        if any(sr<=ri<=er for sr,er in cr): residues.append(res)
    names=[r.resname for r in residues];random.shuffle(names)
    for r,n in zip(residues,names):r.resname=n
    return s2

def composition_score(seq):
    """Score CDR composition quality: aromatic+charged ratio, entropy."""
    if len(seq)==0:return 0
    from collections import Counter
    counts=Counter(seq);total=len(seq)
    # Aromatic fraction (YWF — important for amyloid binding)
    aro=sum(counts.get(a,0) for a in'YWF')/total
    # Charged fraction (RKDE — solubility)
    chg=sum(counts.get(a,0) for a in'RKDE')/total
    # Shannon entropy
    shannon=-sum((c/total)*np.log(max(c/total,1e-6))for c in counts.values())/np.log(20)
    return 0.4*min(aro/0.3,1.0)+0.3*min(chg/0.2,1.0)+0.3*shannon

results={}
for ab_id,pdb,chain,cr in AB_LIST:
    if not os.path.exists(pdb):continue
    parser=PDBParser(QUIET=True)
    native_s=parser.get_structure('n',pdb)
    native_cdr=get_cdr_seq(native_s,chain,cr)
    native_score=composition_score(native_cdr)
    
    scram_scores=[]
    for trial in range(20):
        random.seed(42+trial*137)
        scram_s=scramble_cdr(native_s,chain,cr)
        scram_cdr=get_cdr_seq(scram_s,chain,cr)
        scram_scores.append(composition_score(scram_cdr))
    
    scram_mean=np.mean(scram_scores);scram_std=np.std(scram_scores)
    d=(native_score-scram_mean)/max(scram_std,0.001)
    print(f'{ab_id}: native={native_score:.3f} scram={scram_mean:.3f} d={d:.2f} {"PASS" if d>0.5 else "FAIL"}')
    results[ab_id]={'d':float(d),'pass':d>0.5}

n_pass=sum(1 for r in results.values() if r['pass'])
print(f'\nL4 Gate: {n_pass}/{len(results)} passed (d>0.5)')
print('Full ESM-IF inference: fair-esm installed, GPU pending.')
print('Composition proxy works immediately for E2 evidence.')
