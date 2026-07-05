"""WAY4 V6 P2: Mechanism validation — disorder-conditioned CDR pliability.

Tests V18 vs V17i on high-disorder (Aβ42) vs low-disorder (folded) epitopes.
Measures: AA entropy, unique sequences, aromatic fraction, anti-degen checks.
Key question: does V18 produce more pliable CDRs for high-disorder epitopes?
"""
import sys,os,json,time,tempfile,random,math
from collections import Counter
sys.path.insert(0,'.');sys.path.insert(0,'modules')
import numpy as np
import torch,yaml
from easydict import EasyDict

AA='ACDEFGHIKLMNPQRSTVWY'
V18_CKPT='logs/bfn_v18_disorder_cond_xpu_2026_07_01__15_51_01_v18_way4/checkpoints/best.pt'
V17I_CKPT='logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/best.pt'

EPITOPES={
    'high_disorder':{'seq':'KLVFFAED','disorder':0.93,'label':'Aβ42_agg_core'},
    'low_disorder':{'seq':'AAAAAAAA','disorder':0.05,'label':'polyA_control'},
    'mid_disorder':{'seq':'GFTFSNYA','disorder':0.50,'label':'mid_test'},
}
SCAFFOLD='data/misfolding_targets/3STB.pdb'
SCAFFOLD_CHAIN='A';CDR_SPEC='A:26-33,A:51-58,A:97-113'
N_SAMPLES=10

def load_v18():
    from disorderflow.models.bfn_model import AntibodyBFN
    cfg=EasyDict(yaml.safe_load(open('configs/train/bfn_v18_disorder_cond_xpu.yml',encoding='utf-8')))
    model=AntibodyBFN(cfg.model)
    ckpt=torch.load(V18_CKPT,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'],strict=False);model.eval()
    return model

def load_v17i():
    from disorderflow.models.bfn_model import AntibodyBFN
    cfg_v17=EasyDict(yaml.safe_load(open('configs/train/bfn_v17i_pairrouting_xpu.yml',encoding='utf-8')))
    model=AntibodyBFN(cfg_v17.model)
    ckpt=torch.load(V17I_CKPT,map_location='cpu',weights_only=False)
    state=ckpt['model']
    # Strip new V18 heads that V17i doesn't have
    for k in list(state.keys()):
        if any(n in k for n in ['disorder_proj.','head_seq.','head_seq_fixbb.']):
            state.pop(k)
    model.load_state_dict(state,strict=False);model.eval()
    return model

def generate_cdr(model,epitope_seq,epi_disorder,n_samples=N_SAMPLES):
    """Generate CDR designs with a given epitope disorder context."""
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.transforms import get_transform
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from Bio.PDB import Model,Chain,Residue,Atom,PDBParser,PDBIO

    # Build scaffold + peptide complex
    parser=PDBParser(QUIET=True);s=parser.get_structure('s',SCAFFOLD)
    merged=s[0].copy()
    pep=Chain.Chain('P')
    cdr_ca=[res['CA'].get_coord()for res in merged[SCAFFOLD_CHAIN]if'CA'in res and any(sr<=res.id[1]<=er for sr,er in[(26,32),(52,56),(95,102)])]
    ctr=np.array(cdr_ca).mean(axis=0)if cdr_ca else np.array([0,0,0])
    for i,aa in enumerate(epitope_seq):
        r=Residue.Residue((' ',i+1,' '),{'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN','G':'GLY','H':'HIS','I':'ILE','L':'LEU','LYS':'LYS','MET':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}.get(aa,'GLY'),' ')
        ca=ctr+np.array([12+i*3.8,3,-2])
        r.add(Atom.Atom('CA',ca.tolist(),0,1,' ',' CA ',i+1,'C'))
        r.add(Atom.Atom('N',(ca+[-1.46,0,0]).tolist(),0,1,' ',' N  ',i+1,'N'))
        r.add(Atom.Atom('C',(ca+[1.52,0,0]).tolist(),0,1,' ',' C  ',i+1,'C'))
        r.add(Atom.Atom('O',(ca+[2.1,0,0]).tolist(),0,1,' ',' O  ',i+1,'O'))
        pep.add(r)
    merged.add(pep)

    # Preprocess
    struct={'heavy':{'aa':torch.tensor([AA.index(aa)for aa in ''.join(AA3.get(r.resname.strip(),'X')for r in merged[SCAFFOLD_CHAIN]if'CA'in r)]),
        'pos_heavyatom':torch.tensor([r['CA'].get_coord().tolist()for r in merged[SCAFFOLD_CHAIN]if'CA'in r]).unsqueeze(1).repeat(1,4,1),
        'torsion':torch.zeros(len([r for r in merged[SCAFFOLD_CHAIN]if'CA'in r]),4),'mask':torch.ones(len([r for r in merged[SCAFFOLD_CHAIN]if'CA'in r])).bool(),
        'fragment_type':torch.zeros(len([r for r in merged[SCAFFOLD_CHAIN]if'CA'in r])).long(),
        'chain_id':['A']*len([r for r in merged[SCAFFOLD_CHAIN]if'CA'in r])},
        'light':None,'antigen':{'aa':torch.tensor([AA.index(aa)for aa in epitope_seq]),
        'pos_heavyatom':torch.tensor([[pep[i]['CA'].get_coord().tolist()]*4 for i in range(len(epitope_seq))]),
        'torsion':torch.zeros(len(epitope_seq),4),'mask':torch.ones(len(epitope_seq)).bool(),
        'fragment_type':torch.ones(len(epitope_seq)).long()*2,'chain_id':['P']*len(epitope_seq)}}

    cdrs=[]
    for si in range(n_samples):
        tx=get_transform([{'type':'mask_multiple_cdrs'},{'type':'merge_chains'},{'type':'patch_around_anchor'}])
        try:
            batch=recursive_to(PaddingCollate()([tx(struct)]),'cpu')
        except Exception:continue
        batch['epitope_disorder']=torch.tensor([[epi_disorder]])
        with torch.no_grad():
            traj=model.sample(batch,sample_opt={'deterministic':False,'num_recycles':1})
        pred=traj['pred_logits'][0,batch['generate_flag'][0].bool()].argmax(dim=-1).tolist()
        cdr=''.join(AA[a]for a in pred)
        cdrs.append(cdr)
    return cdrs

AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

print("WAY4 V6 P2: Mechanism Validation")
print("="*60)

# Load models
print("Loading V18...");v18=load_v18()
print("Loading V17i...");v17=load_v17i()
print()

for epi_name,epi in EPITOPES.items():
    print(f"\n[{epi['label']}] disorder={epi['disorder']}")
    v18_cdrs=generate_cdr(v18,epi['seq'],epi['disorder'])
    v17_cdrs=generate_cdr(v17,epi['seq'],epi['disorder'])

    for model_name,cdrs in [('V18',v18_cdrs),('V17i',v17_cdrs)]:
        entropies=[]
        for cdr in cdrs:
            cnt=Counter(cdr);total=len(cdr)
            h=-sum((c/total)*math.log(max(c/total,1e-6))for c in cnt.values())
            entropies.append(h)
        unique=len(set(cdrs))
        aro=np.mean([sum(1 for a in cdr if a in'YWF')/len(cdr)for cdr in cdrs])

        # Anti-degen check
        max_run=0;all_run=0
        for cdr in cdrs:
            r=1
            for i in range(1,len(cdr)):
                if cdr[i]==cdr[i-1]:r+=1
                else:r=1
                if r>max_run:max_run=r
                if r>=6:all_run+=1

        print(f"  {model_name}: entropy={np.mean(entropies):.2f} unique={unique}/{N_SAMPLES} aro={aro:.2f} max_run={max_run} degen={all_run}")

# Gradient check
print(f"\n=== MECHANISM GRADIENT ===")
v18_high=np.mean([sum(1 for a in ''.join(generate_cdr(v18,'KLVFFAED',0.93,10))if a in'YWF')for _ in range(3)])
v18_low=np.mean([sum(1 for a in ''.join(generate_cdr(v18,'AAAAAAAA',0.05,10))if a in'YWF')for _ in range(3)])
print(f"V18 aromatic fraction: high={v18_high:.1f} vs low={v18_low:.1f}")
print(f"Gradient: {'SIGNIFICANT' if v18_high>v18_low*1.2 else 'WEAK' if v18_high>v18_low else 'NONE'}")
