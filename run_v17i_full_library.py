#!/usr/bin/env python3
"""Full CDR library screen: varied strengths, dedup, AF2 validation."""
import sys,os,json,time; sys.path.insert(0,'.');sys.path.insert(0,'modules')
import torch,yaml,numpy as np; from easydict import EasyDict
from disorderflow.models.bfn_model import AntibodyBFN
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.misc import seed_all
from Bio.PDB import PDBParser,PDBIO
from build_design_variant_dataset import _batch_af2_wsl

CKPT='logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/4600.pt'
cfg=EasyDict(yaml.safe_load(open('configs/train/bfn_v17i_pairrouting_xpu.yml',encoding='utf-8')))
model=AntibodyBFN(cfg.model)
ckpt=torch.load(CKPT,map_location='cpu',weights_only=False)
model.load_state_dict(ckpt['model'],strict=False);model.eval()
print(f'V17i step {ckpt["iteration"]} val {ckpt.get("avg_val_loss","?"):.4f}')

AA='ACDEFGHIKLMNPQRSTVWY'
SCAFFOLDS=[('data/misfolding_targets/3STB.pdb','A','A:26-33,A:51-58,A:97-113'),
           ('data/misfolding_targets/5IMK.pdb','B','B:26-33,B:51-58,B:97-113')]
ANTIGENS=['data/abeta_conformations/pdbs/abeta42_seed0_42.pdb',
          'data/abeta_conformations/pdbs/abeta42_seed2_42.pdb',
          'data/abeta_conformations/pdbs/abeta42_seed4_42.pdb']
EPI='DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'

def parse_regions(spec):
    regions={}
    for part in spec.split(','):
        chain,rng=part.split(':');s,e=map(int,rng.split('-'))
        regions.setdefault(chain,[]).extend(range(s-1,e))
    return {k:sorted(set(v)) for k,v in regions.items()}

# Generate
all_designs=[]
for strength in [0.5,1.0,2.0,3.0]:
    for sp,sc,cs in SCAFFOLDS:
        regions_dict=parse_regions(cs)
        tx=get_transform([{'type':'mask_region','regions':regions_dict},{'type':'merge_protein'},{'type':'patch_protein'}])
        for ag in ANTIGENS:
            ss=preprocess_protein_structure(sp,chain_ids=[sc])
            ags=preprocess_protein_structure(ag,chain_ids=['P'])
            if ss is None or ags is None: continue
            merged={'id':f'{sc}+P','chains':ss['chains']+ags['chains'],
                    'num_chains':ss['num_chains']+ags['num_chains'],
                    'all_chain_ids':ss['all_chain_ids']+ags['all_chain_ids']}
            for si in range(5):
                seed_all(42+si*137+int(strength*100)+hash(sp)%1000)
                batch=recursive_to(PaddingCollate()([tx(merged)]),'cpu')
                gm=batch['generate_flag'][0].bool()
                if gm.sum()==0: continue
                with torch.no_grad():
                    traj=model.sample(batch,sample_opt={'deterministic':False,'num_recycles':3,
                        'disorder_guided':True,'disorder_guided_strength':strength})
                pl=traj['pred_logits'][0,gm.cpu()]
                cdr=''.join(AA[a] for a in pl.argmax(dim=-1).tolist())
                all_designs.append({'scaffold':os.path.basename(sp),'antigen':os.path.basename(ag),
                    'cdr_seq':cdr,'bfn_iptm':float(traj.get('iptm',torch.tensor([0])).mean()),
                    'bfn_plddt':float(traj.get('plddt',torch.zeros(1,gm.sum()))[0].mean()),
                    'strength':strength,'seed':si})
    print(f'str={strength}: {len([d for d in all_designs if d["strength"]==strength])} designs')
print(f'Total: {len(all_designs)} designs')

# Dedup at 80%
def sid(s1,s2):
    if len(s1)!=len(s2): return 0
    return sum(a==b for a,b in zip(s1,s2))/len(s1)
clusters=[]
for d in sorted(all_designs,key=lambda x:x['bfn_iptm'],reverse=True):
    if not any(sid(d['cdr_seq'],c['cdr_seq'])>0.80 for c in clusters):
        clusters.append(d)
clusters.sort(key=lambda x:x['bfn_iptm']+x['bfn_plddt'],reverse=True)
print(f'Unique (80%): {len(clusters)}')

# Add known templates
KNOWN=[('5CSZ','GFTFSSYAINSASGTRTARYCARGRGYV'),('4HIX','GFTFSSYAVSGSGGSTTYAKDRYSGY')]
top=list(clusters[:15])
for src,cdr in KNOWN:
    top.append({'scaffold':'3STB.pdb','antigen':'template','cdr_seq':cdr,'bfn_iptm':0,'bfn_plddt':0,'strength':0,'seed':-1})
print(f'AF2 queue: {len(top)} designs')

# Graft
aa3={'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
def graft(sp,sc,cdr_seq,cs,out):
    parser=PDBParser(QUIET=True);s=parser.get_structure('s',sp)
    regions=[]
    for part in cs.split(','):
        ch,rng=part.split(':');start,end=map(int,rng.split('-'))
        if ch==sc: regions.append((start-1,end))
    pos=0
    for start,end in regions:
        for j in range(start,end):
            if pos<len(cdr_seq):
                try: s[0][sc][j+1].resname=aa3.get(cdr_seq[pos],'GLY')
                except: pass
                pos+=1
    io=PDBIO();io.set_structure(s);io.save(out)

af2_dir='oc_validation_results/_complexes_v17i_v2';os.makedirs(af2_dir,exist_ok=True)
for i,d in enumerate(top):
    if d['scaffold']=='3STB.pdb': sp,sc,cs=SCAFFOLDS[0]
    else: sp,sc,cs=SCAFFOLDS[1]
    out=os.path.join(af2_dir,f'{os.path.basename(sp).replace(".pdb","")}_s{d["strength"]:.1f}_{i:02d}.pdb')
    graft(sp,sc,d['cdr_seq'],cs,out); d['grafted_pdb']=out
print(f'Grafted to {af2_dir}/')

# AF2
seqs=[]
for d in top:
    s=[];seen=set()
    with open(d['grafted_pdb']) as fh:
        for l in fh:
            if l.startswith('ATOM') and l[12:16].strip()=='CA':
                ri=l[22:27]
                if ri not in seen: seen.add(ri);s.append(aa3.get(l[17:20].strip(),'X'))
    seqs.append(''.join(s))
results=_batch_af2_wsl(seqs,EPI,num_recycle=1)
print('\n=== AF2 Results ===')
for i,r in enumerate(results):
    if r:
        top[i]['af2_iptm']=r.get('iptm',0); top[i]['af2_plddt']=r.get('plddt',0)
        top[i]['af2_success']=r.get('success',False)
        tag='[TEMPLATE]' if top[i].get('antigen')=='template' else ''
        print(f'{tag} {top[i]["scaffold"]} str={top[i].get("strength",0):.1f} AF2 iptm={r.get("iptm",0):.4f} | {top[i]["cdr_seq"][:50]}')

with open('oc_validation_results/oc_v17i_library_v2_af2.json','w') as f: json.dump(top,f,indent=2)
best=max((d.get('af2_iptm',0) for d in top if d.get('af2_success')),default=0)
prev_best=0.200
print(f'\nBest AF2 iptm: {best:.4f} (prev: {prev_best:.4f}, delta: {best-prev_best:+.4f})')
