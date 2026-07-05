"""WAY4 V6 P0: IDP-antigen training subset — the data gate.

Scans SAbDab phase3 antigens with disorder head, computes epitope disorder
distribution. Must show non-degenerate distribution before P1 training.

Iron rule: P0 fails → stop, do not proceed to P1.
"""
import sys,os,json,time,pickle,random
from collections import defaultdict
sys.path.insert(0,'.');sys.path.insert(0,'modules')
import numpy as np
import torch,lmdb,yaml

V15_CKPT='logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt'

print("="*60)
print("WAY4 V6 P0: IDP-Antigen Training Subset — Data Gate")
print("="*60)

# Load BFN model with disorder head
import bfn_loader
app=yaml.safe_load(open('app_config.yaml',encoding='utf-8'))
orig=app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint']=V15_CKPT
yaml.dump(app,open('app_config.yaml','w',encoding='utf-8'))
bfn_loader._bfn_model=None;bfn_loader._bfn_config=None
model,_=bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint']=orig
yaml.dump(app,open('app_config.yaml','w',encoding='utf-8'))
print("Model loaded")

from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.modules.common.geometry import construct_3d_basis

def predict_disorder_antigen(model,batch):
    """Get per-residue disorder on ANTIGEN chain only."""
    with torch.no_grad():
        N,L=batch['aa'].shape;device=batch['aa'].device
        theta_seq=torch.zeros(N,L,22,device=device)
        pos_mean=model.bfn.position_mean;pos_scale=model.bfn.position_scale
        theta_pos_norm=(batch['pos_heavyatom'][:,:,1].float()-pos_mean)/pos_scale
        theta_ori=construct_3d_basis(batch['pos_heavyatom'][:,:,1],batch['pos_heavyatom'][:,:,2],batch['pos_heavyatom'][:,:,0])
        theta_ang=batch.get('torsion',torch.zeros(N,L,4,device=device))
        t=0.5*torch.ones(N,device=device)
        pair_feat=batch.get('pair_feat',torch.zeros(N,L,L,128,device=device))
        mask_res=batch['mask'].bool()
        backbone_pos=batch['pos_heavyatom'][:,:,:4]
        mask_gen=batch.get('generate_flag',torch.zeros(N,L).bool())
        mask_antigen=batch.get('mask_antigen',None)

        out=model.bfn.receiver(theta_seq,theta_pos_norm,theta_ori,theta_ang,t,pair_feat,mask_res,backbone_pos=backbone_pos,mask_gen=mask_gen,mask_antigen=mask_antigen)
        pred_disorder=out[7]
        if pred_disorder is None:return np.array([])
        disorder=torch.sigmoid(pred_disorder)
        # Filter to antigen only
        ag_mask=batch.get('mask_antigen',torch.zeros(N,L).bool())
        if ag_mask.any():
            return disorder[0].cpu().numpy()[ag_mask[0].cpu().numpy()]
        return disorder[0].cpu().numpy()

# Scan SAbDab phase3 train
env=lmdb.open('data/sabdab_phase3_processed/train.lmdb',subdir=False,readonly=True,lock=False,readahead=False)
ids=pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids','rb'))
random.seed(42);random.shuffle(ids)

transform=get_transform([{'type':'merge_chains'},{'type':'patch_around_anchor'}])

epitope_disorders=[]
n_with_antigen=0;n_scanned=0

print(f"Scanning {len(ids)} SAbDab complexes...")
t0=time.time()
for i,sid in enumerate(ids):
    if n_scanned>=500:break  # sample 500 for speed
    with env.begin() as txn:e=pickle.loads(txn.get(sid.encode()))
    if e.get('antigen') is None or e.get('heavy') is None:continue
    n_with_antigen+=1
    try:
        batch_data=transform(e)
        batch=recursive_to(PaddingCollate()([batch_data]),'cpu')
        # Build mask_antigen manually
        ag_len=len(e['antigen']['aa'])
        ab_len=len(batch_data['aa'])
        mask_antigen=torch.zeros(1,ab_len).bool()
        mask_antigen[0,-ag_len:]=True  # antigen is last after merge
        batch['mask_antigen']=mask_antigen

        disorder=predict_disorder_antigen(model,batch)
        if len(disorder)>0:
            epitope_disorders.append({'sid':sid,'ag_len':ag_len,
                'disorder_mean':float(disorder.mean()),'disorder_max':float(disorder.max()),
                'disorder_std':float(disorder.std()),'n_flexible':int((disorder>0.3).sum())})
        n_scanned+=1
    except Exception:continue
    if (i+1)%100==0:print(f"  {i+1}/{len(ids)} ({n_scanned} done, {time.time()-t0:.0f}s)")
env.close()

print(f"\nScanned {n_scanned} complexes with antigen ({time.time()-t0:.0f}s)")

# Statistics
means=[d['disorder_mean']for d in epitope_disorders]
n_flex=(sum(1 for m in means if m>0.3))
n_mid=(sum(1 for m in means if 0.1<m<=0.3))
n_ordered=(sum(1 for m in means if m<=0.1))

print(f"\n=== P0 GATE: Epitope Disorder Distribution ===")
print(f"Total complexes with antigen: {n_with_antigen}")
print(f"Scanned: {n_scanned}")
print(f"\nEpitope disorder_mean distribution:")
print(f"  Ordered (≤0.1): {n_ordered} ({100*n_ordered/n_scanned:.1f}%)")
print(f"  Mid (0.1-0.3):  {n_mid} ({100*n_mid/n_scanned:.1f}%)")
print(f"  Flexible (>0.3): {n_flex} ({100*n_flex/n_scanned:.1f}%)")
print(f"  Mean disorder: {np.mean(means):.4f} ± {np.std(means):.4f}")
print(f"  Range: {np.min(means):.4f} - {np.max(means):.4f}")

# Gate decision
if n_flex>=5:
    print(f"\nP0 GATE: PASS — {n_flex} flexible complexes (>{5})")
    print("Non-degenerate disorder distribution confirmed.")
    print("→ Proceed to P1 (disorder-conditioned training)")
    gate='PASS'
elif n_flex>=2:
    print(f"\nP0 GATE: MARGINAL — only {n_flex} flexible complexes")
    print("Disorder gradient exists but weak. Augment with IDP data needed.")
    gate='MARGINAL'
else:
    print(f"\nP0 GATE: FAIL — only {n_flex} flexible complexes (need >2)")
    print("0.07% IDP not enough for training signal. STOP.")
    gate='FAIL'

# Save
os.makedirs('data/idp_antigen_subset',exist_ok=True)
with open('data/idp_antigen_subset/disorder_stats.json','w')as f:
    json.dump({'n_scanned':n_scanned,'n_ordered':n_ordered,'n_mid':n_mid,
               'n_flexible':n_flex,'mean':float(np.mean(means)),'std':float(np.std(means)),
               'gate':gate,'details':epitope_disorders},f,indent=2)
print(f"\nSaved data/idp_antigen_subset/disorder_stats.json")
