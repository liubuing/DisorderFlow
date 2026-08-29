#!/bin/bash
# V13.1 Sequential Ablation Training
# Runs A1 -> A2 -> A3 sequentially to avoid GPU competition.
# Each: finetune from V20 best.pt, 4000 iterations on CUDA.
# Estimated wall time: 18-36 hours total.

set -e
source venv_wsl/bin/activate
cd /mnt/d/biological/DisorderFlow

V20_CKPT="logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_BASE="logs/v13_ablation_${TIMESTAMP}"
mkdir -p "$LOG_BASE"

echo "============================================================" | tee "$LOG_BASE/master.log"
echo "V13.1 Sequential Ablation: A1 -> A2 -> A3" | tee -a "$LOG_BASE/master.log"
echo "Started at: $(date)" | tee -a "$LOG_BASE/master.log"
echo "============================================================" | tee -a "$LOG_BASE/master.log"

# ── A1: Remove diversity loss ──
echo "" | tee -a "$LOG_BASE/master.log"
echo "========== A1: NO DIVERSITY LOSS ==========" | tee -a "$LOG_BASE/master.log"
echo "Started: $(date)" | tee -a "$LOG_BASE/master.log"

python train.py \
  logs/bfn_v20_ablate_a1_nodiv/bfn_v20_ablate_a1_nodiv.yml \
  --device cuda \
  --finetune "$V20_CKPT" \
  --tag v13_a1_nodiv \
  2>&1 | tee "$LOG_BASE/a1_train.log"

A1_CKPT=$(find logs/ -name "best.pt" -path "*ablate_a1*" -newer "$LOG_BASE" 2>/dev/null | head -1)
if [ -z "$A1_CKPT" ]; then
  # Fallback: find most recent a1 checkpoint
  A1_CKPT=$(ls -t logs/bfn_v20_ablate_a1_nodiv_*/checkpoints/best.pt 2>/dev/null | head -1)
fi
echo "A1 checkpoint: $A1_CKPT" | tee -a "$LOG_BASE/master.log"
echo "A1 done: $(date)" | tee -a "$LOG_BASE/master.log"

# ── A1 P2 probe ──
echo "Running A1 P2 probe..." | tee -a "$LOG_BASE/master.log"
python -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
import json, pickle, numpy as np, torch, yaml, lmdb, random, time
from collections import defaultdict
from scipy.stats import spearmanr

CKPT = '$A1_CKPT'
print(f'Loading {CKPT}')
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = CKPT
yaml.dump(app, open('app_config.yaml','w',encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig
yaml.dump(app, open('app_config.yaml','w',encoding='utf-8'))
print('Model loaded')

lookup = pickle.load(open('data/sabdab_disorder_lookup.pkl','rb'))
high_flex = [(k,v) for k,v in lookup.items() if v.max()>0.4]
low_flex = [(k,v) for k,v in lookup.items() if v.max()<0.2]
random.seed(42)
test_high = random.sample(high_flex, min(8, len(high_flex)))
test_low = random.sample(low_flex, min(8, len(low_flex)))
env_check = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
valid_ids = set(pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids','rb')))
env_check.close()
test_high = [(k,v) for k,v in test_high if k in valid_ids]
test_low = [(k,v) for k,v in test_low if k in valid_ids]
print(f'Test: {len(test_high)}H, {len(test_low)}L')

transform = get_transform([{'type':'mask_multiple_cdrs'},{'type':'merge_chains'},{'type':'patch_around_anchor'}])
env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
AA = 'ARNDCQEGHILKMFPSTWYV'
N_SEEDS=3

def build_batch(sid, dm, darr):
    with env.begin() as txn:
        raw = txn.get(sid.encode())
        if raw is None: return None, None
        e = pickle.loads(raw)
    if e.get('heavy') is None or e.get('antigen') is None: return None, None
    bd = transform(e)
    batch = recursive_to(PaddingCollate()([bd]), 'cpu')
    L = batch['aa'].shape[1]
    ag_len = len(e['antigen']['aa'])
    if dm == 'per_residue' and darr is not None:
        d = np.zeros(L)
        ag_start = L - min(ag_len, L)
        n_copy = min(len(darr), L - ag_start)
        d[ag_start:ag_start+n_copy] = darr[:n_copy]
        batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
    elif dm == 'scalar' and darr is not None:
        batch['epitope_disorder'] = torch.tensor([[darr.mean()]], dtype=torch.float32)
    batch['mask_antigen'] = torch.zeros(1,L).bool()
    batch['mask_antigen'][0,-ag_len:] = True
    batch['mask'] = torch.ones(1,L).bool()
    return batch, e

def sample_cdr(model, batch, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        try:
            traj = model.sample(batch, sample_opt={'deterministic':False, 'num_recycles':1})
        except Exception as ex:
            return {'error':str(ex)}
    gen_flag = batch['generate_flag'][0].bool()
    if not gen_flag.any(): return {'error':'no gen'}
    logits = traj['pred_logits'][0, gen_flag]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    entropy = -(probs * np.log(probs+1e-8)).sum(axis=-1)
    pred_aa = logits.argmax(dim=-1).cpu().numpy()
    pred_seq = ''.join(AA[a] for a in pred_aa)
    has_6 = any(pred_seq[i:i+6]==pred_seq[i]*6 for i in range(len(pred_seq)-5))
    return {'pred_seq':pred_seq, 'entropy_mean':float(entropy.mean()), 'unique_aa':len(set(pred_seq)),
            'cdr_len':len(pred_seq), 'has_6mer':has_6}

all_results=[]
for label, samples in [('high_flex',test_high), ('low_flex',test_low)]:
    for sid, darr in samples:
        for dm in ['per_residue','none']:
            batch, entry = build_batch(sid, dm, darr)
            if batch is None: continue
            for seed in range(N_SEEDS):
                r = sample_cdr(model, batch, seed)
                r['sid']=sid; r['label']=label; r['mode']='A1_nodiv'; r['disorder_mode']=dm; r['seed']=seed
                r['ag_disorder_mean']=float(darr.mean()); r['ag_disorder_max']=float(darr.max())
                all_results.append(r)
env.close()

by_dm = defaultdict(list)
for r in all_results:
    if 'error' not in r: by_dm[(r['mode'],r['disorder_mode'])].append(r)

print('\\n=== A1 SPEARMAN ===')
for (mn, dm), res in sorted(by_dm.items()):
    vals = [(r['ag_disorder_max'],r['unique_aa'],r['entropy_mean']) for r in res]
    ag=np.array([v[0] for v in vals]); ua=np.array([v[1] for v in vals]); ent=np.array([v[2] for v in vals])
    r_ua,p_ua=spearmanr(ag,ua); r_ent,p_ent=spearmanr(ag,ent)
    print(f'{mn}_{dm} (n={len(vals)}): unique_aa r={r_ua:+.4f} p={p_ua:.4f} | entropy r={r_ent:+.4f} p={p_ent:.4f}')
    e=np.mean([r['entropy_mean'] for r in res]); u=np.mean([r['unique_aa'] for r in res])
    t=np.mean([r['top_aa_pct'] for r in res]) if 'top_aa_pct' in res[0] else 0
    n6=sum(1 for r in res if r.get('has_6mer'))
    print(f'  ent={e:.4f} uniq={u:.1f} 6mer={n6}')

ts=time.strftime('%Y%m%d_%H%M%S')
with open(f'idp_design_results/way4_ablate_a1_p2_{ts}.json','w') as f:
    json.dump({'probe':'V13.1 A1 no-diversity','models':['A1_nodiv'],'n_total':len(all_results),'results':all_results},f,indent=2)
print(f'Saved: idp_design_results/way4_ablate_a1_p2_{ts}.json')
" 2>&1 | tee -a "$LOG_BASE/a1_p2.log"

# ── A2: Remove anti_degen ──
echo "" | tee -a "$LOG_BASE/master.log"
echo "========== A2: NO ANTI_DEGEN ==========" | tee -a "$LOG_BASE/master.log"
echo "Started: $(date)" | tee -a "$LOG_BASE/master.log"

python train.py \
  logs/bfn_v20_ablate_a2_nodegen/bfn_v20_ablate_a2_nodegen.yml \
  --device cuda \
  --finetune "$V20_CKPT" \
  --tag v13_a2_nodegen \
  2>&1 | tee "$LOG_BASE/a2_train.log"

A2_CKPT=$(ls -t logs/bfn_v20_ablate_a2_nodegen_*/checkpoints/best.pt 2>/dev/null | head -1)
echo "A2 checkpoint: $A2_CKPT" | tee -a "$LOG_BASE/master.log"
echo "A2 done: $(date)" | tee -a "$LOG_BASE/master.log"

# A2 P2 probe (inline, same as above but with A2 label)
echo "Running A2 P2 probe..." | tee -a "$LOG_BASE/master.log"
python -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
import json, pickle, numpy as np, torch, yaml, lmdb, random, time
from collections import defaultdict
from scipy.stats import spearmanr
CKPT = '$A2_CKPT'
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = CKPT
yaml.dump(app, open('app_config.yaml','w',encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig
yaml.dump(app, open('app_config.yaml','w',encoding='utf-8'))
lookup = pickle.load(open('data/sabdab_disorder_lookup.pkl','rb'))
high_flex = [(k,v) for k,v in lookup.items() if v.max()>0.4]
low_flex = [(k,v) for k,v in lookup.items() if v.max()<0.2]
random.seed(42)
test_high = random.sample(high_flex, min(8, len(high_flex)))
test_low = random.sample(low_flex, min(8, len(low_flex)))
env_check = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
valid_ids = set(pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids','rb')))
env_check.close()
test_high = [(k,v) for k,v in test_high if k in valid_ids]
test_low = [(k,v) for k,v in test_low if k in valid_ids]
transform = get_transform([{'type':'mask_multiple_cdrs'},{'type':'merge_chains'},{'type':'patch_around_anchor'}])
env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
AA = 'ARNDCQEGHILKMFPSTWYV'
def build_batch(sid, dm, darr):
    with env.begin() as txn:
        raw = txn.get(sid.encode())
        if raw is None: return None, None
        e = pickle.loads(raw)
    if e.get('heavy') is None or e.get('antigen') is None: return None, None
    bd = transform(e)
    batch = recursive_to(PaddingCollate()([bd]), 'cpu')
    L = batch['aa'].shape[1]
    ag_len = len(e['antigen']['aa'])
    if dm == 'per_residue' and darr is not None:
        d = np.zeros(L)
        ag_start = L - min(ag_len, L)
        n_copy = min(len(darr), L - ag_start)
        d[ag_start:ag_start+n_copy] = darr[:n_copy]
        batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
    elif dm == 'scalar' and darr is not None:
        batch['epitope_disorder'] = torch.tensor([[darr.mean()]], dtype=torch.float32)
    batch['mask_antigen'] = torch.zeros(1,L).bool()
    batch['mask_antigen'][0,-ag_len:] = True
    batch['mask'] = torch.ones(1,L).bool()
    return batch, e
def sample_cdr(model, batch, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        try:
            traj = model.sample(batch, sample_opt={'deterministic':False, 'num_recycles':1})
        except Exception as ex:
            return {'error':str(ex)}
    gen_flag = batch['generate_flag'][0].bool()
    if not gen_flag.any(): return {'error':'no gen'}
    logits = traj['pred_logits'][0, gen_flag]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    entropy = -(probs * np.log(probs+1e-8)).sum(axis=-1)
    pred_aa = logits.argmax(dim=-1).cpu().numpy()
    pred_seq = ''.join(AA[a] for a in pred_aa)
    has_6 = any(pred_seq[i:i+6]==pred_seq[i]*6 for i in range(len(pred_seq)-5))
    return {'pred_seq':pred_seq, 'entropy_mean':float(entropy.mean()), 'unique_aa':len(set(pred_seq)),
            'cdr_len':len(pred_seq), 'has_6mer':has_6}
all_results=[]
for label, samples in [('high_flex',test_high), ('low_flex',test_low)]:
    for sid, darr in samples:
        for dm in ['per_residue','none']:
            batch, entry = build_batch(sid, dm, darr)
            if batch is None: continue
            for seed in range(3):
                r = sample_cdr(model, batch, seed)
                r['sid']=sid; r['label']=label; r['mode']='A2_nodegen'; r['disorder_mode']=dm; r['seed']=seed
                r['ag_disorder_mean']=float(darr.mean()); r['ag_disorder_max']=float(darr.max())
                all_results.append(r)
env.close()
by_dm = defaultdict(list)
for r in all_results:
    if 'error' not in r: by_dm[(r['mode'],r['disorder_mode'])].append(r)
print('\\n=== A2 SPEARMAN ===')
for (mn, dm), res in sorted(by_dm.items()):
    vals = [(r['ag_disorder_max'],r['unique_aa'],r['entropy_mean']) for r in res]
    ag=np.array([v[0] for v in vals]); ua=np.array([v[1] for v in vals]); ent=np.array([v[2] for v in vals])
    r_ua,p_ua=spearmanr(ag,ua); r_ent,p_ent=spearmanr(ag,ent)
    print(f'{mn}_{dm} (n={len(vals)}): unique_aa r={r_ua:+.4f} p={p_ua:.4f} | entropy r={r_ent:+.4f} p={p_ent:.4f}')
    e=np.mean([r['entropy_mean'] for r in res]); u=np.mean([r['unique_aa'] for r in res])
    n6=sum(1 for r in res if r.get('has_6mer'))
    print(f'  ent={e:.4f} uniq={u:.1f} 6mer={n6}')
ts=time.strftime('%Y%m%d_%H%M%S')
with open(f'idp_design_results/way4_ablate_a2_p2_{ts}.json','w') as f:
    json.dump({'probe':'V13.1 A2 no-anti_degen','models':['A2_nodegen'],'n_total':len(all_results),'results':all_results},f,indent=2)
print(f'Saved: idp_design_results/way4_ablate_a2_p2_{ts}.json')
" 2>&1 | tee -a "$LOG_BASE/a2_p2.log"

# ── A3: Head only ──
echo "" | tee -a "$LOG_BASE/master.log"
echo "========== A3: HEAD_SEQ ONLY ==========" | tee -a "$LOG_BASE/master.log"
echo "Started: $(date)" | tee -a "$LOG_BASE/master.log"

python train.py \
  logs/bfn_v20_ablate_a3_headonly/bfn_v20_ablate_a3_headonly.yml \
  --device cuda \
  --finetune "$V20_CKPT" \
  --tag v13_a3_headonly \
  2>&1 | tee "$LOG_BASE/a3_train.log"

A3_CKPT=$(ls -t logs/bfn_v20_ablate_a3_headonly_*/checkpoints/best.pt 2>/dev/null | head -1)
echo "A3 checkpoint: $A3_CKPT" | tee -a "$LOG_BASE/master.log"
echo "A3 done: $(date)" | tee -a "$LOG_BASE/master.log"

# A3 P2 probe
echo "Running A3 P2 probe..." | tee -a "$LOG_BASE/master.log"
python -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'modules')
import json, pickle, numpy as np, torch, yaml, lmdb, random, time
from collections import defaultdict
from scipy.stats import spearmanr
CKPT = '$A3_CKPT'
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = CKPT
yaml.dump(app, open('app_config.yaml','w',encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig
yaml.dump(app, open('app_config.yaml','w',encoding='utf-8'))
lookup = pickle.load(open('data/sabdab_disorder_lookup.pkl','rb'))
high_flex = [(k,v) for k,v in lookup.items() if v.max()>0.4]
low_flex = [(k,v) for k,v in lookup.items() if v.max()<0.2]
random.seed(42)
test_high = random.sample(high_flex, min(8, len(high_flex)))
test_low = random.sample(low_flex, min(8, len(low_flex)))
env_check = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
valid_ids = set(pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids','rb')))
env_check.close()
test_high = [(k,v) for k,v in test_high if k in valid_ids]
test_low = [(k,v) for k,v in test_low if k in valid_ids]
transform = get_transform([{'type':'mask_multiple_cdrs'},{'type':'merge_chains'},{'type':'patch_around_anchor'}])
env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
AA = 'ARNDCQEGHILKMFPSTWYV'
def build_batch(sid, dm, darr):
    with env.begin() as txn:
        raw = txn.get(sid.encode())
        if raw is None: return None, None
        e = pickle.loads(raw)
    if e.get('heavy') is None or e.get('antigen') is None: return None, None
    bd = transform(e)
    batch = recursive_to(PaddingCollate()([bd]), 'cpu')
    L = batch['aa'].shape[1]
    ag_len = len(e['antigen']['aa'])
    if dm == 'per_residue' and darr is not None:
        d = np.zeros(L)
        ag_start = L - min(ag_len, L)
        n_copy = min(len(darr), L - ag_start)
        d[ag_start:ag_start+n_copy] = darr[:n_copy]
        batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
    elif dm == 'scalar' and darr is not None:
        batch['epitope_disorder'] = torch.tensor([[darr.mean()]], dtype=torch.float32)
    batch['mask_antigen'] = torch.zeros(1,L).bool()
    batch['mask_antigen'][0,-ag_len:] = True
    batch['mask'] = torch.ones(1,L).bool()
    return batch, e
def sample_cdr(model, batch, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        try:
            traj = model.sample(batch, sample_opt={'deterministic':False, 'num_recycles':1})
        except Exception as ex:
            return {'error':str(ex)}
    gen_flag = batch['generate_flag'][0].bool()
    if not gen_flag.any(): return {'error':'no gen'}
    logits = traj['pred_logits'][0, gen_flag]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    entropy = -(probs * np.log(probs+1e-8)).sum(axis=-1)
    pred_aa = logits.argmax(dim=-1).cpu().numpy()
    pred_seq = ''.join(AA[a] for a in pred_aa)
    has_6 = any(pred_seq[i:i+6]==pred_seq[i]*6 for i in range(len(pred_seq)-5))
    return {'pred_seq':pred_seq, 'entropy_mean':float(entropy.mean()), 'unique_aa':len(set(pred_seq)),
            'cdr_len':len(pred_seq), 'has_6mer':has_6}
all_results=[]
for label, samples in [('high_flex',test_high), ('low_flex',test_low)]:
    for sid, darr in samples:
        for dm in ['per_residue','none']:
            batch, entry = build_batch(sid, dm, darr)
            if batch is None: continue
            for seed in range(3):
                r = sample_cdr(model, batch, seed)
                r['sid']=sid; r['label']=label; r['mode']='A3_headonly'; r['disorder_mode']=dm; r['seed']=seed
                r['ag_disorder_mean']=float(darr.mean()); r['ag_disorder_max']=float(darr.max())
                all_results.append(r)
env.close()
by_dm = defaultdict(list)
for r in all_results:
    if 'error' not in r: by_dm[(r['mode'],r['disorder_mode'])].append(r)
print('\\n=== A3 SPEARMAN ===')
for (mn, dm), res in sorted(by_dm.items()):
    vals = [(r['ag_disorder_max'],r['unique_aa'],r['entropy_mean']) for r in res]
    ag=np.array([v[0] for v in vals]); ua=np.array([v[1] for v in vals]); ent=np.array([v[2] for v in vals])
    r_ua,p_ua=spearmanr(ag,ua); r_ent,p_ent=spearmanr(ag,ent)
    print(f'{mn}_{dm} (n={len(vals)}): unique_aa r={r_ua:+.4f} p={p_ua:.4f} | entropy r={r_ent:+.4f} p={p_ent:.4f}')
    e=np.mean([r['entropy_mean'] for r in res]); u=np.mean([r['unique_aa'] for r in res])
    n6=sum(1 for r in res if r.get('has_6mer'))
    print(f'  ent={e:.4f} uniq={u:.1f} 6mer={n6}')
ts=time.strftime('%Y%m%d_%H%M%S')
with open(f'idp_design_results/way4_ablate_a3_p2_{ts}.json','w') as f:
    json.dump({'probe':'V13.1 A3 head-only','models':['A3_headonly'],'n_total':len(all_results),'results':all_results},f,indent=2)
print(f'Saved: idp_design_results/way4_ablate_a3_p2_{ts}.json')
" 2>&1 | tee -a "$LOG_BASE/a3_p2.log"

# ── Done ──
echo "" | tee -a "$LOG_BASE/master.log"
echo "============================================================" | tee -a "$LOG_BASE/master.log"
echo "V13.1 Ablation Sequence COMPLETE" | tee -a "$LOG_BASE/master.log"
echo "Finished at: $(date)" | tee -a "$LOG_BASE/master.log"
echo "A1: $A1_CKPT" | tee -a "$LOG_BASE/master.log"
echo "A2: $A2_CKPT" | tee -a "$LOG_BASE/master.log"
echo "A3: $A3_CKPT" | tee -a "$LOG_BASE/master.log"
echo "Logs: $LOG_BASE" | tee -a "$LOG_BASE/master.log"
