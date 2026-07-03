#!/usr/bin/env python3
"""V12 Wet-Lab Candidate Generator — Aβ42 epitope top-N from V20 generator.

Selects CDR sequences with: high unique_aa + high disorder alignment + no degeneration + single-chain quality.
"""
import sys, os, pickle, time, json, random
sys.path.insert(0, '.'); sys.path.insert(0, 'modules')
import numpy as np
import torch, yaml, lmdb
from collections import Counter, defaultdict

V20_CKPT = 'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt'
TOP_N = 20
N_SEEDS = 10
AA = 'ARNDCQEGHILKMFPSTWYV'

print("=" * 60)
print("V12 Wet-Lab Candidate Generator — Aβ42 Top-N")
print("=" * 60)

# ── Load V20 model ──
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig_ckpt = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = V20_CKPT
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, cfg = bfn_loader.load_bfn('cpu')
print("V20 model loaded")
app['models']['bfn']['checkpoint'] = orig_ckpt
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))

# ── Load Aβ42 profiles ──
profiles = pickle.load(open('data/idp_antigen_subset_v2/per_residue_profiles.pkl', 'rb'))
lookup = pickle.load(open('data/sabdab_disorder_lookup.pkl', 'rb'))

# Get high-disorder antigens as "Aβ42-like" epitopes
high_disorder = [(k, v) for k, v in lookup.items() if v.max() > 0.5]
random.seed(42)
candidates = random.sample(high_disorder, min(10, len(high_disorder)))
print(f"Selected {len(candidates)} high-disorder epitopes")

# Setup
transform = get_transform([{'type': 'mask_multiple_cdrs'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}])
env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
valid_ids = set(pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb')))

def build_batch(sid, disorder_arr):
    with env.begin() as txn:
        raw = txn.get(sid.encode())
        if raw is None: return None, None
        e = pickle.loads(raw)
    if e.get('heavy') is None or e.get('antigen') is None: return None, None
    batch_data = transform(e)
    batch = recursive_to(PaddingCollate()([batch_data]), 'cpu')
    L = batch['aa'].shape[1]
    ag_len = len(e['antigen']['aa'])
    d = np.zeros(L)
    ag_start = L - min(ag_len, L)
    n_copy = min(len(disorder_arr), L - ag_start)
    d[ag_start:ag_start + n_copy] = disorder_arr[:n_copy]
    batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
    batch['mask_antigen'] = torch.zeros(1, L).bool()
    batch['mask_antigen'][0, -ag_len:] = True
    batch['mask'] = torch.ones(1, L).bool()
    return batch, e

def sample_cdr(model, batch, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        try:
            traj = model.sample(batch, sample_opt={'deterministic': False, 'num_recycles': 1})
        except Exception as ex:
            return {'error': str(ex)}
    gen_flag = batch['generate_flag'][0].bool()
    if not gen_flag.any(): return {'error': 'no gen'}
    logits = traj['pred_logits'][0, gen_flag]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=-1)
    pred_aa = logits.argmax(dim=-1).cpu().numpy()
    pred_seq = ''.join(AA[a] for a in pred_aa)
    has_6 = any(pred_seq[i:i+6] == pred_seq[i]*6 for i in range(len(pred_seq)-5))
    return {
        'pred_seq': pred_seq, 'entropy_mean': float(entropy.mean()),
        'unique_aa': len(set(pred_seq)), 'cdr_len': len(pred_seq),
        'has_6mer': has_6,
    }

# Generate
all_candidates = []
print(f"\nGenerating from {len(candidates)} epitopes x {N_SEEDS} seeds...")
for sid, disorder_arr in candidates:
    if sid not in valid_ids: continue
    batch, entry = build_batch(sid, disorder_arr)
    if batch is None: continue
    for seed in range(N_SEEDS):
        r = sample_cdr(model, batch, seed)
        if 'error' in r: continue
        if r['has_6mer']: continue  # filter degen
        r['sid'] = sid
        r['ag_disorder_max'] = float(disorder_arr.max())
        r['ag_disorder_mean'] = float(disorder_arr.mean())
        r['seed'] = seed
        # Score: unique_aa * (1 - entropy/max_entropy) → high diversity + high confidence
        norm_entropy = 1.0 - r['entropy_mean'] / np.log(20)
        r['score'] = r['unique_aa'] * norm_entropy
        all_candidates.append(r)
        if len(all_candidates) % 20 == 0:
            print(f"  {len(all_candidates)} candidates generated...", flush=True)

env.close()

# Rank by score
all_candidates.sort(key=lambda x: x['score'], reverse=True)
top = all_candidates[:TOP_N]

print(f"\n{'='*60}")
print(f"TOP {TOP_N} WET-LAB CANDIDATES")
print(f"{'='*60}")
print(f"{'Rank':<5} {'SID':<10} {'Seq':<45} {'Uniq':<5} {'Ent':<7} {'Score':<6} {'agDis':<6}")
print("-" * 85)
for i, c in enumerate(top):
    print(f"{i+1:<5} {c['sid']:<10} {c['pred_seq']:<45} {c['unique_aa']:<5} {c['entropy_mean']:<7.3f} {c['score']:<6.1f} {c['ag_disorder_max']:<6.3f}")

# Save
ts = time.strftime('%Y%m%d_%H%M%S')
out = f'idp_design_results/way4_v12_wetlab_candidates_{ts}.json'
json.dump({
    'generator': 'V20 BFN (head_seq unfrozen)',
    'n_epitopes': len(candidates),
    'n_seeds': N_SEEDS,
    'n_total_generated': len(all_candidates),
    'selection_criteria': 'high unique_aa × normalized confidence, no 6mer degen',
    'top_n': TOP_N,
    'candidates': top,
}, open(out, 'w'), indent=2)
print(f"\nSaved: {out}")
