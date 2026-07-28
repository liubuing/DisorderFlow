import os
#!/usr/bin/env python3
"""WAY4 V7 P2 Mechanism Probe: disorder-conditioned CDR plasticity gradient.

Compares V7 (per-residue disorder, newly trained) vs V18 (scalar disorder).
Tests high-flex vs low-flex epitopes for CDR entropy gradient.
"""
import sys, os, pickle, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
import torch, yaml, lmdb, random
from collections import defaultdict

# Config
V7_CKPT = 'logs/bfn_v18_v7_per_residue_xpu_2026_07_02__00_48_49_v18_v7_run/checkpoints/best.pt'
V18_CKPT = 'logs/bfn_v18_disorder_cond_xpu_2026_07_01__15_51_01_v18_way4/checkpoints/best.pt'
V19_CKPT = 'logs/bfn_v19_diversity_xpu_2026_07_02__14_23_57_v19_final/checkpoints/best.pt'
V20_CKPT = 'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt'
PROFILES = 'data/idp_antigen_subset_v2/per_residue_profiles.pkl'
LOOKUP = 'data/sabdab_disorder_lookup.pkl'
OUT_DIR = 'idp_design_results'
N_SAMPLES = 10
N_SEEDS = 3
AA = 'ARNDCQEGHILKMFPSTWYV'

print("=" * 60)
print("WAY4 V7 P2: Mechanism Gradient Probe")
print("=" * 60)

# Load models
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.protein import constants

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig_ckpt = app['models']['bfn']['checkpoint']

models = {}
for name, ckpt in [('V19_per_residue', V19_CKPT), ('V20_per_residue', V20_CKPT),
                       ('V18_scalar', V18_CKPT)]:
    if not os.path.exists(ckpt):
        continue
    app['models']['bfn']['checkpoint'] = ckpt
    yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
    bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
    model, _ = bfn_loader.load_bfn('cpu')
    models[name] = model
    print("  Loaded %s" % name)

app['models']['bfn']['checkpoint'] = orig_ckpt
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))

# Load profiles
profiles = pickle.load(open(PROFILES, 'rb'))
lookup = pickle.load(open(LOOKUP, 'rb'))
print("  Profiles: %d antigens" % len(lookup))

# Select samples
high_flex = [(k, v) for k, v in lookup.items() if v.max() > 0.4]
low_flex = [(k, v) for k, v in lookup.items() if v.max() < 0.2]
random.seed(42)
test_high = random.sample(high_flex, min(N_SAMPLES, len(high_flex)))
test_low = random.sample(low_flex, min(N_SAMPLES, len(low_flex)))

# Filter to valid SAbDab IDs
env_check = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
valid_ids = set(pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids','rb')))
env_check.close()
test_high = [(k,v) for k,v in test_high if k in valid_ids]
test_low = [(k,v) for k,v in test_low if k in valid_ids]
print("  Test: %d high-flex, %d low-flex" % (len(test_high), len(test_low)))

# Open LMDB for batch building
env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
transform = get_transform([
    {'type': 'mask_multiple_cdrs'},
    {'type': 'merge_chains'},
    {'type': 'patch_around_anchor'}
])

def build_batch(sid, disorder_mode, disorder_arr):
    with env.begin() as txn:
        raw = txn.get(sid.encode())
        if raw is None: return None, None
        e = pickle.loads(raw)
    if e.get('heavy') is None or e.get('antigen') is None:
        return None, None
    batch_data = transform(e)
    batch = recursive_to(PaddingCollate()([batch_data]), 'cpu')
    L = batch['aa'].shape[1]
    ag_len = len(e['antigen']['aa'])
    if disorder_mode == 'per_residue' and disorder_arr is not None:
        d = np.zeros(L)
        ag_start = L - min(ag_len, L)
        n_copy = min(len(disorder_arr), L - ag_start)
        d[ag_start:ag_start + n_copy] = disorder_arr[:n_copy]
        batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
    elif disorder_mode == 'scalar' and disorder_arr is not None:
        batch['epitope_disorder'] = torch.tensor([[disorder_arr.mean()]], dtype=torch.float32)
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
    if not gen_flag.any():
        return {'error': 'no gen'}
    logits = traj['pred_logits'][0, gen_flag]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=-1)
    pred_aa = logits.argmax(dim=-1).cpu().numpy()
    pred_seq = ''.join(AA[a] for a in pred_aa)
    from collections import Counter
    cnt = Counter(pred_seq)
    top_pct = max(cnt.values()) / len(pred_seq) if pred_seq else 0
    has_6 = any(pred_seq[i:i+6] == pred_seq[i]*6 for i in range(len(pred_seq)-5))
    return {
        'pred_seq': pred_seq,
        'entropy_mean': float(entropy.mean()),
        'entropy_std': float(entropy.std()),
        'top_aa_pct': float(top_pct),
        'has_6mer': has_6,
        'unique_aa': len(set(pred_seq)),
        'cdr_len': len(pred_seq),
    }

# Run probe
print("\nRunning %d samples x %d models x %d seeds = ~%d generations..."
      % (len(test_high) + len(test_low), len(models), N_SEEDS,
         (len(test_high) + len(test_low)) * N_SEEDS * len(models)))

all_results = []
t0 = time.time()

for label, samples in [('high_flex', test_high), ('low_flex', test_low)]:
    for sid, disorder_arr in samples:
        for mode_name, model in models.items():
            # Determine disorder modes: per_residue/scalar + none (unconditional)
            if 'per_residue' in mode_name:
                d_modes = ['per_residue', 'none']
            else:
                d_modes = ['scalar', 'none']
            for dm in d_modes:
                batch, entry = build_batch(sid, dm, disorder_arr)
                if batch is None: continue
                for seed in range(N_SEEDS):
                    r = sample_cdr(model, batch, seed)
                    r['sid'] = sid; r['label'] = label
                    r['mode'] = mode_name; r['disorder_mode'] = dm; r['seed'] = seed
                    r['ag_disorder_mean'] = float(disorder_arr.mean())
                    r['ag_disorder_max'] = float(disorder_arr.max())
                    all_results.append(r)
                if 'error' not in all_results[-1]:
                    s = all_results[-1]
                    print("  %s/%s/%s/%s/s%d: ent=%.3f uniq=%d seq=%s"
                          % (label, sid, mode_name, dm, seed, s['entropy_mean'], s['unique_aa'], s['pred_seq'][:30]))

env.close()
elapsed = time.time() - t0
print("Done in %.0fs (%d generations)" % (elapsed, len(all_results)))

# Analysis
print("\n" + "=" * 60)
print("P2 Mechanism Gradient Analysis")
print("=" * 60)

# Split by model and disorder_mode
by_full = defaultdict(list)
for r in all_results:
    if 'error' not in r:
        by_full[(r['mode'], r.get('disorder_mode', ''), r['label'])].append(r)

# Also group by (mode, disorder_mode) for Spearman
by_model_dm = defaultdict(list)
for r in all_results:
    if 'error' not in r:
        by_model_dm[(r['mode'], r.get('disorder_mode', ''))].append(r)

print("\n%-30s %-12s %4s %8s %8s %8s %6s" % ('Mode', 'Label', 'N', 'Entropy', 'TopAA%', 'UniqAA', '6mer'))
print("-" * 80)
for (mode_name, dm), results_list in sorted(by_model_dm.items()):
    for label in ['high_flex', 'low_flex']:
        results = [r for r in results_list if r['label'] == label]
        if not results: continue
        e = [r['entropy_mean'] for r in results]
        t = [r['top_aa_pct'] for r in results]
        u = [r['unique_aa'] for r in results]
        n6 = sum(1 for r in results if r.get('has_6mer'))
        print("%-30s %-12s %4d %8.4f %8.3f %8.1f %5d"
              % ("%s_%s" % (mode_name, dm), label, len(results), np.mean(e), np.mean(t), np.mean(u), n6))

# Gradient
print("\n--- Gradient: high_flex - low_flex entropy ---")
for (mode_name, dm), results_list in sorted(by_model_dm.items()):
    hr = [r for r in results_list if r['label'] == 'high_flex']
    lr = [r for r in results_list if r['label'] == 'low_flex']
    if hr and lr:
        he = np.mean([r['entropy_mean'] for r in hr])
        le = np.mean([r['entropy_mean'] for r in lr])
        delta = he - le
        print("  %s_%s: high=%.4f low=%.4f delta=%+.4f" % (mode_name, dm, he, le, delta))

# ── Spearman: KEY METRICS ──
print("\n" + "=" * 60)
print("SPEARMAN CORRELATIONS (THE KEY V12 METRICS)")
print("=" * 60)
try:
    from scipy.stats import spearmanr
    for (mode_name, dm), results_list in sorted(by_model_dm.items()):
        vals = [(r['ag_disorder_max'], r['unique_aa'], r['entropy_mean']) for r in results_list]
        if len(vals) < 10: continue
        ag = np.array([v[0] for v in vals])
        ua = np.array([v[1] for v in vals])
        ent = np.array([v[2] for v in vals])
        r_ua, p_ua = spearmanr(ag, ua)
        r_ent, p_ent = spearmanr(ag, ent)
        print("\n  %s_%s (n=%d):" % (mode_name, dm, len(vals)))
        print("    unique_aa  vs ag_disorder_max: r=%+.4f  p=%.4f" % (r_ua, p_ua))
        print("    entropy    vs ag_disorder_max: r=%+.4f  p=%.4f" % (r_ent, p_ent))
except ImportError:
    print("  scipy not available, skipping Spearman")

# Mann-Whitney
print("\n--- Mann-Whitney U test ---")
try:
    from scipy.stats import mannwhitneyu
    for (mode_name, dm), results_list in sorted(by_model_dm.items()):
        hr = [r for r in results_list if r['label'] == 'high_flex']
        lr = [r for r in results_list if r['label'] == 'low_flex']
        if len(hr) >= 3 and len(lr) >= 3:
            he = [r['entropy_mean'] for r in hr]
            le = [r['entropy_mean'] for r in lr]
            stat, p = mannwhitneyu(he, le, alternative='greater')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print("  %s_%s: U=%.1f, p=%.4f %s" % (mode_name, dm, stat, p, sig))
except ImportError:
    print("  scipy not available, skipping")

# Save
os.makedirs(OUT_DIR, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
out_path = os.path.join(OUT_DIR, 'way4_v12_p2_mechanism_%s.json' % ts)
with open(out_path, 'w') as f:
    json.dump({
        'probe': 'WAY4 V12 P2 mechanism gradient — V20 real-training thesis test',
        'models': list(models.keys()),
        'disorder_modes': ['per_residue', 'scalar', 'none'],
        'n_samples': len(test_high) + len(test_low),
        'n_seeds': N_SEEDS,
        'n_total': len(all_results),
        'results': all_results,
    }, f, indent=2)
print("\nSaved: %s" % out_path)
print("P2 probe complete.")
