import os
#!/usr/bin/env python3
"""WAY4 V7 P0-fix-B probe: per-residue disorder profile vs scalar disorder.

Verifies that:
1. The receiver accepts per-residue (N, L) disorder profiles without NaN
2. Per-residue profiles produce CDR-level disorder gradient (not constant)
3. High-disorder epitope residues → higher entropy CDR output

Compares three modes:
  A. Per-residue disorder (V7): epitope_disorder_profile = (1, L) with per-residue values
  B. Scalar disorder (V6): epitope_disorder = (1, 1) global mean
  C. No disorder (V17i baseline): epitope_disorder = None
"""
import sys, os, pickle, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
import torch, yaml, random
from collections import defaultdict

CKPT = 'logs/bfn_v18_disorder_cond_xpu_2026_07_01__15_51_01_v18_way4/checkpoints/best.pt'
PROFILES = 'data/idp_antigen_subset_v2/per_residue_profiles.pkl'
OUT_DIR = 'idp_design_results'

print("=" * 60)
print("WAY4 V7 P0-fix-B: Per-Residue Disorder Profile Probe")
print("=" * 60)

# ── Load model ──────────────────────────────────────────────────────────
import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.protein import constants

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = CKPT
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
print("Model loaded")

# ── Load pre-computed per-residue profiles ──────────────────────────────
profiles = pickle.load(open(PROFILES, 'rb'))
print(f"Profiles loaded: {len(profiles['high'])}H, {len(profiles['mid'])}M, {len(profiles['low'])}L")

# ── Helper: build batch with per-residue disorder ───────────────────────
import lmdb
SABDAB_LMDB = 'data/sabdab_phase3_processed/train.lmdb'
SABDAB_IDS = 'data/sabdab_phase3_processed/train.lmdb-ids'

# Build SAbDab profile lookup by sample ID
sabdab_lookup = {}
for tier in ['high', 'mid', 'low']:
    for p in profiles[tier]:
        if p.get('source') == 'sabdab':
            sabdab_lookup[p['sid']] = p

print(f"SAbDab lookup: {len(sabdab_lookup)} entries")

def build_test_batch(sid, transform, disorder_mode='per_residue'):
    """Build a test batch for a given SAbDab sample.

    disorder_mode:
      - 'per_residue': (N, L) per-residue profile (V7)
      - 'scalar': (N, 1) mean disorder (V6)
      - 'none': no disorder (baseline)
    """
    env = lmdb.open(SABDAB_LMDB, subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        e = pickle.loads(txn.get(sid.encode()))
    env.close()

    if e.get('antigen') is None or e.get('heavy') is None:
        return None

    batch_data = transform(e)
    batch = recursive_to(PaddingCollate()([batch_data]), 'cpu')
    L = batch['aa'].shape[1]
    ag_len = len(e['antigen']['aa'])

    # Build disorder profile
    profile = sabdab_lookup.get(sid)
    if disorder_mode == 'per_residue' and profile is not None:
        # Per-residue: zero for antibody, disorder values for antigen
        disorder = np.zeros(L)
        ag_start = L - min(ag_len, L)
        ag_disorder = np.array(profile['disorder_per_res'])
        n_copy = min(len(ag_disorder), L - ag_start)
        disorder[ag_start:ag_start + n_copy] = ag_disorder[:n_copy]
        batch['epitope_disorder_profile'] = torch.tensor(disorder, dtype=torch.float32).unsqueeze(0)
    elif disorder_mode == 'scalar' and profile is not None:
        # Scalar mean disorder
        batch['epitope_disorder'] = torch.tensor([[profile['disorder_mean']]], dtype=torch.float32)
    # else: no disorder (both are None)

    batch['mask_antigen'] = torch.zeros(1, L).bool()
    batch['mask_antigen'][0, -ag_len:] = True
    batch['mask'] = torch.ones(1, L).bool()
    return batch, e


def sample_cdr(model, batch, mode_name):
    """Generate CDR sequence from a batch and return AA distribution stats."""
    with torch.no_grad():
        try:
            traj = model.sample(batch, sample_opt={
                'deterministic': False, 'num_recycles': 1,
                'disorder_guided': False,  # We're testing TRAINING-side disorder conditioning
            })
        except Exception as ex:
            return {'error': str(ex), 'mode': mode_name}

    pred_logits = traj['pred_logits']  # (N, L, 20)
    gen_flag = batch['generate_flag'][0].bool()
    if not gen_flag.any():
        return {'error': 'no generated residues', 'mode': mode_name}

    logits_cdr = pred_logits[0, gen_flag]  # (n_cdr, 20)
    probs_cdr = torch.softmax(logits_cdr, dim=-1).cpu().numpy()
    cdr_types = batch['aa'][0, gen_flag].cpu().numpy()

    # Entropy per position
    entropy = -(probs_cdr * np.log(probs_cdr + 1e-8)).sum(axis=-1)

    # Predicted AA
    pred_aa = logits_cdr.argmax(dim=-1).cpu().numpy()
    AA_MAP = 'ARNDCQEGHILKMFPSTWYV'
    pred_seq = ''.join(AA_MAP[a] for a in pred_aa)

    # De-generation check
    from collections import Counter
    aa_counts = Counter(pred_seq)
    top_aa_pct = max(aa_counts.values()) / len(pred_seq) if pred_seq else 0

    # 6-mer repeat check
    has_repeat = any(pred_seq[i:i+6] == pred_seq[i]*6 for i in range(len(pred_seq)-5))

    return {
        'mode': mode_name,
        'cdr_len': len(pred_seq),
        'pred_seq': pred_seq,
        'entropy_mean': float(entropy.mean()),
        'entropy_std': float(entropy.std()),
        'top_aa_pct': float(top_aa_pct),
        'has_6mer_repeat': has_repeat,
        'aa_distribution': dict(aa_counts.most_common(5)),
    }


# ── Run probe ───────────────────────────────────────────────────────────
print("\n--- Running probe: per-residue vs scalar vs none ---")

transform = get_transform([
    {'type': 'mask_multiple_cdrs'},
    {'type': 'merge_chains'},
    {'type': 'patch_around_anchor'}
])

# Select test samples: high-disorder, mid-disorder, low-disorder
# From SAbDab lookup, pick samples with varying max disorder
samples_by_disorder = sorted(sabdab_lookup.items(), key=lambda x: x[1]['disorder_max'], reverse=True)

test_samples = []
# Top 3 (highest max disorder)
for sid, p in samples_by_disorder[:3]:
    test_samples.append(('high_local_flex', sid, p))
# Bottom 3 (lowest max)
for sid, p in samples_by_disorder[-3:]:
    test_samples.append(('low_flex', sid, p))
# Middle 3
mid_idx = len(samples_by_disorder) // 2
for sid, p in samples_by_disorder[mid_idx:mid_idx + 3]:
    test_samples.append(('mid_flex', sid, p))

MODES = ['per_residue', 'scalar', 'none']
AA = 'ARNDCQEGHILKMFPSTWYV'

all_results = []

for label, sid, profile in test_samples:
    for mode in MODES:
        try:
            batch, entry = build_test_batch(sid, transform, mode)
            if batch is None:
                continue
            result = sample_cdr(model, batch, mode)
            result['sid'] = sid
            result['label'] = label
            result['ag_disorder_max'] = profile['disorder_max']
            result['ag_disorder_mean'] = profile['disorder_mean']
            all_results.append(result)
            status = f"ent={result.get('entropy_mean', 0):.3f}" if 'error' not in result else result['error']
            print(f"  {label}/{sid}/{mode}: {status}")
        except Exception as ex:
            print(f"  {label}/{sid}/{mode}: EXCEPTION {ex}")
            all_results.append({'sid': sid, 'label': label, 'mode': mode, 'error': str(ex)})

# ── Analyze results ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Analysis: Per-Residue vs Scalar Disorder Conditioning")
print(f"{'='*60}")

# Group by mode
by_mode = defaultdict(list)
for r in all_results:
    if 'error' not in r:
        by_mode[r['mode']].append(r)

for mode in MODES:
    results = by_mode.get(mode, [])
    if not results:
        print(f"\n{mode}: no results")
        continue
    entropies = [r['entropy_mean'] for r in results]
    top_aa = [r['top_aa_pct'] for r in results]
    repeats = sum(1 for r in results if r.get('has_6mer_repeat'))
    print(f"\n{mode} ({len(results)} samples):")
    print(f"  Entropy:  {np.mean(entropies):.4f} +/- {np.std(entropies):.4f}")
    print(f"  Top AA%:  {np.mean(top_aa):.3f} +/- {np.std(top_aa):.3f}")
    print(f"  6mer repeats: {repeats}/{len(results)}")

    # Show sample CDRs
    for r in results[:3]:
        print(f"    [{r['label']}] {r['sid']}: {r.get('pred_seq', 'N/A')[:40]}")

# Key check: does per_residue mode show entropy gradient vs disorder?
print(f"\n--- Entropy-Disorder Gradient Check ---")
per_res = [r for r in all_results if r.get('mode') == 'per_residue' and 'error' not in r]
scalar_res = [r for r in all_results if r.get('mode') == 'scalar' and 'error' not in r]
none_res = [r for r in all_results if r.get('mode') == 'none' and 'error' not in r]

if per_res and scalar_res:
    # Check: per_residue entropy should be DIFFERENT from scalar (not degenerate)
    pr_entropies = [r['entropy_mean'] for r in per_res]
    sc_entropies = [r['entropy_mean'] for r in scalar_res]

    # Correlate entropy with ag_disorder_max within per_residue mode
    pr_max_disorder = [r['ag_disorder_max'] for r in per_res]
    if len(pr_max_disorder) >= 3:
        # Simple correlation
        corr = np.corrcoef(pr_entropies, pr_max_disorder)[0, 1]
        print(f"  Per-residue: entropy vs ag_disorder_max r = {corr:.3f}")
    else:
        print(f"  Per-residue: too few samples for correlation ({len(per_res)})")

    # Mean entropy difference
    print(f"  Per-residue entropy:  {np.mean(pr_entropies):.4f} +/- {np.std(pr_entropies):.4f}")
    if sc_entropies:
        print(f"  Scalar entropy:       {np.mean(sc_entropies):.4f} +/- {np.std(sc_entropies):.4f}")
    if none_res:
        no_entropies = [r['entropy_mean'] for r in none_res]
        print(f"  No-disorder entropy:  {np.mean(no_entropies):.4f} +/- {np.std(no_entropies):.4f}")

# ── Save ────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
out_path = os.path.join(OUT_DIR, f'way4_v7_p0fixb_probe_{ts}.json')
with open(out_path, 'w') as f:
    json.dump({
        'probe': 'P0-fix-B per-residue disorder profile verification',
        'n_samples': len(all_results),
        'modes_tested': MODES,
        'results': all_results,
    }, f, indent=2)
print(f"\nSaved: {out_path}")
print("P0-fix-B probe complete.")
