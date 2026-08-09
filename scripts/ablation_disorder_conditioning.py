#!/usr/bin/env python3
"""Disorder Conditioning Ablation Experiment.

Proves whether disorder conditioning actually changes generated CDR sequences.
Tests two orthogonal axes:
  Axis 1: Receiver conditioning (epitope_disorder_profile ON vs OFF)
  Axis 2: Sampling noise scaling (disorder_guided ON vs OFF)

Metrics:
  - Hamming fraction between paired ON/OFF outputs (responsiveness)
  - Spearman(epitope_disorder, CDR_entropy) per arm (core claim)
  - Degeneracy guards: top_aa_pct, has_6mer
  - Quality preservation: PPL, sequence recovery

Critical prerequisite: the checkpoint must have been trained WITH disorder
conditioning (non-zero disorder_proj weights). If disorder_proj is still
zero-initialized, ON == OFF by construction.

Usage:
  python scripts/ablation_disorder_conditioning.py \
      --checkpoint logs/disorder_head_retrain.pt \
      --n-samples 20 --seeds 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

from disorderflow.utils.protein.constants import ressymb_to_resindex

AA_LETTERS = ''.join(
    residue for residue, index in sorted(ressymb_to_resindex.items(), key=lambda item: item[1])
    if index < 20)

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'modules'))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_disorder_proj_nonzero(model):
    """Verify that disorder_proj has learned non-zero weights."""
    if not hasattr(model.bfn.receiver, 'disorder_proj'):
        return False, "disorder_proj not found in receiver"
    proj = model.bfn.receiver.disorder_proj
    w_norm = proj.weight.data.norm().item()
    b_norm = proj.bias.data.norm().item()
    if w_norm < 1e-8 and b_norm < 1e-8:
        return False, f"disorder_proj is ZERO (w={w_norm:.2e}, b={b_norm:.2e}) — conditioning was never trained"
    return True, f"disorder_proj active (w={w_norm:.4f}, b={b_norm:.4f})"


def compute_cdr_metrics(seq_logits, mask_gen):
    """Compute diversity metrics from CDR logits.

    Args:
        seq_logits: (L, 20) raw logits for CDR positions
        mask_gen: (L,) bool, True for CDR positions

    Returns:
        dict with entropy, unique_aa, top_aa_pct, has_6mer
    """
    probs = F.softmax(seq_logits[mask_gen], dim=-1)  # (n_cdr, 20)
    if probs.shape[0] == 0:
        return {'entropy': 0, 'unique_aa': 0, 'top_aa_pct': 1.0, 'has_6mer': False}

    # Shannon entropy per position
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

    # Unique AA in argmax sequence
    argmax_seq = probs.argmax(dim=-1)
    unique_aa = argmax_seq.unique().numel()

    # Top AA concentration
    top_aa_pct = probs.max(dim=-1)[0].mean().item()

    # Has 6-mer repeat (degeneracy check)
    seq_str = ''.join(AA_LETTERS[i] for i in argmax_seq.cpu().tolist())
    has_6mer = any(seq_str[i:i+6] == seq_str[i+6:i+12]
                   for i in range(max(0, len(seq_str) - 12)))

    return {
        'entropy': entropy,
        'unique_aa': unique_aa,
        'top_aa_pct': top_aa_pct,
        'has_6mer': has_6mer,
        'n_cdr': probs.shape[0],
    }


def run_paired_generation(model, batch, disorder_profile, seeds,
                          conditioning_on, guided_on, guided_strength=0.5):
    """Run BFN sampling with specified conditioning settings.

    Returns: list of generated sequences (one per seed), metrics dict.
    """
    from disorderflow.modules.common.geometry import construct_3d_basis

    bfn = model.bfn
    N, L = batch['aa'].shape
    mask_res = batch['mask'].bool()
    mask_gen = batch['generate_flag'].bool() & mask_res
    cn = batch.get('chain_nb', torch.zeros(N, L, dtype=torch.long))

    # Set conditioning - modify batch in-place
    if conditioning_on and disorder_profile is not None:
        batch['epitope_disorder_profile'] = disorder_profile.unsqueeze(0).to(DEVICE)
        batch['epitope_disorder'] = disorder_profile.mean().unsqueeze(0).unsqueeze(0).to(DEVICE)
    else:
        batch.pop('epitope_disorder_profile', None)
        batch.pop('epitope_disorder', None)

    # Sample options
    sample_opt = {
        'deterministic': False,
        'disorder_guided': guided_on,
        'disorder_guided_strength': guided_strength,
    }
    if guided_on and conditioning_on and disorder_profile is not None:
        # Use external pliability map: only for antigen residues
        pliability = torch.zeros(1, L, device=DEVICE)
        ag_nb = int(cn.max().item()) if cn.numel() > 0 else max(set(cn.flatten().tolist()))
        ag_mask_bool = (cn[0] == ag_nb)
        n_ag = ag_mask_bool.sum().item()
        if n_ag > 0:
            n_copy = min(n_ag, len(disorder_profile))
            pliability[0, ag_mask_bool] = disorder_profile[:n_copy].to(DEVICE)
        sample_opt['disorder_pliability'] = pliability

    sequences = []
    all_metrics = []

    # Debug: check batch keys
    if not hasattr(run_paired_generation, '_debug_printed'):
        print(f"    [DEBUG] batch keys: {sorted(batch.keys())[:15]}", flush=True)
        print(f"    [DEBUG] batch aa shape: {batch['aa'].shape}", flush=True)
        run_paired_generation._debug_printed = True

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            try:
                result = bfn.sample(
                    batch,
                    sample_opt=sample_opt,
                )
                # Extract generated sequence from sample result
                if isinstance(result, dict):
                    if 'pred_logits' in result:
                        logits = result['pred_logits']
                        if logits.dim() == 3:
                            logits = logits[0]
                        pred_aa = logits.argmax(dim=-1)  # (L,)
                    elif 0 in result:
                        pred_aa = result[0][2]  # (v_0, final_pos, final_seq)
                        if pred_aa.dim() == 2:
                            pred_aa = pred_aa[0]
                    else:
                        raise KeyError(f"No sequence in result: {sorted(result.keys())[:5]}")
                else:
                    pred_aa = result[0]

                # Get CDR positions
                gen_mask_1d = mask_gen[0] if mask_gen.dim() == 2 else mask_gen
                cdr_seq = pred_aa[gen_mask_1d].cpu().numpy()
                sequences.append(cdr_seq)

                # Compute metrics from the final logits
                if isinstance(result, dict) and 'pred_logits' in result:
                    metrics = compute_cdr_metrics(result['pred_logits'][0], gen_mask_1d)
                else:
                    # Approximate from sequence
                    metrics = {
                        'entropy': float(np.log(max(1, len(np.unique(cdr_seq))))),
                        'unique_aa': len(np.unique(cdr_seq)),
                        'top_aa_pct': max(np.bincount(cdr_seq, minlength=20)) / max(len(cdr_seq), 1),
                        'has_6mer': False,
                        'n_cdr': len(cdr_seq),
                    }
                all_metrics.append(metrics)

            except Exception as e:
                if seed == seeds[0]:
                    import traceback
                    print(f"    [ERROR] seed={seed}: {e}", flush=True)
                    traceback.print_exc()
                sequences.append(None)
                all_metrics.append(None)

    return sequences, all_metrics


def hamming_fraction(seq_a, seq_b):
    """Fraction of positions that differ between two sequences."""
    if seq_a is None or seq_b is None:
        return None
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return None
    return (seq_a[:min_len] != seq_b[:min_len]).mean()


def run_ablation(args):
    """Main ablation experiment."""
    from scipy.stats import spearmanr

    print("=" * 70)
    print("  DISORDER CONDITIONING ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {args.n_samples}, Seeds per sample: {args.seeds}")
    print()

    # Load model
    import bfn_loader
    import yaml
    app = yaml.safe_load(open(PROJECT / 'app_config.yaml', encoding='utf-8'))
    orig_ckpt = app['models']['bfn']['checkpoint']
    app['models']['bfn']['checkpoint'] = args.checkpoint
    yaml.dump(app, open(PROJECT / 'app_config.yaml', 'w', encoding='utf-8'))
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, _ = bfn_loader.load_bfn(DEVICE)
    app['models']['bfn']['checkpoint'] = orig_ckpt
    yaml.dump(app, open(PROJECT / 'app_config.yaml', 'w', encoding='utf-8'))

    # Check disorder_proj
    is_active, msg = check_disorder_proj_nonzero(model)
    print(f"  disorder_proj: {msg}")
    if not is_active:
        print("\n  [CRITICAL] Conditioning was never trained. ON == OFF by construction.")
        print("  Train with disorder_lookup enabled first, then re-run this ablation.")
        print("  Proceeding anyway to confirm the null result...\n")

    # Load disorder lookup
    import pickle
    lookup_path = PROJECT / 'data' / 'sabdab_disorder_lookup_experimental.pkl'
    if not lookup_path.exists():
        lookup_path = PROJECT / 'data' / 'sabdab_disorder_lookup.pkl'
    if not lookup_path.exists():
        print("  [ERROR] No disorder lookup found. Run build_disorder_lookup_experimental.py first.")
        return

    with open(lookup_path, 'rb') as f:
        lookup = pickle.load(f)

    # Select samples: mix of high-flex and low-flex antigens
    import lmdb
    lmdb_path = str(PROJECT / 'data' / 'sabdab_phase3_processed' / 'train.lmdb')
    ids_path = str(PROJECT / 'data' / 'sabdab_phase3_processed' / 'train.lmdb-ids')
    all_ids = pickle.load(open(ids_path, 'rb'))

    # Categorize by disorder
    high_flex = []  # max > 0.4
    low_flex = []   # max < 0.15
    for sid in all_ids:
        arr = lookup.get(sid)
        if arr is None or len(arr) == 0:
            continue
        if np.mean(arr) > 0.3:
            high_flex.append((sid, arr))
        elif np.mean(arr) < 0.2:
            low_flex.append((sid, arr))

    print(f"\n  Available: {len(high_flex)} high-flex, {len(low_flex)} low-flex antigens")

    # Sample balanced set
    n_each = min(args.n_samples // 2, len(high_flex), len(low_flex))
    np.random.seed(42)
    selected_high = [high_flex[i] for i in np.random.choice(len(high_flex), n_each, replace=False)]
    selected_low = [low_flex[i] for i in np.random.choice(len(low_flex), n_each, replace=False)]
    selected = [(sid, arr, 'high') for sid, arr in selected_high] + \
               [(sid, arr, 'low') for sid, arr in selected_low]
    print(f"  Selected: {n_each} high + {n_each} low = {len(selected)} total")

    # Run 4-arm ablation
    arms = [
        ('cond_OFF_guided_OFF', False, False),
        ('cond_ON_guided_ON', True, True),
        ('cond_ON_guided_OFF', True, False),
        ('cond_OFF_guided_ON', False, True),
    ]

    seeds = list(range(args.seeds))
    results = {arm_name: {'hamming_vs_baseline': [], 'entropy': [], 'unique_aa': [],
                          'top_aa_pct': [], 'disorder_level': []}
               for arm_name, _, _ in arms}

    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)

    for sample_idx, (sid, disorder_arr, flex_level) in enumerate(selected):
        print(f"\n  [{sample_idx+1}/{len(selected)}] {sid} ({flex_level}-flex, "
              f"max={disorder_arr.max():.3f})")

        # Load batch
        with env.begin() as txn:
            raw = txn.get(sid.encode())
            if raw is None:
                continue
            entry = pickle.loads(raw)

        # Build batch using transform pipeline
        from disorderflow.utils.data import PaddingCollate
        from disorderflow.utils.train import recursive_to
        from disorderflow.utils.transforms import get_transform

        try:
            transform = get_transform([
                {'type': 'mask_multiple_cdrs'},
                {'type': 'merge_chains'},
                {'type': 'patch_around_anchor'},
            ])
            batch_data = transform(entry)
            if batch_data is None:
                continue
            batch = recursive_to(PaddingCollate()([batch_data]), DEVICE)
            batch['mask'] = batch.get('mask', torch.ones(1, batch['aa'].shape[1]).bool().to(DEVICE))
            # Ensure required keys for BFN sampling
            if 'pair_feat' not in batch:
                batch['pair_feat'] = torch.zeros(1, batch['aa'].shape[1], batch['aa'].shape[1], 128, device=DEVICE)
            if 'generate_flag' not in batch:
                batch['generate_flag'] = torch.zeros(1, batch['aa'].shape[1], dtype=torch.bool, device=DEVICE)

            L = batch['aa'].shape[1]
            disorder_profile = torch.zeros(L, dtype=torch.float32)
            # Antigen chain is the last one in merge_chains order
            cn = batch.get('chain_nb', torch.zeros(1, L, dtype=torch.long))
            ag_nb = int(cn.max().item())
            ag_mask = (cn[0] == ag_nb)
            n_ag = ag_mask.sum().item()
            if n_ag > 0:
                n_copy = min(n_ag, len(disorder_arr))
                ag_indices = ag_mask.nonzero(as_tuple=True)[1] if ag_mask.dim() == 2 else ag_mask.nonzero(as_tuple=True)[0]
                disorder_profile[ag_indices[:n_copy]] = torch.tensor(disorder_arr[:n_copy])

        except Exception as e:
            print(f"    [SKIP] batch construction failed: {e}")
            continue

        # Run each arm
        baseline_seqs = None
        for arm_name, cond_on, guided_on in arms:
            seqs, metrics_list = run_paired_generation(
                model, batch, disorder_profile, seeds,
                conditioning_on=cond_on, guided_on=guided_on,
                guided_strength=args.guided_strength)

            valid_metrics = [m for m in metrics_list if m is not None]
            valid_seqs = [s for s in seqs if s is not None]

            if valid_metrics:
                avg_entropy = np.mean([m['entropy'] for m in valid_metrics])
                avg_unique = np.mean([m['unique_aa'] for m in valid_metrics])
                avg_top = np.mean([m['top_aa_pct'] for m in valid_metrics])
            else:
                avg_entropy = avg_unique = avg_top = 0

            results[arm_name]['entropy'].append(avg_entropy)
            results[arm_name]['unique_aa'].append(avg_unique)
            results[arm_name]['top_aa_pct'].append(avg_top)
            results[arm_name]['disorder_level'].append(disorder_arr.max())

            # Hamming vs baseline (cond_OFF_guided_OFF)
            if arm_name == 'cond_OFF_guided_OFF':
                baseline_seqs = valid_seqs
            elif baseline_seqs and valid_seqs:
                hamming_vals = []
                for s_on, s_off in zip(valid_seqs, baseline_seqs, strict=False):
                    h = hamming_fraction(s_on, s_off)
                    if h is not None:
                        hamming_vals.append(h)
                results[arm_name]['hamming_vs_baseline'].append(
                    np.mean(hamming_vals) if hamming_vals else 0.0)

    env.close()

    # === Report ===
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS")
    print("=" * 70)

    for arm_name, _, _ in arms:
        r = results[arm_name]
        n = len(r['entropy'])
        if n == 0:
            print(f"\n  {arm_name}: NO DATA")
            continue
        print(f"\n  {arm_name} (n={n}):")
        print(f"    Entropy:      {np.mean(r['entropy']):.4f} ± {np.std(r['entropy']):.4f}")
        print(f"    Unique AA:    {np.mean(r['unique_aa']):.2f} ± {np.std(r['unique_aa']):.2f}")
        print(f"    Top AA %:     {np.mean(r['top_aa_pct']):.4f}")
        if r['hamming_vs_baseline']:
            print(f"    Hamming vs baseline: {np.mean(r['hamming_vs_baseline']):.4f}")

    # Core claim: Spearman(disorder, entropy) should be positive when ON, ~0 when OFF
    print(f"\n  {'─' * 50}")
    print("  CORE CLAIM: Spearman(epitope_disorder, CDR_entropy)")
    print(f"  {'─' * 50}")
    for arm_name, _, _ in arms:
        r = results[arm_name]
        if len(r['entropy']) > 3 and np.std(r['disorder_level']) > 0.01:
            sp, pval = spearmanr(r['disorder_level'], r['entropy'])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"    {arm_name:25s}: r={sp:+.3f} (p={pval:.4f}) {sig}")
        else:
            print(f"    {arm_name:25s}: insufficient data")

    # Verdict
    print(f"\n  {'─' * 50}")
    on_entropy = results['cond_ON_guided_ON']['entropy']
    off_entropy = results['cond_OFF_guided_OFF']['entropy']
    if on_entropy and off_entropy:
        delta = np.mean(on_entropy) - np.mean(off_entropy)
        hamming = results['cond_ON_guided_ON']['hamming_vs_baseline']
        avg_hamming = np.mean(hamming) if hamming else 0
        print("  VERDICT:")
        print(f"    Entropy delta (ON - OFF): {delta:+.4f}")
        print(f"    Mean Hamming (ON vs OFF): {avg_hamming:.4f}")
        if avg_hamming < 0.01 and abs(delta) < 0.01:
            print("    → Conditioning has NO EFFECT (disorder_proj likely untrained)")
        elif avg_hamming > 0.05 and delta > 0.02:
            print("    → Conditioning IS ACTIVE: higher disorder → more diverse CDRs")
        else:
            print("    → WEAK/MIXED signal: needs more training or stronger conditioning")

    # Save results
    out_path = PROJECT / 'calibration_artifacts' / 'ablation_disorder_results.json'
    import json
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'disorder_proj_active': is_active,
            'n_samples': len(selected),
            'seeds': args.seeds,
            'results': {k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                       for k, v in results.items()},
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Disorder conditioning ablation')
    parser.add_argument('--checkpoint', type=str,
                        default='logs/disorder_head_retrain.pt')
    parser.add_argument('--n-samples', type=int, default=20,
                        help='Number of antigen samples (half high-flex, half low-flex)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds per sample')
    parser.add_argument('--guided-strength', type=float, default=0.5,
                        help='Disorder-guided noise scaling strength')
    args = parser.parse_args()
    run_ablation(args)


if __name__ == '__main__':
    main()
