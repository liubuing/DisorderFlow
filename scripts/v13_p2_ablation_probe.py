#!/usr/bin/env python3
"""V13.1 P2 Ablation Probe — reusable for any ablation checkpoint.

Usage: python scripts/v13_p2_ablation_probe.py <checkpoint_path> <label> [--n-samples 8]
Output: idp_design_results/way4_ablate_{label}_p2_{ts}.json

Reports: Spearman r (unique_aa vs disorder), entropy, 6mer, summary stats.
"""
import sys, os, pickle, time, json, argparse, random
sys.path.insert(0, '.'); sys.path.insert(0, 'modules')
import numpy as np
import torch, yaml, lmdb
from collections import defaultdict

AA = 'ARNDCQEGHILKMFPSTWYV'

def load_model(ckpt_path):
    import bfn_loader
    app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
    orig = app['models']['bfn']['checkpoint']
    app['models']['bfn']['checkpoint'] = ckpt_path
    yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
    bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
    model, _ = bfn_loader.load_bfn('cpu')
    app['models']['bfn']['checkpoint'] = orig
    yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
    return model

def run_probe(model, n_samples=8, n_seeds=3):
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.transforms import get_transform

    lookup = pickle.load(open('data/sabdab_disorder_lookup.pkl', 'rb'))
    high_flex = [(k, v) for k, v in lookup.items() if v.max() > 0.4]
    low_flex = [(k, v) for k, v in lookup.items() if v.max() < 0.2]
    random.seed(42)
    test_high = random.sample(high_flex, min(n_samples, len(high_flex)))
    test_low = random.sample(low_flex, min(n_samples, len(low_flex)))

    env_check = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)
    valid_ids = set(pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb')))
    env_check.close()
    test_high = [(k, v) for k, v in test_high if k in valid_ids]
    test_low = [(k, v) for k, v in test_low if k in valid_ids]
    print(f"Test: {len(test_high)}H, {len(test_low)}L", flush=True)

    transform = get_transform([{'type': 'mask_multiple_cdrs'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}])
    env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False, readonly=True, lock=False, readahead=False)

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
            d[ag_start:ag_start + n_copy] = darr[:n_copy]
            batch['epitope_disorder_profile'] = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
        elif dm == 'scalar' and darr is not None:
            batch['epitope_disorder'] = torch.tensor([[darr.mean()]], dtype=torch.float32)
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
        cnt = __import__('collections').Counter(pred_seq)
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

    all_results = []
    total = 0
    for label, samples in [('high_flex', test_high), ('low_flex', test_low)]:
        for sid, darr in samples:
            for dm in ['per_residue', 'none']:
                batch, entry = build_batch(sid, dm, darr)
                if batch is None: continue
                for seed in range(n_seeds):
                    r = sample_cdr(model, batch, seed)
                    r['sid'] = sid; r['label'] = label; r['disorder_mode'] = dm; r['seed'] = seed
                    r['ag_disorder_mean'] = float(darr.mean()); r['ag_disorder_max'] = float(darr.max())
                    all_results.append(r)
                    total += 1
                if 'error' not in all_results[-1]:
                    s = all_results[-1]
                    print(f"[{total}] {label}/{sid}/{dm}/s{seed}: ent={s['entropy_mean']:.3f} uniq={s['unique_aa']}", flush=True)

    env.close()
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V13.1 Ablation P2 Probe')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('label', type=str, help='Ablation label (e.g. A1_nodiv)')
    parser.add_argument('--n-samples', type=int, default=8, help='Samples per flex group')
    parser.add_argument('--n-seeds', type=int, default=3, help='Seeds per sample')
    args = parser.parse_args()

    print(f"V13.1 P2 Probe: {args.label}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint)
    print("Model loaded", flush=True)
    results = run_probe(model, n_samples=args.n_samples, n_seeds=args.n_seeds)

    # Spearman analysis
    print("\n" + "="*60)
    print(f"SPEARMAN: {args.label}")
    print("="*60)

    from scipy.stats import spearmanr
    by_dm = defaultdict(list)
    for r in results:
        if 'error' not in r:
            by_dm[r['disorder_mode']].append(r)

    for dm, res_list in sorted(by_dm.items()):
        vals = [(r['ag_disorder_max'], r['unique_aa'], r['entropy_mean']) for r in res_list]
        ag = np.array([v[0] for v in vals])
        ua = np.array([v[1] for v in vals])
        ent = np.array([v[2] for v in vals])
        r_ua, p_ua = spearmanr(ag, ua)
        r_ent, p_ent = spearmanr(ag, ent)
        print(f"\n  {args.label}_{dm} (n={len(vals)}):")
        print(f"    unique_aa vs ag_disorder_max: r={r_ua:+.4f}  p={p_ua:.4f}")
        print(f"    entropy   vs ag_disorder_max: r={r_ent:+.4f}  p={p_ent:.4f}")

        e = np.mean([r['entropy_mean'] for r in res_list])
        u = np.mean([r['unique_aa'] for r in res_list])
        t = np.mean([r['top_aa_pct'] for r in res_list])
        n6 = sum(1 for r in res_list if r.get('has_6mer'))
        print(f"    Summary: ent={e:.4f}  uniq={u:.1f}  top_aa={t:.3f}  6mer={n6}")

    # Save
    os.makedirs('idp_design_results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = f'idp_design_results/way4_ablate_{args.label}_p2_{ts}.json'
    json.dump({
        'probe': f'V13.1 Ablation {args.label}',
        'checkpoint': args.checkpoint,
        'n_samples': args.n_samples,
        'n_seeds': args.n_seeds,
        'n_total': len(results),
        'results': results,
    }, open(out, 'w'), indent=2)
    print(f"\nSaved: {out}")
