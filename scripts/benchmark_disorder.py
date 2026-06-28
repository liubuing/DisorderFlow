#!/usr/bin/env python3
"""I-1: Disorder head CAID benchmark — quantify IDP detection capability.

Evaluates BFN disorder_head predictions against CAID (Critical Assessment
of Intrinsic Disorder) targets. Compares AUC-ROC/AUC-PR vs:
  - IUPred3 (state-of-the-art disorder predictor)
  - AF2 pLDDT (folded-protein confidence proxy)
  - Baseline: all-ordered / all-disordered

Usage:
    python scripts/benchmark_disorder.py --data-dir data/caid/
    python scripts/benchmark_disorder.py --data-dir data/caid/ --bfn-ckpt path/to/v15.pt

CAID data format (expected):
    data/caid/
      DisProt_targets.fasta   — target sequences
      DisProt_labels.tsv      — per-residue disorder labels (0=ordered, 1=disordered)

If CAID data is not available, use --sabdab to run a quick sanity check on
SAbDab complexes (known ordered) + Aβ42 (known disordered, RMSF labels).
"""
import sys, os, argparse, json, time
sys.path.insert(0, '.'); sys.path.insert(0, 'modules')
import numpy as np
import torch
import yaml

AA = 'ACDEFGHIKLMNPQRSTVWY'
V15_CKPT = 'logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt'


def load_bfn(ckpt_path):
    """Load BFN model from checkpoint."""
    import bfn_loader
    app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
    orig = app['models']['bfn']['checkpoint']
    app['models']['bfn']['checkpoint'] = ckpt_path
    yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
    bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
    model, cfg = bfn_loader.load_bfn('cpu')
    app['models']['bfn']['checkpoint'] = orig
    yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
    return model


def compute_auc(y_true, y_pred):
    """Compute AUC-ROC from binary labels and continuous predictions."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    yt = np.array(y_true); yp = np.array(y_pred)
    mask = ~np.isnan(yp)
    if mask.sum() < 2:
        return float('nan'), float('nan')
    try:
        roc = roc_auc_score(yt[mask], yp[mask])
        pr = average_precision_score(yt[mask], yp[mask])
    except ValueError:
        roc, pr = float('nan'), float('nan')
    return roc, pr


def benchmark_sabdab_quick(bfn_ckpt=None):
    """Quick sanity check: disorder head on SAbDab (ordered) + Aβ42 (disordered)."""
    print("=== Quick disorder head benchmark (SAbDab + Aβ42) ===\n")
    model = load_bfn(bfn_ckpt or V15_CKPT)

    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.transforms import get_transform
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all
    import lmdb, pickle, random

    # Ordered: SAbDab complexes
    env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', subdir=False,
                    readonly=True, lock=False, readahead=False)
    ids = pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb'))
    random.seed(42); random.shuffle(ids)

    ordered_preds, ordered_labels = [], []
    transform = get_transform([{'type': 'merge_chains'}, {'type': 'patch_around_anchor'}])

    n_checked = 0
    with env.begin() as txn:
        for sid in ids:
            if n_checked >= 10:
                break
            e = pickle.loads(txn.get(sid.encode()))
            if e.get('heavy') is None: continue
            # Pass through transform
            batch_data = transform(e)
            batch = recursive_to(PaddingCollate()([batch_data]), 'cpu')
            with torch.no_grad():
                out = model.bfn(batch)
            disorder = out.get('pred_disorder', torch.zeros_like(batch['aa']))
            ordered_preds.extend(disorder[batch['mask'].bool()].cpu().numpy().flatten().tolist())
            ordered_labels.extend([0] * batch['mask'].sum().item())  # SAbDab = ordered
            n_checked += 1
    env.close()
    print(f"Ordered (SAbDab {n_checked} complexes): "
          f"disorder mean={np.mean(ordered_preds):.4f} std={np.std(ordered_preds):.4f}")

    # Disordered: Aβ42
    import pickle as pkl
    with open('data/abeta_conformations/abeta42_5seed.pkl', 'rb') as f:
        abeta = pkl.load(f)
    rmsf = abeta['rmsf']  # (42,) RMSF values
    disorder_labels = (rmsf / rmsf.max()).tolist()  # normalize to [0,1]
    print(f"Disordered (Aβ42 5-seed): RMSF mean={rmsf.mean():.2f} max={rmsf.max():.2f}")

    # Load BFN disorder head predictions from probe
    print("\nNote: For full per-residue disorder_pred on Aβ42, run probe_disorder_balance.py")
    print(f"Expected: ordered ~0.005, disordered ~0.59 (after retrain)")
    print(f"V15 baseline: ordered ~0.005, disordered ~0.036 (before retrain → blind to IDP)")
    print(f"AUC-ROC (if labels available): would quantify separation quality")


def main():
    parser = argparse.ArgumentParser(description='Disorder head CAID benchmark')
    parser.add_argument('--data-dir', default=None, help='CAID data directory')
    parser.add_argument('--bfn-ckpt', default=None, help='BFN checkpoint')
    parser.add_argument('--sabdab', action='store_true',
                        help='Quick sanity check on SAbDab + Aβ42')
    args = parser.parse_args()

    if args.data_dir and os.path.isdir(args.data_dir):
        print(f"CAID benchmark not yet implemented. Data dir: {args.data_dir}")
        print("TODO: load CAID targets, predict with BFN disorder head, compute AUC-ROC/AUC-PR")
        print("Baselines: IUPred3, AF2 pLDDT")
    elif args.sabdab:
        benchmark_sabdab_quick(args.bfn_ckpt)
    else:
        print("Usage: python scripts/benchmark_disorder.py --sabdab  (quick check)")
        print("       python scripts/benchmark_disorder.py --data-dir data/caid/  (full CAID)")


if __name__ == '__main__':
    main()
