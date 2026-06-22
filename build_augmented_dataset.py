#!/usr/bin/env python
"""Plan B: Build augmented training dataset with BFN-generated CDR variants.

For each entry with a CDR region, uses BFN to generate N variants with
different PPL values. This gives the variance/ranking loss real contrast.

Output: data/confidence_augmented_v1/train.lmdb

Usage:
  python build_augmented_dataset.py --n_backbones 200 --n_variants 3 --device cuda
"""
import sys, os
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.'); sys.path.insert(0, 'modules')

import argparse, pickle, random, time
import lmdb, torch, yaml, numpy as np
from disorderflow.utils.misc import seed_all
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to

SRC_LMDB = 'data/confidence_merged_v11/confidence_train.lmdb'
DST_DIR = 'data/confidence_augmented_v1'
V12_CKPT = 'logs/bfn_v12_seqconf_xpu_2026_06_20__02_54_40/checkpoints/best.pt'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def generate_variants(model, batch, n_variants, device):
    """Generate n_variants CDR designs using BFN. Returns list of (seq, ppl)."""
    gen_mask = batch['generate_flag']
    n_cdr = gen_mask.sum().item()
    if n_cdr < 20 or n_cdr > 200:
        return []

    batch_dev = recursive_to(PaddingCollate()([batch]), device)
    gen_mask_dev = batch_dev['generate_flag'][0].bool()

    variants = []
    for _ in range(n_variants):
        with torch.no_grad():
            traj = model.sample(batch_dev, sample_opt={
                'deterministic': False, 'num_recycles': 2})

        pred_aa = traj[0][2][0][gen_mask_dev].cpu()
        seq = ''.join(AA[a] if a < 20 else 'X' for a in pred_aa)

        logits = traj['pred_logits'][0][gen_mask_dev]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        idx = torch.arange(len(pred_aa))
        nll = -lp[idx, pred_aa.long()].mean()
        ppl = torch.exp(nll).item()

        variants.append((seq, ppl))

    return variants


def build_augmented(n_backbones=200, n_variants=3, device='cuda'):
    cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    with open(cfg_path) as f:
        app_cfg = yaml.safe_load(f)
    orig_ckpt = app_cfg['models']['bfn']['checkpoint']
    app_cfg['models']['bfn']['checkpoint'] = V12_CKPT
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)

    try:
        from bfn_loader import load_bfn
        import bfn_loader
        bfn_loader._bfn_model = None
        bfn_loader._bfn_config = None
        seed_all(42)
        model, _ = load_bfn(device)
        model.eval()
        print(f'Model loaded')

        src_env = lmdb.open(SRC_LMDB, readonly=True)
        n_entries = src_env.stat()['entries'] - 1  # exclude __len__
        print(f'Source: {n_entries} entries')

        # Find entries with CDR regions
        cdr_entries = []
        with src_env.begin() as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                try:
                    idx = int(key.decode())
                except (ValueError, UnicodeDecodeError):
                    continue  # skip non-numeric keys ('__len__', etc.)
                e = pickle.loads(val)
                n_cdr = e['batch']['generate_flag'].sum().item()
                if 20 <= n_cdr <= 200:
                    cdr_entries.append(idx)
            cursor.close()

        random.seed(42)
        random.shuffle(cdr_entries)
        cdr_entries = cdr_entries[:n_backbones]
        print(f'Sampled {len(cdr_entries)} entries with CDR 20-200')

        # Create destination LMDB
        dst_path = os.path.join(DST_DIR, 'train.lmdb')
        os.makedirs(DST_DIR, exist_ok=True)
        dst_env = lmdb.open(dst_path, map_size=int(5e10))

        n_augmented = 0
        n_skipped = 0

        with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
            # Copy all originals
            for i in range(n_entries):
                dst_txn.put(f'{i:08d}'.encode(), src_txn.get(f'{i:08d}'.encode()))
            print(f'Copied {n_entries} originals')

            t0 = time.time()
            for i, idx in enumerate(cdr_entries):
                entry = pickle.loads(src_txn.get(f'{idx:08d}'.encode()))
                batch = entry['batch']
                aa_orig = batch['aa'].clone()

                variants = generate_variants(model, batch, n_variants, device)
                if not variants:
                    n_skipped += 1
                    continue

                for v_idx, (seq, ppl) in enumerate(variants):
                    aug_batch = dict(batch)
                    gen_idx = batch['generate_flag'].nonzero(as_tuple=True)[0]
                    aug_aa = aa_orig.clone()
                    for j, pos in enumerate(gen_idx):
                        if j < len(seq):
                            aug_aa[pos] = AA.index(seq[j])

                    aug_batch['aa'] = aug_aa
                    aug_entry = dict(entry)
                    aug_entry['batch'] = aug_batch
                    aug_entry['ppl'] = ppl
                    aug_entry['augmented'] = True

                    # ── Key fix: lower confidence labels for variants ──
                    # BFN-designed variants are less reliable than the original.
                    # Set low confidence so model learns: variant = low score.
                    if 'af2_plddt' in aug_entry and aug_entry['af2_plddt'] is not None:
                        aug_entry['af2_plddt'] = aug_entry['af2_plddt'] * 0.3
                    if 'af2_iptm' in aug_entry and aug_entry['af2_iptm'] is not None:
                        aug_entry['af2_iptm'] = torch.tensor(0.05)
                    if 'af2_pae_matrix' in aug_entry and aug_entry['af2_pae_matrix'] is not None:
                        aug_entry['af2_pae_matrix'] = torch.clamp(
                            aug_entry['af2_pae_matrix'] * 2.0, max=0.95)

                    aug_key = f'{n_entries + n_augmented:08d}'.encode()
                    dst_txn.put(aug_key, pickle.dumps(aug_entry))
                    n_augmented += 1

                if (i + 1) % 20 == 0:
                    dt = time.time() - t0
                    eta = dt / max(1, n_augmented) * (len(cdr_entries) * n_variants - n_augmented)
                    print(f'  [{i+1}/{len(cdr_entries)}] '
                          f'{n_augmented} variants ({dt:.0f}s, ETA {eta:.0f}s)')

        # Write metadata
        total_entries = n_entries + n_augmented
        with dst_env.begin(write=True) as meta_txn:
            meta_txn.put(b'__len__', pickle.dumps(total_entries))
        dst_env.close()
        src_env.close()

        print(f'\nDone! {n_augmented} augmented + {n_entries} original = {total_entries} total')
        print(f'  Skipped: {n_skipped}')
        print(f'  Output: {dst_path}')
        print(f'\nNext: Set db_path to {dst_path} and train V12 Phase B')

    finally:
        app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(app_cfg, f, default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_backbones', type=int, default=200)
    parser.add_argument('--n_variants', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    build_augmented(n_backbones=args.n_backbones, n_variants=args.n_variants, device=args.device)
