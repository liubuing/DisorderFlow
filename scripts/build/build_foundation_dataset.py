import os
#!/usr/bin/env python
"""Foundation: build training data with per-backbone quality labels.

For each of 500 antibody backbones:
  1 × native (good)    → keep original AF2 labels
  2 × BFN-generated    → label = original × 0.5 (medium)
  1 × scrambled        → label = original × 0.1 (bad)

This gives the confidence heads REAL contrast signal:
  same backbone, different CDRs, different quality labels.

Output: data/confidence_foundation_v1/train.lmdb

Usage:
  python build_foundation_dataset.py --n_backbones 500 --device cuda

################################################################################
DEPRECATED (2026-06-23): The "label = original × quality_mult" heuristic below
is SYNTHETIC — it does NOT run real AF2 on the designed sequences. Two different
designs on the same scaffold therefore receive the SAME fake label
(~0.5 × native), which actively trains the confidence head to be
sequence-invariant. This is a root cause of the stuck 9.47x overconfidence
(see DIAGNOSIS.md and modules/bfn/core.py V14 loss notes).

For V14 training, use build_design_variant_dataset.py instead — it runs REAL
AF2 per design variant and writes scaffold_grouped entries with true labels.
This script is retained for reproducibility/ablation only.
################################################################################
"""
import sys, os
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import argparse, pickle, random, time, lmdb, torch, yaml, numpy as np
from disorderflow.utils.misc import seed_all
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to

SRC_LMDB = 'data/confidence_merged_v11/confidence_train.lmdb'
DST_DIR = 'data/confidence_foundation_v2'
V12_CKPT = 'logs/bfn_v12_seqconf_xpu_2026_06_20__02_54_40/checkpoints/best.pt'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def bfn_generate(model, batch, n_variants, device):
    """Generate CDR variants using BFN. Returns [(seq, ppl), ...]."""
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


def scramble_cdr(aa_tensor, gen_mask):
    """Shuffle CDR residues."""
    result = aa_tensor.clone()
    cdr_idx = gen_mask.nonzero(as_tuple=True)[0]
    vals = result[cdr_idx].clone()
    result[cdr_idx] = vals[torch.randperm(len(cdr_idx))]
    return result


def build(n_backbones=500, device='cuda'):
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

        src_env = lmdb.open(SRC_LMDB, readonly=True)
        n_data = src_env.stat()['entries'] - 1
        print(f'Source: {n_data} data entries')

        # Find CDR entries
        cdr_indices = []
        with src_env.begin() as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                try:
                    idx = int(key.decode())
                except (ValueError, UnicodeDecodeError):
                    continue
                e = pickle.loads(val)
                n_cdr = e['batch']['generate_flag'].sum().item()
                if 20 <= n_cdr <= 200:
                    cdr_indices.append(idx)
            cursor.close()

        random.seed(42)
        random.shuffle(cdr_indices)
        chosen = cdr_indices[:n_backbones]
        print(f'CDR entries: {len(cdr_indices)}, using {len(chosen)}')

        # Build destination
        dst_path = os.path.join(DST_DIR, 'train.lmdb')
        os.makedirs(DST_DIR, exist_ok=True)
        dst_env = lmdb.open(dst_path, map_size=int(5e10))

        n_aug = 0
        n_skip = 0
        next_key = n_data  # sequential numeric keys

        with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
            # Copy all originals
            for idx in range(n_data):
                key = f'{idx:08d}'
                val = src_txn.get(key.encode())
                if val is not None:
                    dst_txn.put(key.encode(), val)
            print(f'Copied originals')

            t0 = time.time()
            for i, idx in enumerate(chosen):
                entry = pickle.loads(src_txn.get(f'{idx:08d}'.encode()))
                batch = entry['batch']
                aa_orig = batch['aa'].clone()
                gen_mask = batch['generate_flag']

                # Generate 2 BFN variants (medium quality)
                variants = bfn_generate(model, batch, 2, device)
                if len(variants) < 2:
                    n_skip += 1
                    continue

                # Create augmented entries
                gen_idx = gen_mask.nonzero(as_tuple=True)[0]

                def update_entry(seq, quality_mult, tag):
                    aug_batch = dict(batch)
                    aug_aa = aa_orig.clone()
                    for j, pos in enumerate(gen_idx):
                        if j < len(seq):
                            aug_aa[pos] = AA.index(seq[j])
                    aug_batch['aa'] = aug_aa
                    aug_entry = dict(entry)
                    aug_entry['batch'] = aug_batch
                    if 'af2_plddt' in aug_entry and aug_entry['af2_plddt'] is not None:
                        aug_entry['af2_plddt'] = aug_entry['af2_plddt'] * quality_mult
                    if 'af2_iptm' in aug_entry and aug_entry['af2_iptm'] is not None:
                        aug_entry['af2_iptm'] = torch.tensor(
                            float(aug_entry['af2_iptm']) * quality_mult)
                    if 'af2_pae_matrix' in aug_entry and aug_entry['af2_pae_matrix'] is not None:
                        aug_entry['af2_pae_matrix'] = torch.clamp(
                            aug_entry['af2_pae_matrix'] * (2.0 - quality_mult), max=0.95)
                    aug_entry['foundation_tag'] = tag
                    return aug_entry

                e1 = update_entry(variants[0][0], 0.5, 'bfn_med_1')
                dst_txn.put(f'{next_key:08d}'.encode(), pickle.dumps(e1))
                n_aug += 1; next_key += 1

                e2 = update_entry(variants[1][0], 0.5, 'bfn_med_2')
                dst_txn.put(f'{next_key:08d}'.encode(), pickle.dumps(e2))
                n_aug += 1; next_key += 1

                scrambled_aa = scramble_cdr(aa_orig, gen_mask)
                scrambled_seq = ''.join(AA[a] if a < 20 else 'X'
                                        for a in scrambled_aa[gen_idx])
                e3 = update_entry(scrambled_seq, 0.1, 'scrambled_bad')
                dst_txn.put(f'{next_key:08d}'.encode(), pickle.dumps(e3))
                n_aug += 1; next_key += 1

                if (i + 1) % 50 == 0:
                    dt = time.time() - t0
                    eta = dt / (i + 1) * (len(chosen) - i - 1)
                    print(f'  [{i+1}/{len(chosen)}] {n_aug} variants ({dt:.0f}s, ETA {eta:.0f}s)')

        # Write metadata
        total = n_data + n_aug
        with dst_env.begin(write=True) as mt:
            mt.put(b'__len__', pickle.dumps(total))
        dst_env.close()
        src_env.close()

        print(f'\nDone!')
        print(f'  Native (good):  {n_data}')
        print(f'  BFN medium:     {n_backbones - n_skip}×2 ~{2*(n_backbones-n_skip)}')
        print(f'  Scrambled bad:  {n_backbones - n_skip}')
        print(f'  Total:          {total}')
        print(f'  Skipped:        {n_skip}')
        print(f'  Output: {dst_path}')

    finally:
        app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(app_cfg, f, default_flow_style=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n_backbones', type=int, default=500)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()
    build(n_backbones=args.n_backbones, device=args.device)
