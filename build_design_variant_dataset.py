#!/usr/bin/env python
"""V14 design-variant dataset builder — REAL AF2 labels per BFN design.

This is the fix for the stuck 9.47x overconfidence. The old builder
(build_foundation_dataset.py) generated BFN design variants per scaffold but
assigned FAKE AF2 labels (native_iptm × quality_mult), which trained the
confidence head to be sequence-invariant — the root cause of OC.

This builder instead:
  1. For each source scaffold (entry in the V11 confidence LMDB), generate
     K BFN-designed CDR variants.
  2. For each variant, run REAL AF2 (JAX-multimer or colabfold CLI) on the
     designed full sequence → record the TRUE af2_iptm / af2_plddt / af2_pae.
  3. Store each variant as an entry whose batch['aa'] = the DESIGNED sequence
     (native framework + designed CDRs), with scaffold_id grouping, and the
     TRUE AF2 labels.

The grouped margin/variance losses in disorderflow/modules/bfn/core.py then
supervise within-scaffold design-specificity against real labels.

Output: data/confidence_design_variants_v14/{train,val}.lmdb

Usage:
  # Smoke test: 10 scaffolds × 4 variants = 40 AF2 runs (~30-60 min on GPU)
  python build_design_variant_dataset.py --smoke-test --device cuda

  # Full build: ~200 scaffolds × 8 variants = 1600 AF2 runs (1-2 days GPU)
  python build_design_variant_dataset.py --n-scaffolds 200 --k-variants 8 --device cuda
"""
import sys
import os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import argparse
import pickle
import time
import lmdb
import torch
import yaml

from disorderflow.utils.misc import seed_all
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to

AA = 'ACDEFGHIKLMNPQRSTVWY'
SRC_LMDB = 'data/confidence_merged_v11/confidence_train.lmdb'
DST_DIR = 'data/confidence_design_variants_v14'
V12_CKPT = 'logs/bfn_v12_seqconf_xpu_2026_06_20__02_54_40/checkpoints/best.pt'


def _load_src_entries(src_lmdb, limit=None):
    """Load source scaffolds (native backbone + native seq + native AF2 labels)."""
    env = lmdb.open(src_lmdb, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        n = pickle.loads(txn.get(b'__len__'))
    entries = []
    with env.begin() as txn:
        for i in range(n):
            raw = txn.get(f'{i:08d}'.encode())
            if raw is None:
                continue
            e = pickle.loads(raw)
            if e.get('batch') is None:
                continue
            entries.append(e)
            if limit and len(entries) >= limit:
                break
    env.close()
    return entries


def bfn_generate_variants(model, batch, k, device):
    """Generate k CDR design variants. Returns list of full designed aa tensors."""
    gen_mask = batch['generate_flag']
    n_cdr = int(gen_mask.sum())
    if n_cdr < 4 or n_cdr > 200:
        return []
    batch_dev = recursive_to(PaddingCollate()([batch]), device)
    variants = []
    for _ in range(k):
        with torch.no_grad():
            traj = model.sample(batch_dev, sample_opt={'deterministic': False, 'num_recycles': 2})
        pred_aa = traj[0][2][0]  # (L,)
        designed = pred_aa.cpu().clone()
        designed[designed >= 20] = 20  # keep non-standard as 'X' idx
        variants.append(designed)
    return variants


def _full_designed_sequence(native_aa, gen_mask, designed_aa):
    """Build the full-length sequence tensor: native framework + designed CDRs."""
    full = native_aa.clone()
    full[gen_mask] = designed_aa[gen_mask]
    return full


def real_af2_labels(full_aa_tensor, model_config=None, num_recycle=2):
    """Run REAL AF2 on a designed full sequence → (iptm, plddt, pae_matrix).

    Uses modules/af2_jax_runner.run_multimer_prediction. single-chain if no
    antigen context. Returns None on failure.
    """
    from af2_jax_runner import run_multimer_prediction
    seq = ''.join(AA[a] if a < 20 else 'X' for a in full_aa_tensor.tolist())
    try:
        res = run_multimer_prediction(seq, "", data_dir=None, num_recycle=num_recycle)
    except Exception as e:
        print(f"      [AF2] failed: {e}")
        return None
    if not res.get('success'):
        return None
    import numpy as np
    plddt = torch.tensor(res.get('plddt_seq', res.get('plddt', [0.5])),
                         dtype=torch.float32)
    if plddt.dim() == 0:
        plddt = plddt.unsqueeze(0).expand(len(seq))
    iptm = torch.tensor(float(res.get('iptm', 0.0)), dtype=torch.float32)
    pae = torch.tensor(np.array(res.get('pae', [[0.0]])), dtype=torch.float32) / 31.0
    return {'af2_plddt': plddt, 'af2_iptm': iptm, 'af2_pae_matrix': pae}


def _write_entry(env, key_idx, scaffold_id, design_idx, batch, af2_labels):
    """Write one design-variant entry to the LMDB."""
    entry = {
        'pdb_id': batch.get('pdb_id', f'scaffold_{scaffold_id}'),
        'sequence': ''.join(AA[a] if a < 20 else 'X' for a in batch['aa'].tolist()),
        'scaffold_id': scaffold_id,
        'design_idx': design_idx,
        'batch': batch,
        'af2_plddt': af2_labels['af2_plddt'],
        'af2_iptm': af2_labels['af2_iptm'],
        'af2_pae_matrix': af2_labels['af2_pae_matrix'],
        'is_idp': batch.get('is_idp', False),
        'source': 'design_variant_v14',
    }
    with env.begin(write=True) as txn:
        txn.put(f'{key_idx:08d}'.encode(), pickle.dumps(entry))


def build(src_lmdb, dst_dir, n_scaffolds, k_variants, device, num_recycle, smoke):
    from bfn_loader import load_bfn

    seed_all(42)
    os.makedirs(dst_dir, exist_ok=True)
    src = _load_src_entries(src_lmdb, limit=(10 if smoke else n_scaffolds))
    print(f"Loaded {len(src)} source scaffolds from {src_lmdb}")

    # Point the loader at the V12 checkpoint.
    cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    with open(cfg_path, encoding='utf-8') as f:
        app_cfg = yaml.safe_load(f)
    app_cfg['models']['bfn']['checkpoint'] = V12_CKPT
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)

    import bfn_loader
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, config = load_bfn(device)
    print(f"BFN loaded (ckpt={V12_CKPT})")

    train_env = lmdb.open(os.path.join(dst_dir, 'train.lmdb'), map_size=1 << 35)
    key_idx = 0
    n_written, n_af2_fail = 0, 0
    t0 = time.time()

    for s_idx, scaffold in enumerate(src):
        batch = scaffold['batch']
        gen_mask = batch.get('generate_flag')
        if gen_mask is None or int(gen_mask.sum()) == 0:
            continue
        variants = bfn_generate_variants(model, batch, k_variants, device)
        if not variants:
            continue
        print(f"[{s_idx+1}/{len(src)}] scaffold {s_idx}: {len(variants)} variants, "
              f"AF2... ({time.time()-t0:.0f}s)", flush=True)

        for d_idx, designed_aa in enumerate(variants):
            full_aa = _full_designed_sequence(batch['aa'], gen_mask, designed_aa)
            var_batch = dict(batch)
            var_batch['aa'] = full_aa
            # The designed region is now "context" (the head reads it directly);
            # generate_flag stays all-zero (dataset overrides it on load).
            labels = real_af2_labels(full_aa, num_recycle=num_recycle)
            if labels is None:
                n_af2_fail += 1
                continue
            _write_entry(train_env, key_idx, s_idx, d_idx, var_batch, labels)
            key_idx += 1
            n_written += 1

    with train_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(key_idx))
    train_env.close()

    print(f"\nDone. Wrote {n_written} design-variant entries "
          f"({n_af2_fail} AF2 failures skipped) in {time.time()-t0:.0f}s")
    print(f"Output: {os.path.join(dst_dir, 'train.lmdb')}")
    if n_written == 0:
        print("WARNING: no entries written — check BFN/AF2 availability and gen_mask.")


def main():
    p = argparse.ArgumentParser(description="Build V14 real-AF2 design-variant dataset")
    p.add_argument('--src', default=SRC_LMDB)
    p.add_argument('--dst', default=DST_DIR)
    p.add_argument('--n-scaffolds', type=int, default=200)
    p.add_argument('--k-variants', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-recycle', type=int, default=2)
    p.add_argument('--smoke-test', action='store_true',
                   help="Quick run: 10 scaffolds × 4 variants (validated by the runner)")
    args = p.parse_args()
    if args.smoke_test:
        args.k_variants = 4
    build(args.src, args.dst, args.n_scaffolds, args.k_variants,
          args.device, args.num_recycle, args.smoke_test)


if __name__ == '__main__':
    main()
