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
# Default epitope: Abeta42 peptide (2NAO). Used as the AF2 multimer partner
# for all designs so that ipTM/lDDT-cdr reflect real antibody-antigen fit.
DEFAULT_EPITOPE = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'


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
    """Build the full-length sequence tensor: native framework + designed CDRs.

    PaddingCollate rounds lengths up to a multiple of 8, so gen_mask (the
    unpadded generate_flag) may be shorter than native_aa / designed_aa.
    Truncate to the shortest common length and use index-based assignment.
    """
    full = native_aa.clone()
    L = min(full.shape[0], gen_mask.shape[0], designed_aa.shape[0])
    gen = gen_mask[:L]
    cdr_idx = gen.nonzero(as_tuple=True)[0]
    full[cdr_idx] = designed_aa[cdr_idx]
    return full


def real_af2_labels(full_aa_tensor, epitope_seq=DEFAULT_EPITOPE, num_recycle=2, use_wsl=False):
    """Run REAL AF2 on a designed full sequence → (iptm, plddt, pae_matrix).

    Runs AF2 multimer (antibody + epitope), so ipTM reflects antibody-epitope
    interface quality — not just monomer self-consistency.

    When use_wsl=True, calls the WSL2 GPU JAX batch processor via subprocess.
    Falls back to local JAX on failure. Returns None on failure.
    """
    import numpy as np
    import subprocess
    import json as _json
    seq = ''.join(AA[a] if a < 20 else 'X' for a in full_aa_tensor.tolist())

    if use_wsl:
        # One-by-one WSL worker for backward compat (batch mode preferred)
        cmd = [
            'wsl', '-d', 'Debian', '--', 'bash', '-c',
            f'source venv_wsl/bin/activate && python af2_wsl_worker.py "$1" "$2" "$3"',
            '--', seq, epitope_seq, str(num_recycle),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                  cwd=PROJECT_ROOT)
            out_text = proc.stdout.strip()
            res = _json.loads(out_text.split('\n')[-1])
        except Exception as e:
            print(f"      [AF2-WSL] failed: {e}")
            return None
        if not res.get('success'):
            print(f"      [AF2-WSL] error: {res.get('error', 'unknown')[:120]}")
            return None
        plddt = torch.tensor(np.array(res.get('plddt_seq', res.get('plddt', [0.5]))),
                             dtype=torch.float32)
        if plddt.dim() == 0:
            plddt = plddt.unsqueeze(0).expand(len(seq))
        iptm = torch.tensor(float(res.get('iptm', 0.0)), dtype=torch.float32)
        pae = torch.tensor(np.array(res.get('pae', [[0.0]])), dtype=torch.float32) / 31.0
        return {'af2_plddt': plddt, 'af2_iptm': iptm, 'af2_pae_matrix': pae}

    # Legacy: local JAX (CPU on Windows)
    from af2_jax_runner import run_multimer_prediction
    try:
        res = run_multimer_prediction(seq, epitope_seq, data_dir=None, num_recycle=num_recycle)
    except Exception as e:
        print(f"      [AF2] failed: {e}")
        return None
    if not res.get('success'):
        return None
    plddt = torch.tensor(res.get('plddt_seq', res.get('plddt', [0.5])),
                         dtype=torch.float32)
    if plddt.dim() == 0:
        plddt = plddt.unsqueeze(0).expand(len(seq))
    iptm = torch.tensor(float(res.get('iptm', 0.0)), dtype=torch.float32)
    pae = torch.tensor(np.array(res.get('pae', [[0.0]])), dtype=torch.float32) / 31.0
    return {'af2_plddt': plddt, 'af2_iptm': iptm, 'af2_pae_matrix': pae}


def _write_entry(env, key_idx, scaffold_id, design_idx, batch, af2_labels):
    """Write one design-variant entry to the LMDB."""
    aa_len = batch['aa'].shape[0]
    plddt = af2_labels['af2_plddt']
    pae = af2_labels['af2_pae_matrix']
    # AF2 multimer runs on antibody + epitope complex. Truncate to ab only.
    if plddt.shape[0] > aa_len:
        plddt = plddt[:aa_len]
    if pae.dim() >= 2 and pae.shape[0] > aa_len:
        pae = pae[:aa_len, :aa_len]
    entry = {
        'pdb_id': batch.get('pdb_id', f'scaffold_{scaffold_id}'),
        'sequence': ''.join(AA[a] if a < 20 else 'X' for a in batch['aa'].tolist()),
        'scaffold_id': scaffold_id,
        'design_idx': design_idx,
        'batch': batch,
        'af2_plddt': plddt,
        'af2_iptm': af2_labels['af2_iptm'],
        'af2_pae_matrix': pae,
        'is_idp': batch.get('is_idp', False),
        'source': 'design_variant_v14',
    }
    with env.begin(write=True) as txn:
        txn.put(f'{key_idx:08d}'.encode(), pickle.dumps(entry))


def _batch_af2_wsl(seqs, epitope_seq, num_recycle, warmup_seq=None, output_dir=None):
    """Run AF2 on sequences via WSL2 GPU in chunks to avoid JAX memory exhaustion.

    JAX compiles a new XLA kernel for each unique sequence length. With 100+
    different lengths, GPU memory fills after ~38 unique shapes. Chunking
    restarts the Python process periodically, clearing the JIT cache.
    """
    import subprocess
    import json as _json

    CHUNK_SIZE = 30
    jobs_path = os.path.join(PROJECT_ROOT, '.af2_wsl_jobs.jsonl')
    results_path = os.path.join(PROJECT_ROOT, '.af2_wsl_results.jsonl')
    wsl_jobs = '/mnt/c/biological/DisorderFlow/.af2_wsl_jobs.jsonl'
    wsl_results = '/mnt/c/biological/DisorderFlow/.af2_wsl_results.jsonl'
    epitope_arg = f'--epitope {epitope_seq}'
    wsl_distribution = os.environ.get('DISORDERFLOW_AF2_WSL_DISTRO', 'Ubuntu-24.04')

    all_results = [None] * len(seqs)
    output_dir_wsl = None
    if output_dir:
        output_dir_abs = os.path.abspath(output_dir).replace('\\', '/')
        if len(output_dir_abs) >= 3 and output_dir_abs[1:3] == ':/':
            output_dir_wsl = f'/mnt/{output_dir_abs[0].lower()}{output_dir_abs[2:]}'
    n_chunks = (len(seqs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    t0 = time.time()
    total_processed = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(seqs))
        chunk_seqs = seqs[start:end]
        chunk_len = len(chunk_seqs)

        # Write chunk jobs with original global IDs
        with open(jobs_path, 'w', encoding='utf-8') as f:
            for local_i, global_i in enumerate(range(start, end)):
                job = {
                    'seq': seqs[global_i], 'epi_seq': epitope_seq,
                    'id': global_i, 'recycle': num_recycle
                }
                if output_dir_wsl:
                    job['output_pdb'] = f'{output_dir_wsl}/{global_i:03d}.pdb'
                f.write(_json.dumps(job) + '\n')

        # Only use warmup on first chunk (compiles JAX kernels for subsequent)
        warmup_arg = f'--warmup-seq {chunk_seqs[0]}' if chunk_idx == 0 and chunk_seqs else ''

        cmd = (
            f'wsl -d {wsl_distribution} -- bash -c '
            f'"source venv_wsl/bin/activate && python scripts/utils/af2_wsl_batch.py '
            f'--recycle {num_recycle} {epitope_arg} {warmup_arg} '
            f'< {wsl_jobs} > {wsl_results} 2>/dev/null"'
        )

        print(f"    Chunk {chunk_idx+1}/{n_chunks}: {chunk_len} seqs (global {start}-{end-1})...",
              end=' ', flush=True)
        t_chunk = time.time()
        try:
            subprocess.run(cmd, shell=True, timeout=chunk_len * 600 + 600,
                           cwd=PROJECT_ROOT)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT")
            continue
        dt_chunk = time.time() - t_chunk
        print(f"{dt_chunk:.0f}s ({dt_chunk/chunk_len:.1f}s/seq)", flush=True)

        # Parse chunk results
        if os.path.exists(results_path):
            with open(results_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    rid = r.get('id', -1)
                    if 0 <= rid < len(all_results):
                        all_results[rid] = r
            os.unlink(results_path)
        os.unlink(jobs_path)

        n_done = sum(1 for r in all_results if r is not None)
        print(f"    Progress: {n_done}/{len(seqs)} results collected", flush=True)

    dt = time.time() - t0
    n_done = sum(1 for r in all_results if r is not None)
    print(f"    WSL batch done: {n_done}/{len(seqs)} results in {dt:.0f}s "
          f"({dt/max(n_done,1):.1f}s/seq)", flush=True)

    return all_results


def build(src_lmdb, dst_dir, n_scaffolds, k_variants, device, num_recycle, smoke, use_wsl=True, epitope_seq=DEFAULT_EPITOPE):
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

    t0 = time.time()

    # Phase 1: Generate all BFN variants (GPU), collect sequences
    jobs = []  # list of (scaffold_idx, design_idx, var_batch, seq_string)
    for s_idx, scaffold in enumerate(src):
        batch = scaffold['batch']
        gen_mask = batch.get('generate_flag')
        if gen_mask is None or int(gen_mask.sum()) == 0:
            continue
        variants = bfn_generate_variants(model, batch, k_variants, device)
        if not variants:
            continue
        print(f"[{s_idx+1}/{len(src)}] scaffold {s_idx}: {len(variants)} variants "
              f"({time.time()-t0:.0f}s)", flush=True)
        for d_idx, designed_aa in enumerate(variants):
            full_aa = _full_designed_sequence(batch['aa'], gen_mask, designed_aa)
            var_batch = dict(batch)
            var_batch['aa'] = full_aa
            seq_str = ''.join(AA[a] if a < 20 else 'X' for a in full_aa.tolist())
            jobs.append((s_idx, d_idx, var_batch, seq_str))

    n_variants = len(jobs)
    print(f"\nPhase 1 done: {n_variants} variants from {len(src)} scaffolds "
          f"({time.time()-t0:.0f}s)")

    if n_variants == 0:
        print("WARNING: no design variants generated — check gen_mask and BFN model.")
        return

    # Phase 2: AF2 inference — WSL GPU batch (fast) or local CPU (fallback)
    af2_results = None
    if use_wsl and not smoke:
        seqs = [j[3] for j in jobs]
        # Use first sequence as warmup (compiled kernels persist for subsequent)
        af2_results = _batch_af2_wsl(seqs, epitope_seq, num_recycle,
                                     warmup_seq=seqs[0] if seqs else None)
        if af2_results is None:
            print("WSL batch AF2 failed. Falling back to local CPU AF2.")

    if af2_results is None:
        # Legacy: one-by-one local AF2 (slow CPU)
        af2_results = []
        for i, (s_idx, d_idx, var_batch, seq_str) in enumerate(jobs):
            full_aa = var_batch['aa']
            print(f"  AF2 [{i+1}/{n_variants}] ({time.time()-t0:.0f}s)...", end=' ', flush=True)
            labels = real_af2_labels(full_aa, epitope_seq=epitope_seq,
                                     num_recycle=num_recycle, use_wsl=False)
            if labels is None:
                af2_results.append(None)
                print("FAIL")
            else:
                af2_results.append({
                    'success': True,
                    'iptm': float(labels['af2_iptm']),
                    'plddt_seq': labels['af2_plddt'].tolist(),
                    'pae': labels['af2_pae_matrix'].tolist(),
                })
                print("OK")

    # Phase 3: Write LMDB
    train_env = lmdb.open(os.path.join(dst_dir, 'train.lmdb'), map_size=1 << 35)
    key_idx = 0
    n_written, n_af2_fail, n_none, n_no_success = 0, 0, 0, 0
    import numpy as np

    for (s_idx, d_idx, var_batch, seq_str), af2_r in zip(jobs, af2_results):
        if af2_r is None:
            n_none += 1
            n_af2_fail += 1
            continue
        if not af2_r.get('success'):
            n_no_success += 1
            if n_no_success <= 3:
                err = af2_r.get('error', 'no error field')
                print(f"  [FAIL #{n_no_success}] id={af2_r.get('id','?')} error={str(err)[:120]}", flush=True)
            n_af2_fail += 1
            continue
        labels = {
            'af2_plddt': torch.tensor(np.array(af2_r.get('plddt_seq', [0.5])), dtype=torch.float32),
            'af2_iptm': torch.tensor(float(af2_r.get('iptm', 0.0)), dtype=torch.float32),
            'af2_pae_matrix': torch.tensor(np.array(af2_r.get('pae', [[0.0]])), dtype=torch.float32) / 31.0,
        }
        _write_entry(train_env, key_idx, s_idx, d_idx, var_batch, labels)
        key_idx += 1
        n_written += 1

    with train_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(key_idx))
    train_env.close()

    print(f"\nDone. Wrote {n_written} design-variant entries "
          f"({n_af2_fail} AF2 failures: {n_none} None, {n_no_success} no-success) in {time.time()-t0:.0f}s")
    print(f"Output: {os.path.join(dst_dir, 'train.lmdb')}")
    if n_written == 0:
        print("WARNING: no entries written — check BFN/AF2 availability and gen_mask.")


def main():
    p = argparse.ArgumentParser(description="Build V14 real-AF2 design-variant dataset")
    p.add_argument('--src', default=SRC_LMDB)
    p.add_argument('--dst', default=DST_DIR)
    p.add_argument('--n-scaffolds', type=int, default=1000)
    p.add_argument('--k-variants', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-recycle', type=int, default=2)
    p.add_argument('--smoke-test', action='store_true',
                   help="Quick run: 10 scaffolds × 4 variants (validated by the runner)")
    p.add_argument('--no-wsl', action='store_true',
                   help='Use local CPU JAX instead of WSL2 GPU batch processor')
    p.add_argument('--epitope-seq', default=DEFAULT_EPITOPE,
                   help='Epitope sequence for AF2 multimer partner (default: Abeta42)')
    args = p.parse_args()
    if args.smoke_test:
        args.k_variants = 4
    build(args.src, args.dst, args.n_scaffolds, args.k_variants,
          args.device, args.num_recycle, args.smoke_test,
          use_wsl=not args.no_wsl, epitope_seq=args.epitope_seq)


if __name__ == '__main__':
    main()
