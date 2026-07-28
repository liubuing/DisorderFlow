#!/usr/bin/env python
"""
Build Multi-Conformation LMDB from AF2 ensemble predictions.

For each IDP in the source confidence dataset:
  1. Run AF2 multimer × N seeds → get backbone coordinates
  2. Align conformations to reference (first seed) via Kabsch on Cα
  3. Compute per-residue Cα RMSF → tanh-normalized disorder label
  4. Store: N sets of backbone coordinates + continuous RMSF label

Output: data/confidence_conformation_v1/{train,val}.lmdb

Usage:
  python scripts/build/build_conformation_dataset.py --n_seeds 5 --max_idp 10 --device xpu
  python scripts/build/build_conformation_dataset.py --n_seeds 5 --resume_from 42
"""

import os, sys, json, time, pickle, argparse, lmdb, traceback
import numpy as np
import torch
from tqdm.auto import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'modules'))
os.chdir(_project_root)

from af2_jax_runner import run_multimer_prediction

SRC_LMDB = 'data/confidence_merged_v11'
DST_DIR = 'data/confidence_conformation_v1'

# AlphaFold atom indexing — backbone atoms
CA_IDX = 1   # Cα
N_IDX = 0    # N
C_IDX = 2    # C

# Restrict to IDP entries under this length (AF2 memory limit)
MAX_SEQ_LEN = 400


def kabsch_align(P, Q):
    """Align P to Q via Kabsch algorithm. Returns rotation matrix R and translation t."""
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P_centered = P - p_mean
    Q_centered = Q - q_mean
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = q_mean - R @ p_mean
    return R, t


def compute_rmsf(positions_list, ca_idx=CA_IDX):
    """Compute per-residue Cα RMSF from list of aligned conformations.

    Args:
        positions_list: list of (N_res, 37, 3) arrays

    Returns:
        rmsf: (N_res,) array in Angstroms
    """
    n_conf = len(positions_list)
    ca_positions = [p[:, ca_idx, :] for p in positions_list]  # each (N_res, 3)

    # Align all to first conformation
    ref = ca_positions[0]
    aligned = [ref]
    for i in range(1, n_conf):
        R, t = kabsch_align(ca_positions[i], ref)
        aligned_i = ca_positions[i] @ R.T + t
        aligned.append(aligned_i)

    aligned = np.stack(aligned, axis=0)  # (n_conf, N_res, 3)
    mean_pos = aligned.mean(axis=0)  # (N_res, 3)
    sq_diffs = ((aligned - mean_pos) ** 2).sum(axis=-1)  # (n_conf, N_res)
    rmsf = np.sqrt(sq_diffs.mean(axis=0))  # (N_res,)
    return rmsf


def rmsf_to_disorder(rmsf, scale=2.0):
    """Convert RMSF (Å) to disorder score [0, 1] via tanh."""
    return np.tanh(rmsf / scale)


def generate_conformations(sequence, n_seeds=5, num_recycle=1, verbose=False):
    """Run AF2 × N seeds on a single-chain sequence.

    Returns:
        positions: list of (N_res, 37, 3) arrays, or None if failed
        rmsf: (N_res,) array, or None if failed
    """
    seeds = [1, 3, 5, 7, 9][:n_seeds]
    positions = []
    for seed in seeds:
        r = run_multimer_prediction(
            sequence, '',  # single-chain: empty epitope
            num_recycle=num_recycle,
            jax_random_seed=seed,
            return_structure=True)
        if not r.get('success'):
            if verbose:
                print(f'  seed={seed} FAILED: {r.get("error", "?")[:100]}')
            return None, None
        pos = r.get('final_atom_positions')
        if pos is None:
            return None, None
        positions.append(pos)

    rmsf = compute_rmsf(positions)
    return positions, rmsf


def process_idp_entry(entry, n_seeds=5, num_recycle=1):
    """Process one IDP entry: generate conformations, compute RMSF."""
    sequence = entry['sequence']
    seq_clean = ''.join(aa for aa in sequence.upper()
                        if aa in 'ARNDCQEGHILKMFPSTWYV')

    if len(seq_clean) > MAX_SEQ_LEN:
        return None

    positions, rmsf = generate_conformations(seq_clean, n_seeds=n_seeds,
                                             num_recycle=num_recycle)
    if positions is None:
        return None

    disorder = rmsf_to_disorder(rmsf)
    ca_positions = [p[:, CA_IDX, :] for p in positions]  # (N_res, 3) each

    # Only store Cα positions to save space (backbone atoms can be
    # reconstructed from Cα trace + ideal geometry during training)
    return {
        'pdb_id': entry.get('pdb_id', ''),
        'sequence': seq_clean,
        'n_conformations': len(ca_positions),
        'ca_positions': [p.astype(np.float32) for p in ca_positions],
        'rmsf': rmsf.astype(np.float32),
        'disorder_label': disorder.astype(np.float32),
        'batch': entry['batch'],
    }


def build_lmdb(src_lmdb_path, dst_lmdb_path, n_seeds, num_recycle, max_idp=None,
               resume_from=0):
    """Build conformation LMDB from source confidence dataset."""
    src_env = lmdb.open(src_lmdb_path, readonly=True, lock=False)
    with src_env.begin() as txn:
        total = pickle.loads(txn.get(b'__len__'))

    # Collect IDP indices
    idp_indices = []
    with src_env.begin() as txn:
        for i in range(total):
            entry = pickle.loads(txn.get(f'{i:08d}'.encode()))
            if entry.get('is_idp', False):
                seq = entry.get('sequence', '')
                seq_clean = ''.join(aa for aa in seq.upper()
                                    if aa in 'ARNDCQEGHILKMFPSTWYV')
                if len(seq_clean) <= MAX_SEQ_LEN:
                    idp_indices.append(i)

    # Apply resume_from first (absolute offset on full list),
    # then limit batch size — otherwise max_idp truncation
    # pollutes the window that resume_from operates on.
    idp_indices = idp_indices[resume_from:]
    if max_idp is not None:
        idp_indices = idp_indices[:max_idp]
    n_idp = len(idp_indices)

    print(f'IDP entries to process: {n_idp} (total IDPs in source: '
          f'{len(idp_indices) + resume_from}, resume_from={resume_from})')
    print(f'n_seeds={n_seeds}, num_recycle={num_recycle}')
    print(f'Estimated AF2 runs: {n_idp * n_seeds} (~{n_idp * n_seeds * 17 / 60:.0f} min)')

    os.makedirs(os.path.dirname(dst_lmdb_path), exist_ok=True)
    dst_env = lmdb.open(dst_lmdb_path, map_size=32 * 1024 * 1024 * 1024, subdir=True)

    n_success = 0
    t_start = time.time()

    with dst_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(0))  # placeholder, updated at end

    for local_idx, src_idx in enumerate(tqdm(idp_indices, desc='Processing IDPs')):
        global_idx = local_idx + resume_from
        try:
            with src_env.begin() as txn:
                entry = pickle.loads(txn.get(f'{src_idx:08d}'.encode()))

            result = process_idp_entry(entry, n_seeds=n_seeds,
                                       num_recycle=num_recycle)
            if result is None or result.get('n_conformations', 0) < 2:
                continue

            with dst_env.begin(write=True) as txn:
                key = f'{global_idx:08d}'.encode()
                txn.put(key, pickle.dumps(result))

            n_success += 1

            if n_success % 10 == 0:
                elapsed = time.time() - t_start
                rate = n_success / max(elapsed, 1)
                eta = (n_idp - local_idx - 1) / max(rate, 0.001)
                tqdm.write(f'  [{n_success}/{local_idx+1}] success rate={rate:.2f}/s '
                           f'ETA={eta/60:.0f}min')

        except Exception as e:
            tqdm.write(f'  [ERROR idx={src_idx}]: {e}')

    # Update count
    with dst_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(n_success))

    src_env.close()
    dst_env.close()

    elapsed = time.time() - t_start
    print(f'\nDone. {n_success}/{n_idp} succeeded in {elapsed/60:.0f} min.')
    print(f'Output: {dst_lmdb_path}')


def main():
    parser = argparse.ArgumentParser(description='Build multi-conformation LMDB')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--num_recycle', type=int, default=1)
    parser.add_argument('--max_idp', type=int, default=None,
                        help='Limit number of IDPs to process')
    parser.add_argument('--resume_from', type=int, default=0,
                        help='Resume from this IDP index')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'])
    args = parser.parse_args()

    src_path = os.path.join(SRC_LMDB, f'confidence_{args.split}.lmdb')
    dst_path = os.path.join(DST_DIR, f'confidence_{args.split}.lmdb')

    if not os.path.exists(src_path):
        print(f'Source not found: {src_path}')
        sys.exit(1)

    print('=' * 70)
    print(f'Building Conformation Dataset: {args.split}')
    print('=' * 70)

    build_lmdb(src_path, dst_path, n_seeds=args.n_seeds,
               num_recycle=args.num_recycle, max_idp=args.max_idp,
               resume_from=args.resume_from)


if __name__ == '__main__':
    main()
