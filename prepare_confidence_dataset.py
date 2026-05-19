#!/usr/bin/env python
"""Prepare dataset for BFN confidence head fine-tuning.

Workflow:
  1. Take a set of single-chain protein PDB files
  2. Extract native sequence from each PDB
  3. Run AF2 on native sequence to get ground-truth pLDDT/ipTM/PAE
  4. Preprocess PDB into BFN batch format (MaskRegion full → MergeProtein → PatchProtein)
  5. Save as LMDB: each entry = {batch_dict, af2_plddt, af2_iptm, af2_pae_matrix}

Usage:
  python prepare_confidence_dataset.py --pdb_dir ./data/general_proteins \
      --output ./data/confidence_dataset --split train 0.8
"""
import sys, os, json, time, argparse, pickle
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from collections import OrderedDict

PROJECT_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description='Prepare BFN confidence fine-tuning dataset')
    p.add_argument('--pdb_dir', required=True, help='Directory of single-chain protein PDB files')
    p.add_argument('--output', default='./data/confidence_dataset', help='Output LMDB directory')
    p.add_argument('--split', type=float, nargs=2, default=[0.8, 0.2],
                   help='Train/val split ratios (e.g. 0.8 0.2)')
    p.add_argument('--af2_timeout', type=int, default=600, help='AF2 timeout per protein (seconds)')
    p.add_argument('--af2_num_recycle', type=int, default=1, help='AF2 recycling steps')
    p.add_argument('--max_proteins', type=int, default=100, help='Max proteins to process')
    p.add_argument('--no_af2', action='store_true', help='Skip AF2 (use dummy targets for testing)')
    p.add_argument('--resume', action='store_true', help='Resume: skip proteins already in LMDB')
    p.add_argument('--seed', type=int, default=2026, help='Random seed for train/val split')
    return p.parse_args()


def load_pdb_sequence(pdb_path):
    """Extract amino acid sequence from a PDB file using Biopython."""
    from Bio.PDB import PDBParser
    aa3_to_1 = {
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
        'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
        'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
    }
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_path)
    model = structure[0]
    # Collect all chains' sequences
    chains = {}
    for chain in model:
        seq = []
        for res in chain:
            if res.get_resname() in aa3_to_1:
                seq.append(aa3_to_1[res.get_resname()])
        if seq:
            chains[chain.id] = ''.join(seq)
    return chains


def preprocess_pdb_for_bfn(pdb_path, chain_id=None):
    """Convert a PDB to a BFN-compatible batch dict.
    Auto-detects first available chain if chain_id not specified.
    Returns (batch, sequence, num_residues).
    """
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform

    # Auto-detect chain ID if not specified
    if chain_id is None:
        chains = load_pdb_sequence(pdb_path)
        if not chains:
            return None, None, 0
        chain_id = list(chains.keys())[0]
        chain_seq = chains[chain_id]
    else:
        chain_seq = None

    structure = preprocess_protein_structure(pdb_path, chain_ids=[chain_id])
    if structure is None:
        return None, None, 0

    # Extract sequence from the chain data
    chain_data = structure['chains'][0]['data']
    aa_indices = chain_data['aa']
    AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
    seq = ''.join(AA_LETTERS[a] if 0 <= a < 20 else 'X' for a in aa_indices.cpu())
    n_res = len(seq)

    if chain_seq is not None and len(chain_seq) != n_res:
        print(f'  Note: PDB residue count ({n_res}) differs from Bio.PDB seq ({len(chain_seq)})')

    # Mask the entire sequence as "design region" so BFN sees all features
    transform = get_transform([
        {'type': 'mask_region', 'regions': {chain_id: list(range(n_res))}},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    data = transform(structure)
    batch = recursive_to(data, 'cpu')
    # PaddingCollate is applied later by the DataLoader — do NOT add batch dim here
    return batch, seq, n_res


def run_af2_on_sequence(sequence, pdb_id, output_dir, timeout=600, num_recycle=1):
    """Run AF2 on a single sequence and extract confidence scores.
    Returns {plddt_array, iptm, ptm, pae_matrix, pdb_path} or None on failure.
    """
    from af2_validator import validate_sequences
    results = validate_sequences(
        [sequence],
        output_dir=output_dir,
        num_recycle=num_recycle,
        stop_at_score=50,
        timeout=timeout,
    )
    if not results or not results[0].get('success'):
        return None

    r = results[0]

    # Read per-residue pLDDT from scores JSON
    result_dir = r.get('result_dir')
    plddt_array = []
    pae_matrix = None
    if result_dir:
        import glob as _glob
        scores_files = sorted(Path(result_dir).glob('*_scores_rank_001_*.json'))
        if scores_files:
            with open(scores_files[0]) as f:
                scores = json.load(f)
            plddt_array = scores.get('plddt', [])
            pae_matrix = scores.get('pae')

    # pLDDT from result is already in [0,1] (normalized by af2_validator)
    plddt = r.get('plddt', 0.0)  # already [0,1]
    iptm = r.get('iptm') or r.get('ptm', 0.0)
    ptm = r.get('ptm', 0.0)

    return {
        'plddt_array': plddt_array if plddt_array else [plddt] * len(sequence),
        'iptm': iptm,
        'ptm': ptm,
        'pae_matrix': pae_matrix,
        'pdb_path': r.get('pdb_path'),
    }


def build_lmdb_entry(batch, seq, af2_result, pdb_id):
    """Build a single LMDB entry by combining BFN batch with AF2 ground truth."""
    n_res = batch['aa'].shape[1]
    plddt_array = af2_result['plddt_array']

    # Pad/pad pLDDT to match batch size
    if len(plddt_array) < n_res:
        plddt_array = list(plddt_array) + [0.0] * (n_res - len(plddt_array))
    af2_plddt = torch.tensor(plddt_array[:n_res], dtype=torch.float32)

    # ipTM
    af2_iptm = torch.tensor(af2_result['iptm'], dtype=torch.float32)

    # PAE matrix — fill with identity-like if missing
    pae = af2_result.get('pae_matrix')
    if pae is not None:
        pae_t = torch.tensor(pae, dtype=torch.float32)
        if pae_t.dim() == 2:
            if pae_t.shape[0] < n_res:
                pae_t = torch.nn.functional.pad(pae_t, (0, n_res - pae_t.shape[0], 0, n_res - pae_t.shape[1]))
            af2_pae = pae_t[:n_res, :n_res]
        else:
            af2_pae = torch.zeros(n_res, n_res)
    else:
        af2_pae = torch.zeros(n_res, n_res)

    # Keep only CPU-compatible tensors (remove any _extra or non-serializable fields)
    batch_clean = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_clean[k] = v.cpu()
        elif isinstance(v, (list, tuple)):
            batch_clean[k] = v
        elif isinstance(v, (int, float, str, bool)):
            batch_clean[k] = v
        else:
            batch_clean[k] = str(v)

    return {
        'pdb_id': pdb_id,
        'sequence': seq,
        'batch': batch_clean,
        'af2_plddt': af2_plddt.cpu(),
        'af2_iptm': af2_iptm.cpu(),
        'af2_pae_matrix': af2_pae.cpu(),
    }


def save_lmdb(entries, db_path):
    """Save entries to an LMDB database."""
    # Simple pickle-based storage (LMDB requires lmdb package)
    # Fall back to a directory of pickle files if lmdb not available
    os.makedirs(db_path, exist_ok=True)

    try:
        import lmdb
        env = lmdb.open(db_path, map_size=2 * 1024 * 1024 * 1024)  # 2GB
        with env.begin(write=True) as txn:
            for i, entry in enumerate(entries):
                key = f'{i:08d}'.encode()
                value = pickle.dumps(entry)
                txn.put(key, value)
            txn.put(b'__len__', pickle.dumps(len(entries)))
        env.close()
        print(f'  Saved {len(entries)} entries to LMDB: {db_path}')
    except ImportError:
        # Fallback: pickle files in directory
        meta = {'n_entries': len(entries)}
        with open(os.path.join(db_path, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        for i, entry in enumerate(entries):
            with open(os.path.join(db_path, f'{i:08d}.pkl'), 'wb') as f:
                pickle.dump(entry, f)
        print(f'  Saved {len(entries)} entries to pickle dir: {db_path}')


def load_lmdb_keys(db_path):
    """Load existing keys from LMDB for resume."""
    try:
        import lmdb
        env = lmdb.open(db_path, readonly=True)
        with env.begin() as txn:
            keys = [k.decode() for k in txn.cursor().iternext(keys=True) if k != b'__len__']
        env.close()
        return set(keys)
    except ImportError:
        meta_path = os.path.join(db_path, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            return set(f'{i:08d}' for i in range(meta['n_entries']))
        return set()


# ── Main ──
if __name__ == '__main__':
    args = parse_args()
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output)
    af2_dir = output_dir / 'af2_results'

    if not pdb_dir.exists():
        print(f'ERROR: PDB directory not found: {pdb_dir}')
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    af2_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDB files
    pdb_files = sorted(pdb_dir.glob('*.pdb')) + sorted(pdb_dir.glob('*.ent'))
    pdb_files = pdb_files[:args.max_proteins]

    if not pdb_files:
        print(f'ERROR: No PDB files found in {pdb_dir}')
        sys.exit(1)

    print(f'Found {len(pdb_files)} PDB files')
    print(f'Output: {output_dir}')
    print(f'AF2 timeout: {args.af2_timeout}s | Recycle: {args.af2_num_recycle}')
    if args.no_af2:
        print('WARNING: AF2 disabled — using dummy confidence targets')

    # Resume: skip already-processed
    processed = set()
    if args.resume:
        for split_name in ['train', 'val']:
            db_path = str(output_dir / f'confidence_{split_name}.lmdb')
            if os.path.exists(db_path):
                processed |= load_lmdb_keys(db_path)

    all_entries = []
    n_processed = 0
    n_skipped = 0

    for i, pdb_path in enumerate(pdb_files):
        pdb_id = pdb_path.stem
        key = f'{pdb_id}'
        if key in processed:
            n_skipped += 1
            print(f'[{i+1}/{len(pdb_files)}] {pdb_id}: SKIPPED (already processed)')
            continue

        print(f'[{i+1}/{len(pdb_files)}] {pdb_id}: processing...')
        t_start = time.time()

        # Step 1: Preprocess PDB
        batch, seq, n_res = preprocess_pdb_for_bfn(str(pdb_path))
        if batch is None:
            print(f'  FAILED: could not preprocess PDB')
            continue

        print(f'  Sequence: {seq[:50]}... ({n_res} aa)')

        # Step 2: AF2 on native sequence
        if args.no_af2:
            af2_result = {
                'plddt_array': [0.8] * n_res,
                'iptm': 0.7,
                'ptm': 0.7,
                'pae_matrix': None,
                'pdb_path': None,
            }
        else:
            af2_result = run_af2_on_sequence(
                seq, pdb_id,
                output_dir=str(af2_dir),
                timeout=args.af2_timeout,
                num_recycle=args.af2_num_recycle,
            )
            if af2_result is None:
                print(f'  AF2 FAILED for {pdb_id} — using fallback')
                af2_result = {
                    'plddt_array': [0.5] * n_res,
                    'iptm': 0.4,
                    'ptm': 0.4,
                    'pae_matrix': None,
                    'pdb_path': None,
                }

        print(f'  AF2: pLDDT={np.mean(af2_result["plddt_array"]):.3f}  '
              f'ipTM={af2_result["iptm"]:.3f}  pTM={af2_result["ptm"]:.3f}')

        # Step 3: Build LMDB entry
        entry = build_lmdb_entry(batch, seq, af2_result, pdb_id)
        all_entries.append(entry)
        n_processed += 1

        elapsed = time.time() - t_start
        print(f'  Done in {elapsed:.0f}s')

    if n_skipped > 0:
        print(f'\nSkipped {n_skipped} already-processed proteins')

    if not all_entries:
        print('No new entries to save.')
        sys.exit(0)

    # Split into train/val
    np.random.seed(args.seed)
    indices = np.random.permutation(len(all_entries))
    split_idx = int(len(all_entries) * args.split[0])
    train_entries = [all_entries[i] for i in indices[:split_idx]]
    val_entries = [all_entries[i] for i in indices[split_idx:]]

    print(f'\nSaving: {len(train_entries)} train + {len(val_entries)} val')

    save_lmdb(train_entries, str(output_dir / 'confidence_train.lmdb'))
    save_lmdb(val_entries, str(output_dir / 'confidence_val.lmdb'))

    # Also save a quick summary
    summary = {
        'n_train': len(train_entries),
        'n_val': len(val_entries),
        'split_ratio': args.split,
        'pdb_dir': str(pdb_dir),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDone! {len(all_entries)} entries → {output_dir}')
