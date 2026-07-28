"""Shared utilities for dataset builder scripts.
Reduces code duplication across build_*.py files.
"""
import sys, os, json, pickle, shutil
from pathlib import Path
import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Win32 stdout encoding fix
if sys.platform == 'win32':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def compute_ptm_from_pae(pae_matrix):
    """Compute pTM score from PAE matrix (AlphaFold convention).

    pTM = max_i (1/L * sum_j 1/(1 + (PAE_ij / d0)^2))
    d0 = max(1.24 * (L - 15)^(1/3) - 1.8, 0.5)
    """
    pae = np.array(pae_matrix, dtype=np.float64)
    L = pae.shape[0]
    if L < 2:
        return 0.0
    d0 = max(1.24 * (L - 15) ** (1 / 3) - 1.8, 0.5)
    best = 0.0
    for i in range(L):
        f_ij = 1.0 / (1.0 + (pae[i, :] / d0) ** 2)
        best = max(best, float(np.mean(f_ij)))
    return best


def resilient_get(url, timeout=30, retries=3, raise_on_404=False):
    """HTTP GET with retry logic."""
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if raise_on_404:
                r.raise_for_status()
            elif r.status_code != 200:
                return r  # caller handles non-200
            return r
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt+1}/{retries} for {url[:80]}...")
            import time
            time.sleep(2 ** attempt)
    return None


def preprocess_pdb_for_bfn(pdb_path):
    """Preprocess a PDB file into a BFN-compatible batch dict."""
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.data import DEFAULT_NO_PADDING
    from disorderflow.utils.transforms import get_transform

    protein = preprocess_protein_structure(pdb_path, DEFAULT_NO_PADDING)
    batch = next(iter(
        get_transform(DEFAULT_NO_PADDING, 'train', 'cpu')([protein]).values()
    ))
    return recursive_to(batch, 'cpu')


def load_pdb_sequence(pdb_path):
    """Extract amino acid sequence from ATOM records in a PDB file."""
    chain_seqs = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            res_name = line[17:20].strip()
            res_num = int(line[22:26].strip())
            aa = AA3_TO_1.get(res_name)
            if aa is None:
                continue
            key = chain
            if key not in chain_seqs:
                chain_seqs[key] = {}
            chain_seqs[key][res_num] = aa
    seqs = {}
    for chain, residues in chain_seqs.items():
        seqs[chain] = ''.join(aa for _, aa in sorted(residues.items()))
    return seqs


def save_lmdb(db_path, entries):
    """Save list of entries to an LMDB database."""
    import lmdb
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    # Estimate from multiple samples to handle variable protein sizes
    n_samples_est = min(20, len(entries))
    total_sample = sum(len(pickle.dumps(entries[i])) for i in range(n_samples_est))
    avg_entry = total_sample / n_samples_est
    # 5x margin for PAE matrix variability + 200MB overhead + 2GB minimum
    est_size = max(int(avg_entry * len(entries) * 5) + 200 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
    import lmdb
    env = lmdb.open(str(db_path), map_size=est_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()


def split_and_save_entries(entries, output_dir, train_split=0.8, seed=2026):
    """Homology-cluster and save entries as train/val LMDBs."""
    from disorderflow.utils.homology_split import (
        annotate_homology_clusters,
        grouped_train_val_split,
    )

    if not any(entry.get('sequence') for entry in entries):
        raise ValueError('Entries need a sequence field for homology-safe splitting')
    annotate_homology_clusters(entries, {'sequence': 0.3})
    train, val = grouped_train_val_split(
        entries, val_ratio=1.0 - train_split, seed=seed,
        cluster_fields=['sequence_cluster_id'])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_lmdb(output_dir / 'confidence_train.lmdb', train)
    save_lmdb(output_dir / 'confidence_val.lmdb', val)

    import time, json
    summary = {
        'n_train': len(train), 'n_val': len(val),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'split_strategy': 'MMseqs2 sequence homology clusters',
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'  Saved {len(train)} train + {len(val)} val -> {output_dir}')
    return train, val
