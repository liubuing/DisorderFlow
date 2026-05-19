#!/usr/bin/env python
"""
Strategy 3: Build BFN confidence dataset from local ColabFold AF2 results.

Scans the af2_results directory for completed ColabFold runs, reads the
predicted PDB + pLDDT scores + PAE matrices, preprocesses each PDB into
a BFN batch, and stores everything in LMDB format.
"""
import sys, os, json, pickle, time, shutil
from pathlib import Path
import numpy as np
import torch

if sys.platform == 'win32':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from antibodydesignbfn.datasets.protein import preprocess_protein_structure
from antibodydesignbfn.utils.train import recursive_to
from antibodydesignbfn.utils.transforms import get_transform

AA3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def load_pdb_sequence(pdb_path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', str(pdb_path))
    chains = {}
    for chain in structure[0]:
        seq = ''.join(AA3_TO_1.get(res.get_resname(), '') for res in chain
                      if res.get_resname() in AA3_TO_1)
        if seq:
            chains[chain.id] = seq
    return chains


def preprocess_pdb_for_bfn(pdb_path):
    chains = load_pdb_sequence(pdb_path)
    if not chains:
        return None, None, 0
    chain_id = list(chains.keys())[0]

    structure = preprocess_protein_structure(str(pdb_path), chain_ids=[chain_id])
    if structure is None:
        return None, None, 0

    chain_data = structure['chains'][0]['data']
    aa_indices = chain_data['aa']
    seq = ''.join(AA_LETTERS[a] if 0 <= a < 20 else 'X' for a in aa_indices.cpu())
    n_res = len(seq)

    transform = get_transform([
        {'type': 'mask_region', 'regions': {chain_id: list(range(n_res))}},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    data = transform(structure)
    data = recursive_to(data, 'cpu')
    return data, seq, n_res


def compute_ptm_from_pae(pae_matrix):
    pae = np.array(pae_matrix, dtype=np.float64)
    L = pae.shape[0]
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    ptm_scores = []
    for i in range(L):
        f_ij = 1.0 / (1.0 + (pae[i, :] / d0) ** 2)
        ptm_scores.append(np.mean(f_ij))
    return float(max(ptm_scores))


def main():
    import argparse
    p = argparse.ArgumentParser(description='Build dataset from local ColabFold results')
    p.add_argument('--af2_results_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset' / 'af2_results'),
                   help='Directory containing ColabFold output subdirectories')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset_colabfold'),
                   help='Output directory for LMDB')
    p.add_argument('--max_proteins', type=int, default=200)
    args = p.parse_args()

    results_dir = Path(args.af2_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find completed ColabFold runs
    completed = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / 'design_0.done.txt').exists():
            pdb_path = d / 'design_0_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb'
            scores_path = d / 'design_0_scores_rank_001_alphafold2_ptm_model_1_seed_000.json'
            pae_path = d / 'design_0_predicted_aligned_error_v1.json'
            if pdb_path.exists() and scores_path.exists():
                completed.append((d.name, str(pdb_path), str(scores_path), str(pae_path)))

    print(f'Found {len(completed)} completed ColabFold runs')
    print(f'Processing up to {args.max_proteins}...')

    np.random.seed(2026)
    if len(completed) > args.max_proteins:
        completed = completed[:args.max_proteins]

    entries = []
    n_skipped = 0

    for i, (name, pdb_path, scores_path, pae_path) in enumerate(completed):
        print(f'[{i+1}/{len(completed)}] {name}...', end=' ', flush=True)

        # Load scores
        try:
            with open(scores_path) as f:
                scores = json.load(f)
        except Exception as e:
            print(f'SKIP (scores: {e})')
            n_skipped += 1
            continue

        # ColabFold pLDDT is 0-100 scale, convert to [0,1]
        plddt_raw = scores.get('plddt', [])
        if not plddt_raw:
            print('SKIP (no pLDDT)')
            n_skipped += 1
            continue
        plddt_list = [v / 100.0 for v in plddt_raw]

        # Load PAE matrix
        pae_matrix = None
        if os.path.exists(pae_path):
            try:
                with open(pae_path) as f:
                    pae_data = json.load(f)
                if isinstance(pae_data, list) and len(pae_data) > 0:
                    pae_matrix = pae_data[0].get('predicted_aligned_error') or pae_data[0].get('pae')
                elif isinstance(pae_data, dict):
                    pae_matrix = pae_data.get('predicted_aligned_error') or pae_data.get('pae')
            except Exception:
                pass

        # Preprocess PDB
        try:
            batch, seq, n_res = preprocess_pdb_for_bfn(pdb_path)
        except Exception as e:
            print(f'SKIP (preprocess: {e})')
            n_skipped += 1
            continue

        if batch is None:
            print('SKIP (batch is None)')
            n_skipped += 1
            continue

        # Align pLDDT to BFN sequence length
        if len(plddt_list) < n_res:
            plddt_list += [0.5] * (n_res - len(plddt_list))
        plddt_tensor = torch.tensor(plddt_list[:n_res], dtype=torch.float32)

        # Build PAE tensor
        if pae_matrix is not None:
            pae_np = np.array(pae_matrix, dtype=np.float64)
            if pae_np.shape[0] < n_res:
                pad_w = n_res - pae_np.shape[0]
                pae_np = np.pad(pae_np, ((0, pad_w), (0, pad_w)), constant_values=0)
            pae_np = pae_np[:n_res, :n_res]
            pae_t = torch.tensor(pae_np, dtype=torch.float32)
            iptm = compute_ptm_from_pae(pae_np)
        else:
            pae_t = torch.zeros(n_res, n_res, dtype=torch.float32)
            iptm = 0.5

        # Try to get pTM from scores, otherwise compute from PAE
        ptm_from_file = scores.get('ptm')
        if ptm_from_file is not None:
            iptm = float(ptm_from_file)
        iptm_tensor = torch.tensor(iptm, dtype=torch.float32)

        # Clean batch for pickling
        batch_clean = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_clean[k] = v.cpu()
            elif isinstance(v, (list, tuple, int, float, str, bool)):
                batch_clean[k] = v
            else:
                batch_clean[k] = str(v)

        entry = {
            'pdb_id': name,
            'sequence': seq,
            'batch': batch_clean,
            'af2_plddt': plddt_tensor.cpu(),
            'af2_iptm': iptm_tensor.cpu(),
            'af2_pae_matrix': pae_t.cpu(),
        }
        entries.append(entry)
        print(f'OK (L={n_res}, pLDDT={plddt_tensor.mean():.3f}, pTM={iptm:.3f})')

    print(f'\n{"="*60}')
    print(f'  Results: {len(entries)}/{len(completed)} succeeded ({n_skipped} skipped)')
    print(f'{"="*60}')

    if not entries:
        print('No entries!')
        sys.exit(1)

    # Split
    np.random.seed(2026)
    indices = np.random.permutation(len(entries))
    split_idx = int(len(entries) * 0.8)
    train_entries = [entries[i] for i in indices[:split_idx]]
    val_entries = [entries[i] for i in indices[split_idx:]]

    import lmdb
    for name, ents in [('confidence_train.lmdb', train_entries), ('confidence_val.lmdb', val_entries)]:
        db_path = str(output_dir / name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        sample = pickle.dumps(ents[0]) if ents else b'x'
        est_size = max(len(sample) * max(len(ents), 1) * 2 + 1024 * 1024, 10 * 1024 * 1024)
        env = lmdb.open(db_path, map_size=est_size)
        with env.begin(write=True) as txn:
            for j, entry in enumerate(ents):
                txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
            txn.put(b'__len__', pickle.dumps(len(ents)))
        env.close()
        print(f'Saved {len(ents)} entries to {name}')

    summary = {
        'n_train': len(train_entries),
        'n_val': len(val_entries),
        'source': 'Local ColabFold AF2 results',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDone! Combined with EBI dataset for larger training corpus.')


if __name__ == '__main__':
    main()
