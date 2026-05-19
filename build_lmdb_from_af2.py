#!/usr/bin/env python
"""Build LMDB from existing AF2 results — matches PDBs to AF2 outputs by sequence."""
import sys, os, json, pickle, time
from pathlib import Path
import torch
import numpy as np

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from antibodydesignbfn.datasets.protein import preprocess_protein_structure
from antibodydesignbfn.utils.train import recursive_to
from antibodydesignbfn.utils.data import PaddingCollate
from antibodydesignbfn.utils.transforms import get_transform
from Bio.PDB import PDBParser

PROJECT_DIR = Path(__file__).parent
pdb_dir = PROJECT_DIR / 'data' / 'general_proteins'
af2_dir = PROJECT_DIR / 'data' / 'confidence_dataset' / 'af2_results'
output_dir = PROJECT_DIR / 'data' / 'confidence_dataset'

aa3_to_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def load_pdb_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', str(pdb_path))
    model = structure[0]
    chains = {}
    for chain in model:
        seq = []
        for res in chain:
            if res.get_resname() in aa3_to_1:
                seq.append(aa3_to_1[res.get_resname()])
        if seq:
            chains[chain.id] = ''.join(seq)
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


# ── Build AF2 mapping ──
af2_map = {}  # sequence -> {dir, scores, pae}
for d in sorted(af2_dir.iterdir()):
    if not d.is_dir():
        continue
    pdb_files = sorted(d.glob('*_unrelaxed_rank_001_*.pdb'))
    scores_files = sorted(d.glob('*_scores_rank_001_*.json'))
    pae_files = sorted(d.glob('*_predicted_aligned_error_v1.json'))

    if not pdb_files:
        print(f'  SKIP {d.name}: no PDB output (still running or failed)')
        continue
    if not scores_files:
        print(f'  SKIP {d.name}: no scores JSON')
        continue

    chains = load_pdb_sequence(str(pdb_files[0]))
    if chains:
        seq = list(chains.values())[0]
        af2_map[seq] = {
            'dir': d.name,
            'scores': str(scores_files[0]),
            'pae': str(pae_files[0]) if pae_files else None,
        }
        print(f'  AF2 {d.name}: {seq[:40]}... ({len(seq)} aa)')

print(f'\n{len(af2_map)} complete AF2 results')

# ── Process PDBs ──
entries = []
pdb_files = sorted(pdb_dir.glob('*.pdb'))
print(f'Processing {len(pdb_files)} PDB files...\n')

for i, pdb_path in enumerate(pdb_files):
    pdb_id = pdb_path.stem
    print(f'[{i+1}/{len(pdb_files)}] {pdb_id}...', end=' ', flush=True)

    batch, seq, n_res = preprocess_pdb_for_bfn(str(pdb_path))
    if batch is None:
        print('FAILED (preprocess)')
        continue

    # Exact sequence match
    match = af2_map.get(seq)
    if match is None:
        # Try substring match
        for af2_seq, af2_data in af2_map.items():
            if seq in af2_seq or af2_seq in seq:
                match = af2_data
                break
    if match is None:
        print(f'NO AF2 MATCH (seq={seq[:40]}...)')
        continue

    # Load AF2 scores
    with open(match['scores']) as f:
        scores = json.load(f)
    plddt_array = scores.get('plddt', [])
    iptm = scores.get('iptm') or scores.get('ptm') or 0.5

    # Load PAE
    pae_matrix = None
    if match['pae']:
        with open(match['pae']) as f:
            pae_data = json.load(f)
        pae_matrix = pae_data.get('predicted_aligned_error') or pae_data.get('pae')

    # pLDDT — normalize from [0,100] to [0,1]
    if len(plddt_array) < n_res:
        plddt_array = list(plddt_array) + [50.0] * (n_res - len(plddt_array))
    af2_plddt = torch.tensor([v / 100.0 for v in plddt_array[:n_res]], dtype=torch.float32)
    af2_iptm = torch.tensor(iptm, dtype=torch.float32)

    # PAE matrix
    if pae_matrix is not None:
        pae_t = torch.tensor(pae_matrix, dtype=torch.float32)
        if pae_t.dim() == 2:
            if pae_t.shape[0] < n_res:
                pae_t = torch.nn.functional.pad(pae_t, (0, n_res - pae_t.shape[0], 0, n_res - pae_t.shape[1]))
            af2_pae = pae_t[:n_res, :n_res]
        else:
            af2_pae = torch.zeros(n_res, n_res)
    else:
        af2_pae = torch.zeros(n_res, n_res)

    # Clean batch for serialization
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

    entry = {
        'pdb_id': pdb_id,
        'sequence': seq,
        'batch': batch_clean,
        'af2_plddt': af2_plddt.cpu(),
        'af2_iptm': af2_iptm.cpu(),
        'af2_pae_matrix': af2_pae.cpu(),
    }
    entries.append(entry)
    print(f'OK (L={n_res}, pLDDT={af2_plddt.mean():.3f}, ipTM={af2_iptm.item():.3f})')

print(f'\nMatched: {len(entries)}/{len(pdb_files)}')

if not entries:
    print('No entries to save!')
    sys.exit(1)

# Split train/val
np.random.seed(2026)
indices = np.random.permutation(len(entries))
split_idx = int(len(entries) * 0.8)
train_entries = [entries[i] for i in indices[:split_idx]]
val_entries = [entries[i] for i in indices[split_idx:]]

# Save LMDB
import lmdb
for name, ents in [('confidence_train.lmdb', train_entries), ('confidence_val.lmdb', val_entries)]:
    db_path = str(output_dir / name)
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    env = lmdb.open(db_path, map_size=50 * 1024 * 1024)  # 50 MB, enough for ~20 entries with tensors
    with env.begin(write=True) as txn:
        for i, entry in enumerate(ents):
            txn.put(f'{i:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(ents)))
    env.close()
    print(f'Saved {len(ents)} entries to {name}')

# Summary
summary = {
    'n_train': len(train_entries),
    'n_val': len(val_entries),
    'split_ratio': [0.8, 0.2],
    'pdb_dir': str(pdb_dir),
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
with open(output_dir / 'dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nDone! {len(entries)} entries saved to {output_dir}')
