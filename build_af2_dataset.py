#!/usr/bin/env python
"""
Strategy 1: Build BFN confidence dataset from EBI AlphaFold Database.

Downloads AF2-predicted mmCIF structures + PAE JSON for small single-chain
proteins. Converts each mmCIF to a temp PDB, uses preprocess_protein_structure()
to build the BFN batch, and stores pLDDT/pTM/PAE as teacher targets.
"""
import sys, os, json, pickle, time, io, tempfile, shutil, re
from pathlib import Path
import requests
import numpy as np
import torch

if sys.platform == 'win32':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from antibodydesignbfn.datasets.protein import preprocess_protein_structure
from antibodydesignbfn.utils.train import recursive_to
from antibodydesignbfn.utils.data import DEFAULT_NO_PADDING
from antibodydesignbfn.utils.transforms import get_transform

AFDB_BASE = 'https://alphafold.ebi.ac.uk/files'
UNIPROT_API = 'https://rest.uniprot.org/uniprotkb/search'

def compute_ptm_from_pae(pae_matrix):
    """Compute pTM score from PAE matrix (AlphaFold convention).

    pTM = max_i (1/L * sum_j 1/(1 + (PAE_ij / d0)^2))
    where d0 = max(1.24 * (L - 15)^(1/3) - 1.8, 0.5)
    """
    pae = np.array(pae_matrix, dtype=np.float64)
    L = pae.shape[0]
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    ptm_scores = []
    for i in range(L):
        f_ij = 1.0 / (1.0 + (pae[i, :] / d0) ** 2)
        ptm_scores.append(np.mean(f_ij))
    return float(max(ptm_scores))

AA3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {c: i for i, c in enumerate(AA_LETTERS)}


def fetch_uniprot_accessions(max_results=300, min_len=50, max_len=250):
    """Fetch reviewed Swiss-Prot entries via UniProt REST API."""
    print(f'Fetching UniProt accessions (length {min_len}-{max_len}, max {max_results})...')
    query = f'(length:[{min_len} TO {max_len}] AND reviewed:true AND fragment:false)'
    url = f'{UNIPROT_API}?query={query}&size={min(500, max_results * 2)}&format=tsv&fields=accession,length,protein_name'
    r = requests.get(url, timeout=30)
    lines = r.text.strip().split('\n')
    accessions = []
    for line in lines[1:max_results + 1]:
        parts = line.split('\t')
        if len(parts) >= 2 and parts[0]:
            accessions.append(parts[0])
    print(f'  Got {len(accessions)} accessions')
    return accessions


def mmcif_to_temp_pdb(cif_text, accession):
    """Convert mmCIF text to a temp PDB file using Bio.PDB. Returns path."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB import PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(accession, io.StringIO(cif_text))

    fd, tmp_path = tempfile.mkstemp(suffix='.pdb', prefix=f'af2_{accession}_')
    os.close(fd)

    io_pdb = PDBIO()
    io_pdb.set_structure(structure)
    io_pdb.save(tmp_path)
    return tmp_path


def load_pdb_sequence(pdb_path):
    """Extract chain sequences from PDB."""
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


def extract_plddt_from_mmcif(cif_text):
    """Extract per-residue pLDDT from mmCIF B-factor column (CA atom)."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('tmp', io.StringIO(cif_text))
    plddt = []
    seq = []
    for chain in structure[0]:
        for res in chain:
            if res.get_resname() not in AA3_TO_1:
                continue
            if 'CA' in res:
                plddt.append(res['CA'].get_bfactor() / 100.0)
                seq.append(AA3_TO_1[res.get_resname()])
    return plddt, ''.join(seq)


def preprocess_pdb_for_bfn(pdb_path):
    """Preprocess a PDB file into BFN batch format."""
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


def main():
    import argparse
    p = argparse.ArgumentParser(description='Build AF2 confidence dataset from EBI AFDB')
    p.add_argument('--max_proteins', type=int, default=200, help='Max proteins')
    p.add_argument('--min_len', type=int, default=50, help='Min sequence length')
    p.add_argument('--max_len', type=int, default=250, help='Max sequence length')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset'),
                   help='Output directory for LMDB')
    p.add_argument('--accessions_file', default=None,
                   help='Pre-fetched UniProt accessions file (one per line)')
    p.add_argument('--start_from', type=int, default=0, help='Resume from this index')
    p.add_argument('--af2_version', type=int, default=6, help='AFDB version (default 6)')
    p.add_argument('--seed', type=int, default=2026, help='Random seed for train/val split')
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get accessions
    if args.accessions_file and os.path.exists(args.accessions_file):
        with open(args.accessions_file) as f:
            accessions = [line.strip() for line in f if line.strip()]
        print(f'Loaded {len(accessions)} from {args.accessions_file}')
    else:
        accessions = fetch_uniprot_accessions(
            max_results=args.max_proteins,
            min_len=args.min_len,
            max_len=args.max_len
        )
        acc_file = output_dir / 'accessions.txt'
        with open(acc_file, 'w') as f:
            for acc in accessions:
                f.write(f'{acc}\n')
        print(f'Saved to {acc_file}')

    n_total = len(accessions)
    print(f'\n{"="*60}')
    print(f'  Processing {n_total} proteins (AFDB v{args.af2_version})')
    print(f'{"="*60}\n')

    entries = []
    n_failed = 0
    n_pae_miss = 0

    for i in range(args.start_from, n_total):
        acc = accessions[i]
        print(f'[{i+1}/{n_total}] {acc}...', end=' ', flush=True)

        # 1. Download mmCIF
        url = f'{AFDB_BASE}/AF-{acc}-F1-model_v{args.af2_version}.cif'
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print('SKIP (no AF2 prediction)')
            n_failed += 1
            continue
        cif_text = r.text

        # 2. Extract pLDDT and sequence from mmCIF
        plddt_list, mmcif_seq = extract_plddt_from_mmcif(cif_text)
        if not plddt_list:
            print('SKIP (no pLDDT)')
            n_failed += 1
            continue

        # 3. Convert to temp PDB and preprocess
        tmp_pdb = None
        try:
            tmp_pdb = mmcif_to_temp_pdb(cif_text, acc)
            batch, seq, n_res = preprocess_pdb_for_bfn(tmp_pdb)
        except Exception as e:
            print(f'SKIP (preprocess: {e})')
            n_failed += 1
            if tmp_pdb and os.path.exists(tmp_pdb):
                os.unlink(tmp_pdb)
            continue

        if tmp_pdb and os.path.exists(tmp_pdb):
            os.unlink(tmp_pdb)

        if batch is None:
            print('SKIP (batch is None)')
            n_failed += 1
            continue

        # 4. Align pLDDT to BFN sequence length
        if len(plddt_list) < n_res:
            plddt_list += [0.5] * (n_res - len(plddt_list))
        plddt_tensor = torch.tensor(plddt_list[:n_res], dtype=torch.float32)

        # 5. Download PAE (v6 format: list of dict)
        pae_matrix = None
        pae_url = f'{AFDB_BASE}/AF-{acc}-F1-predicted_aligned_error_v{args.af2_version}.json'
        try:
            r_pae = requests.get(pae_url, timeout=30)
            if r_pae.status_code == 200:
                pae_data = r_pae.json()
                # v6 format: [{"predicted_aligned_error": [[...], ...], "max_predicted_aligned_error": ...}]
                if isinstance(pae_data, list) and len(pae_data) > 0:
                    pae_matrix = pae_data[0].get('predicted_aligned_error') or pae_data[0].get('pae')
                elif isinstance(pae_data, dict):
                    pae_matrix = pae_data.get('predicted_aligned_error') or pae_data.get('pae')
        except Exception:
            pass


        if pae_matrix is None:
            n_pae_miss += 1
            pae_matrix = [[0.0] * n_res for _ in range(n_res)]

        pae_t = torch.tensor(pae_matrix, dtype=torch.float32)
        if pae_t.dim() == 2:
            if pae_t.shape[0] < n_res:
                pae_t = torch.nn.functional.pad(pae_t, (0, n_res - pae_t.shape[0], 0, n_res - pae_t.shape[1]))
            pae_t = pae_t[:n_res, :n_res]
        else:
            pae_t = torch.zeros(n_res, n_res)

        # Compute pTM from PAE matrix
        pae_np = pae_t[:n_res, :n_res].cpu().numpy()
        iptm = compute_ptm_from_pae(pae_np) if pae_np.shape[0] > 0 else 0.5
        iptm_tensor = torch.tensor(iptm, dtype=torch.float32)

        # 6. Clean batch for pickling
        batch_clean = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_clean[k] = v.cpu()
            elif isinstance(v, (list, tuple, int, float, str, bool)):
                batch_clean[k] = v
            else:
                batch_clean[k] = str(v)

        entry = {
            'pdb_id': acc,
            'sequence': seq,
            'batch': batch_clean,
            'af2_plddt': plddt_tensor.cpu(),
            'af2_iptm': iptm_tensor.cpu(),
            'af2_pae_matrix': pae_t.cpu(),
        }
        entries.append(entry)
        print(f'OK (L={n_res}, pLDDT={plddt_tensor.mean():.3f}, pTM={iptm:.3f})')

    print(f'\n{"="*60}')
    print(f'  Results: {len(entries)}/{n_total} succeeded')
    print(f'  Failed: {n_failed},  Missing PAE: {n_pae_miss}')
    print(f'{"="*60}')

    if not entries:
        print('No entries!')
        sys.exit(1)

    # Split and save
    np.random.seed(args.seed)
    indices = np.random.permutation(len(entries))
    split_idx = int(len(entries) * 0.8)
    train_entries = [entries[i] for i in indices[:split_idx]]
    val_entries = [entries[i] for i in indices[split_idx:]]

    import lmdb
    for name, ents in [('confidence_train.lmdb', train_entries), ('confidence_val.lmdb', val_entries)]:
        db_path = str(output_dir / name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        # Estimate needed map size
        sample = pickle.dumps(ents[0])
        est_size = len(sample) * len(ents) * 2 + 1024 * 1024
        map_size = max(est_size, 10 * 1024 * 1024)
        env = lmdb.open(db_path, map_size=map_size)
        with env.begin(write=True) as txn:
            for j, entry in enumerate(ents):
                txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
            txn.put(b'__len__', pickle.dumps(len(ents)))
        env.close()
        print(f'Saved {len(ents)} entries to {name} ({map_size/1024/1024:.0f} MB map)')

    summary = {
        'n_train': len(train_entries),
        'n_val': len(val_entries),
        'source': f'EBI AlphaFold DB v{args.af2_version}',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDone!')


if __name__ == '__main__':
    main()
