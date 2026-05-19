#!/usr/bin/env python
"""
Strategy 2 (revised): Build BFN confidence dataset from PDB structures
matched to AF2 predictions, starting from UniProt entries with PDB cross-references.

Approach:
1. Query UniProt for reviewed Swiss-Prot entries (L=50-250) WITH PDB structures
2. For each hit, resolve the best PDB structure (highest resolution, single chain)
3. Download PDB, map to AF2 prediction (already have UniProt accession)
4. Preprocess into BFN batch
"""
import sys, os, json, pickle, time, shutil
from pathlib import Path
import requests
import numpy as np
import torch

import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

UNIPROT_API = 'https://rest.uniprot.org/uniprotkb/search'
AFDB_BASE = 'https://alphafold.ebi.ac.uk/files'
RCSB_FILES = 'https://files.rcsb.org/download'


def fetch_uniprot_with_pdb(max_results=200, min_len=50, max_len=250):
    """Fetch UniProt Swiss-Prot entries that have PDB cross-references."""
    print(f'Querying UniProt for reviewed entries (L={min_len}-{max_len}) with PDB structures...')
    query = f'(length:[{min_len} TO {max_len}] AND reviewed:true)'
    results = []
    seen = set()

    url = f'{UNIPROT_API}?query={query}&size=500&format=tsv&fields=accession,length,xref_pdb'

    for _ in range(20):  # Max 20 pages (10000 entries)
        if len(results) >= max_results:
            break

        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f'  API error: {r.status_code}')
            break

        lines = r.text.strip().split('\n')
        for line in lines[1:]:
            if len(results) >= max_results:
                break
            parts = line.split('\t')
            if len(parts) < 3 or not parts[0]:
                continue
            acc = parts[0]
            if acc in seen:
                continue
            seen.add(acc)
            try:
                length = int(parts[1])
            except (ValueError, IndexError):
                continue
            pdb_field = parts[2] if len(parts) > 2 else ''
            if not pdb_field:
                continue
            # Parse PDB IDs (semicolon-separated, e.g. "1ABC A;2DEF B")
            pdb_ids_raw = [p.strip() for p in pdb_field.split(';') if p.strip()]
            pdb_ids = []
            for pid in pdb_ids_raw:
                pdb_id = pid.split(' ')[0].split('/')[0].upper()
                if len(pdb_id) == 4 and pdb_id.isalnum():
                    pdb_ids.append(pdb_id)
            if pdb_ids and min_len <= length <= max_len:
                results.append((acc, length, pdb_ids))

        # Follow Link header for next page
        link = r.headers.get('Link', '')
        next_url = None
        for part in link.split(','):
            if 'rel="next"' in part:
                next_url = part.split('>')[0].lstrip('<')
                break
        if not next_url:
            break
        url = next_url

    print(f'  Got {len(results)} entries with PDB structures')
    return results


def get_pdb_resolution(pdb_id):
    """Get experimental resolution for a PDB ID."""
    try:
        url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            res_info = data.get('rcsb_entry_info', {})
            resolution = res_info.get('resolution_combined')
            if isinstance(resolution, list):
                resolution = resolution[0] if resolution else None
            if resolution is None:
                dr = res_info.get('diffrn_resolution_high', {})
                if isinstance(dr, dict):
                    resolution = dr.get('value')
            polymer_count = res_info.get('polymer_entity_count_protein', 0)
            if isinstance(polymer_count, list):
                polymer_count = polymer_count[0] if polymer_count else 0
            instance_count = res_info.get('deposited_polymer_entity_instance_count', 0)
            if isinstance(instance_count, list):
                instance_count = instance_count[0] if instance_count else 0
            return float(resolution) if resolution else None, int(polymer_count), int(instance_count)
    except Exception:
        pass
    return None, 0, 0


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


def extract_plddt_from_mmcif(cif_text):
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('tmp', io.StringIO(cif_text))
    plddt = []
    for chain in structure[0]:
        for res in chain:
            if res.get_resname() not in AA3_TO_1:
                continue
            if 'CA' in res:
                plddt.append(res['CA'].get_bfactor() / 100.0)
    return plddt


def compute_ptm_from_pae(pae_matrix):
    pae = np.array(pae_matrix, dtype=np.float64)
    L = pae.shape[0]
    if L < 2:
        return 0.5
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    ptm_scores = []
    for i in range(L):
        f_ij = 1.0 / (1.0 + (pae[i, :] / d0) ** 2)
        ptm_scores.append(np.mean(f_ij))
    return float(max(ptm_scores))


def main():
    import argparse
    p = argparse.ArgumentParser(description='Build PDB+AF2 dataset via UniProt xrefs')
    p.add_argument('--max_proteins', type=int, default=100)
    p.add_argument('--min_len', type=int, default=50)
    p.add_argument('--max_len', type=int, default=250)
    p.add_argument('--max_resolution', type=float, default=2.5)
    p.add_argument('--start_from', type=int, default=0, help='Resume from this UniProt entry index')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset_pdb'))
    p.add_argument('--pdb_cache_dir', default=str(PROJECT_DIR / 'data' / 'pdb_cache'))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_cache = Path(args.pdb_cache_dir)
    pdb_cache.mkdir(parents=True, exist_ok=True)

    # 1. Get UniProt entries with PDB structures
    entries = fetch_uniprot_with_pdb(args.max_proteins, args.min_len, args.max_len)

    print(f'\n{"="*60}')
    print(f'  Processing up to {args.max_proteins} entries')
    print(f'{"="*60}\n')

    results = []
    n_processed = 0
    n_no_pdb = 0
    n_bad_res = 0
    n_no_af2 = 0
    n_preprocess_fail = 0

    for i, (uniprot, length, pdb_candidates) in enumerate(entries):
        if i < args.start_from:
            continue
        if n_processed >= args.max_proteins:
            break

        print(f'[{n_processed+1}/{args.max_proteins}] {uniprot}...', end=' ', flush=True)

        # 2. Find best PDB structure (highest resolution)
        best_pdb = None
        best_res = 999
        for pdb_id in pdb_candidates[:5]:
            resolution, polymer_count, instance_count = get_pdb_resolution(pdb_id)
            if resolution and polymer_count >= 1:
                if resolution < best_res:
                    best_res = resolution
                    best_pdb = pdb_id

        if not best_pdb:
            print(f'SKIP (no suitable PDB among {pdb_candidates[:3]})')
            n_no_pdb += 1
            continue

        if best_res > args.max_resolution:
            print(f'SKIP (resolution={best_res}A > {args.max_resolution}A)')
            n_bad_res += 1
            continue

        # 3. Download PDB with retry
        pdb_path = pdb_cache / f'{best_pdb}.pdb'
        if not pdb_path.exists():
            url = f'{RCSB_FILES}/{best_pdb}.pdb'
            for retry in range(3):
                try:
                    r = requests.get(url, timeout=30)
                    break
                except requests.exceptions.ConnectionError:
                    if retry < 2:
                        print(f'(retry {retry+2}/3)', end=' ', flush=True)
                        time.sleep(2)
                    else:
                        raise
            if r.status_code != 200:
                print(f'SKIP (PDB download failed: {r.status_code})')
                continue
            with open(pdb_path, 'w') as f:
                f.write(r.text)

        # Verify length
        chains = load_pdb_sequence(pdb_path)
        if not chains:
            print('SKIP (no chains)')
            continue
        seq_len = sum(len(s) for s in chains.values())
        if seq_len < args.min_len or seq_len > args.max_len:
            print(f'SKIP (length={seq_len})')
            continue

        # 4. Download AF2 prediction with retry
        af2_url = f'{AFDB_BASE}/AF-{uniprot}-F1-model_v6.cif'
        for retry in range(3):
            try:
                r = requests.get(af2_url, timeout=30)
                break
            except requests.exceptions.ConnectionError:
                if retry < 2:
                    time.sleep(2)
                else:
                    raise
        if r.status_code != 200:
            print('SKIP (no AF2 prediction)')
            n_no_af2 += 1
            continue
        cif_text = r.text

        # Extract pLDDT from AF2 mmCIF
        plddt_list = extract_plddt_from_mmcif(cif_text)
        if not plddt_list:
            print('SKIP (no pLDDT)')
            continue

        # 5. Preprocess PDB into BFN batch
        try:
            batch, seq, n_res = preprocess_pdb_for_bfn(str(pdb_path))
        except Exception as e:
            print(f'SKIP (preprocess: {e})')
            n_preprocess_fail += 1
            continue

        if batch is None:
            print('SKIP (batch is None)')
            n_preprocess_fail += 1
            continue

        # Align pLDDT
        if len(plddt_list) < n_res:
            plddt_list += [0.5] * (n_res - len(plddt_list))
        plddt_tensor = torch.tensor(plddt_list[:n_res], dtype=torch.float32)

        # 6. Download AF2 PAE
        pae_t = torch.zeros(n_res, n_res)
        iptm = 0.5
        pae_url = f'{AFDB_BASE}/AF-{uniprot}-F1-predicted_aligned_error_v6.json'
        try:
            r_pae = requests.get(pae_url, timeout=30)
            if r_pae.status_code == 200:
                pae_data = r_pae.json()
                pae_matrix = None
                if isinstance(pae_data, list) and len(pae_data) > 0:
                    pae_matrix = pae_data[0].get('predicted_aligned_error')
                elif isinstance(pae_data, dict):
                    pae_matrix = pae_data.get('predicted_aligned_error')

                if pae_matrix is not None:
                    pae_np = np.array(pae_matrix, dtype=np.float64)
                    if pae_np.shape[0] < n_res:
                        pad_w = n_res - pae_np.shape[0]
                        pae_np = np.pad(pae_np, ((0, pad_w), (0, pad_w)), constant_values=0)
                    pae_np = pae_np[:n_res, :n_res]
                    pae_t = torch.tensor(pae_np, dtype=torch.float32)
                    iptm = compute_ptm_from_pae(pae_np)
        except Exception:
            pass

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
            'pdb_id': best_pdb,
            'sequence': seq,
            'batch': batch_clean,
            'af2_plddt': plddt_tensor.cpu(),
            'af2_iptm': iptm_tensor.cpu(),
            'af2_pae_matrix': pae_t.cpu(),
        }
        results.append(entry)
        n_processed += 1
        print(f'OK (L={n_res}, PDB={best_pdb}, res={best_res}A, pLDDT={plddt_tensor.mean():.3f}, pTM={iptm:.3f})')

    print(f'\n{"="*60}')
    print(f'  Matched: {len(results)} proteins')
    print(f'  No suitable PDB: {n_no_pdb}  Bad res: {n_bad_res}  No AF2: {n_no_af2}  Preprocess fail: {n_preprocess_fail}')
    print(f'{"="*60}')

    if not results:
        print('No entries!')
        sys.exit(1)

    # Split
    np.random.seed(2026)
    indices = np.random.permutation(len(results))
    split_idx = int(len(results) * 0.8)
    train_entries = [results[i] for i in indices[:split_idx]]
    val_entries = [results[i] for i in indices[split_idx:]]

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
        'source': 'RCSB PDB + EBI AlphaFold DB v6 (via UniProt xrefs)',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDone!')


if __name__ == '__main__':
    main()
