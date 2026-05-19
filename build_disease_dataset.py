#!/usr/bin/env python
"""
Build disease-associated human protein dataset for BFN confidence training.

Searches UniProt for reviewed human proteins linked to organ-specific diseases,
downloads AF2 predictions from EBI AFDB v6, and builds LMDB entries.
"""
import sys, os, json, pickle, time, io, tempfile, re
from pathlib import Path

if sys.platform == 'win32':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from antibodydesignbfn.datasets.protein import preprocess_protein_structure
from antibodydesignbfn.utils.train import recursive_to
from antibodydesignbfn.utils.transforms import get_transform

AFDB_BASE = 'https://alphafold.ebi.ac.uk/files'
UNIPROT_API = 'https://rest.uniprot.org/uniprotkb/search'

AA3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

# Organ-specific disease UniProt queries
DISEASE_QUERIES = [
    # Liver: hepatitis, cirrhosis, HCC, NAFLD, Wilson disease
    ('liver_disease', '(hepatitis OR cirrhosis OR hepatocellular OR NAFLD OR "Wilson disease" OR "alpha-1 antitrypsin") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Kidney: CKD, nephritis, PKD, nephrotic syndrome
    ('kidney_disease', '(nephritis OR nephrotic OR "polycystic kidney" OR "renal failure" OR nephropathy OR "Alport syndrome") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Heart/cardiovascular: cardiomyopathy, arrhythmia, CHD
    ('heart_disease', '(cardiomyopathy OR arrhythmia OR "long QT" OR "Brugada syndrome" OR "Marfan syndrome" OR "Noonan syndrome") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Lung: COPD, IPF, cystic fibrosis, asthma
    ('lung_disease', '("cystic fibrosis" OR "pulmonary fibrosis" OR "COPD" OR emphysema OR "surfactant" OR "alpha-1 antitrypsin") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Neurodegenerative: Alzheimer's, Parkinson's, ALS, Huntington's, prion
    ('neuro_disease', '("Alzheimer" OR "Parkinson" OR "amyotrophic lateral sclerosis" OR "Huntington" OR "prion" OR "frontotemporal dementia" OR "spinocerebellar ataxia" OR "Charcot-Marie-Tooth") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Cancer: oncogenes, tumor suppressors
    ('cancer', '("tumor suppressor" OR oncogene OR "tyrosine kinase" OR "transcription factor" AND cancer) AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Metabolic: diabetes, obesity, gout
    ('metabolic_disease', '("diabetes" OR "obesity" OR "gout" OR "hypercholesterolemia" OR "lysosomal storage" OR "Gaucher" OR "Pompe" OR "Fabry") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Blood/immune: hemophilia, thalassemia, SCID
    ('blood_disease', '("hemophilia" OR "thalassemia" OR "sickle cell" OR "severe combined immunodeficiency" OR "chronic granulomatous" OR "leukemia" OR "lymphoma") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Musculoskeletal: muscular dystrophy, osteogenesis imperfecta
    ('muscle_disease', '("muscular dystrophy" OR "osteogenesis imperfecta" OR "myopathy" OR "rhabdomyosarcoma" OR "Ehlers-Danlos") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
    # Eye: retinitis pigmentosa, macular degeneration
    ('eye_disease', '("retinitis pigmentosa" OR "macular degeneration" OR "cataract" OR "glaucoma" OR "Leber congenital amaurosis") AND (organism_id:9606) AND (length:[50 TO 250]) AND (reviewed:true) AND (fragment:false)'),
]


def resilient_get(url, timeout=60, max_retries=3):
    """HTTP GET with retry and backoff."""
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    session = requests.Session()
    retry = Retry(total=max_retries, backoff_factor=2,
                  status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session.get(url, timeout=timeout)


def compute_ptm_from_pae(pae_matrix):
    pae = np.array(pae_matrix, dtype=np.float64)
    L = pae.shape[0]
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    ptm_scores = []
    for i in range(L):
        f_ij = 1.0 / (1.0 + (pae[i, :] / d0) ** 2)
        ptm_scores.append(np.mean(f_ij))
    return float(max(ptm_scores))


def fetch_uniprot_accessions(query, max_per_query=40):
    """Fetch UniProt accessions for a disease query."""
    url = f'{UNIPROT_API}?query={requests.utils.quote(query)}&size={max_per_query}&format=tsv&fields=accession,length,protein_name'
    r = requests.get(url, timeout=30)
    lines = r.text.strip().split('\n')
    accessions = []
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) >= 2 and parts[0]:
            accessions.append((parts[0], parts[2] if len(parts) > 2 else ''))
    return accessions


def mmcif_to_temp_pdb(cif_text, accession):
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


def extract_plddt_from_mmcif(cif_text):
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


def main():
    import argparse, shutil
    p = argparse.ArgumentParser(description='Build disease protein dataset')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset_disease'),
                   help='Output LMDB directory')
    p.add_argument('--max_per_query', type=int, default=40, help='Max proteins per disease category')
    p.add_argument('--af2_version', type=int, default=6, help='AFDB version')
    p.add_argument('--resume', action='store_true', help='Resume from existing LMDB')
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load already-processed entries
    processed_ids = set()
    entries = []
    if args.resume:
        import lmdb as _lmdb
        db_path = str(output_dir / 'confidence_disease.lmdb')
        if os.path.exists(db_path):
            env = _lmdb.open(db_path, readonly=True)
            with env.begin() as txn:
                n = pickle.loads(txn.get(b'__len__'))
                for idx in range(n):
                    entry = pickle.loads(txn.get(f'{idx:08d}'.encode()))
                    entries.append(entry)
                    processed_ids.add(entry['pdb_id'])
            env.close()
            print(f'Resumed: {len(entries)} existing entries')

    all_accessions = []
    seen = set()

    for category, query in DISEASE_QUERIES:
        print(f'\n{"="*60}')
        print(f'  {category}: searching UniProt...')
        print(f'{"="*60}')
        results = fetch_uniprot_accessions(query, max_per_query=args.max_per_query)
        new_results = [(acc, name) for acc, name in results if acc not in seen]
        for acc, name in new_results:
            seen.add(acc)
            all_accessions.append((acc, name, category))
        print(f'  Found {len(results)} results, {len(new_results)} new (total unique: {len(seen)})')

    # Filter out already-processed
    all_accessions = [(acc, name, cat) for acc, name, cat in all_accessions if acc not in processed_ids]
    print(f'\n{"="*60}')
    print(f'  Total unique accessions: {len(all_accessions)} (skipping {len(processed_ids)} already processed)')
    print(f'{"="*60}\n')

    if not all_accessions:
        print('All accessions already processed!')
        sys.exit(0)

    n_failed = 0
    n_pae_miss = 0

    for i, (acc, name, category) in enumerate(all_accessions):
        print(f'[{i+1}/{len(all_accessions)}] [{category}] {acc}: {name[:60]}...', end=' ', flush=True)

        # Download mmCIF
        url = f'{AFDB_BASE}/AF-{acc}-F1-model_v{args.af2_version}.cif'
        r = resilient_get(url, timeout=60)
        if r.status_code != 200:
            print('SKIP (no AF2 prediction)')
            n_failed += 1
            continue
        cif_text = r.text

        # Extract pLDDT and sequence from mmCIF
        plddt_list, mmcif_seq = extract_plddt_from_mmcif(cif_text)
        if not plddt_list:
            print('SKIP (no pLDDT)')
            n_failed += 1
            continue

        # Preprocess
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

        # Align pLDDT
        if len(plddt_list) < n_res:
            plddt_list += [0.5] * (n_res - len(plddt_list))
        plddt_tensor = torch.tensor(plddt_list[:n_res], dtype=torch.float32)

        # Download PAE
        pae_matrix = None
        pae_url = f'{AFDB_BASE}/AF-{acc}-F1-predicted_aligned_error_v{args.af2_version}.json'
        try:
            r_pae = resilient_get(pae_url, timeout=30)
            if r_pae.status_code == 200:
                pae_data = r_pae.json()
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

        iptm = compute_ptm_from_pae(pae_t[:n_res, :n_res].cpu().numpy()) if pae_t.shape[0] > 0 else 0.5
        iptm_tensor = torch.tensor(iptm, dtype=torch.float32)

        # Clean batch
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

        # Incremental save every 30 proteins
        if len(entries) % 30 == 0:
            import shutil as _shutil
            import lmdb as _lmdb
            db_path_tmp = str(output_dir / 'confidence_disease.lmdb')
            if os.path.exists(db_path_tmp):
                _shutil.rmtree(db_path_tmp)
            sample = pickle.dumps(entries[0])
            map_size = len(sample) * len(entries) * 3 + 10 * 1024 * 1024
            env_tmp = _lmdb.open(db_path_tmp, map_size=map_size)
            with env_tmp.begin(write=True) as txn:
                for j, e in enumerate(entries):
                    txn.put(f'{j:08d}'.encode(), pickle.dumps(e))
                txn.put(b'__len__', pickle.dumps(len(entries)))
            env_tmp.close()
            print(f'  [saved {len(entries)} entries so far]')

    print(f'\n{"="*60}')
    print(f'  Results: {len(entries)}/{len(all_accessions)} succeeded')
    print(f'  Failed: {n_failed},  Missing PAE: {n_pae_miss}')
    print(f'{"="*60}')

    if not entries:
        print('No entries!')
        sys.exit(1)

    # Save as single dataset (will be merged with main training set)
    import lmdb
    db_path = str(output_dir / 'confidence_disease.lmdb')
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    sample = pickle.dumps(entries[0])
    map_size = len(sample) * len(entries) * 3 + 10 * 1024 * 1024
    env = lmdb.open(db_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()

    summary = {
        'n_entries': len(entries),
        'n_failed': n_failed,
        'n_pae_miss': n_pae_miss,
        'categories': {cat: sum(1 for _, _, c in all_accessions if c == cat) for cat in set(c for _, _, c in all_accessions)},
        'source': f'EBI AlphaFold DB v{args.af2_version}',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved {len(entries)} disease-protein entries to {db_path}')
    print(f'Categories: {summary["categories"]}')
    print('Done!')


if __name__ == '__main__':
    main()
