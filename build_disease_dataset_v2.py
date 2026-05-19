#!/usr/bin/env python
"""Build expanded disease-associated protein dataset v2 for BFN confidence training.

Improvements over v1:
  - 16 disease categories (up from 10): added autoimmune, skin, endocrine,
    GI, mitochondrial, neurodevelopmental
  - Wider length range: 50-400 aa (up from 50-250) to capture kinases, receptors
  - UniProt pagination: fetches up to 200 entries per query (up from 40)
  - Keyword-based queries: "Disease mutation" (KW-0225), "Oncogene", "Tumor suppressor"
  - Tags entries with disease category and source for traceability
  - Deduplication across all queries

Usage:
  python build_disease_dataset_v2.py --output ./data/confidence_dataset_disease_v2
  python build_disease_dataset_v2.py --output ./data/confidence_dataset_disease_v2 --resume
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


# ── Expanded disease queries ──
# Phase A: Existing 10 categories with wider length range + larger page size
DISEASE_QUERIES_EXPANDED = [
    ('liver_disease', '(hepatitis OR cirrhosis OR hepatocellular OR NAFLD OR "Wilson disease" OR "alpha-1 antitrypsin" OR "cholestasis" OR "hemochromatosis") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('kidney_disease', '(nephritis OR nephrotic OR "polycystic kidney" OR "renal failure" OR nephropathy OR "Alport syndrome" OR "Bartter syndrome" OR "Gitelman syndrome") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('heart_disease', '(cardiomyopathy OR arrhythmia OR "long QT" OR "Brugada syndrome" OR "Marfan syndrome" OR "Noonan syndrome" OR "hypertrophic cardiomyopathy" OR "dilated cardiomyopathy") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('lung_disease', '("cystic fibrosis" OR "pulmonary fibrosis" OR "COPD" OR emphysema OR "surfactant" OR "alpha-1 antitrypsin" OR "primary ciliary dyskinesia") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_disease', '("Alzheimer" OR "Parkinson" OR "amyotrophic lateral sclerosis" OR "Huntington" OR "prion" OR "frontotemporal dementia" OR "spinocerebellar ataxia" OR "Charcot-Marie-Tooth" OR "epilepsy" OR "ataxia") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('cancer', '("tumor suppressor" OR oncogene OR "tyrosine kinase" OR "DNA repair" AND cancer) AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('metabolic_disease', '("diabetes" OR "obesity" OR "gout" OR "hypercholesterolemia" OR "lysosomal storage" OR "Gaucher" OR "Pompe" OR "Fabry" OR "phenylketonuria" OR "galactosemia") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('blood_disease', '("hemophilia" OR "thalassemia" OR "sickle cell" OR "severe combined immunodeficiency" OR "chronic granulomatous" OR "leukemia" OR "lymphoma" OR "anemia" OR "myelodysplastic") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('muscle_disease', '("muscular dystrophy" OR "osteogenesis imperfecta" OR "myopathy" OR "rhabdomyosarcoma" OR "Ehlers-Danlos" OR "Marfan syndrome" OR "dystrophin") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('eye_disease', '("retinitis pigmentosa" OR "macular degeneration" OR "cataract" OR "glaucoma" OR "Leber congenital amaurosis" OR "retinoblastoma" OR "Usher syndrome") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
]

# Phase B: 6 new disease categories
NEW_DISEASE_QUERIES = [
    ('autoimmune_disease', '("rheumatoid arthritis" OR "lupus" OR "multiple sclerosis" OR "type 1 diabetes" OR "inflammatory bowel" OR "Crohn" OR "ulcerative colitis" OR "psoriasis" OR "ankylosing spondylitis") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('skin_disease', '("epidermolysis bullosa" OR "ichthyosis" OR "albinism" OR "xeroderma pigmentosum" OR "vitiligo" OR "dermatitis" OR "pemphigus") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('endocrine_disease', '("thyroid hormone" OR "adrenal insufficiency" OR "pituitary" OR "parathyroid" OR "congenital adrenal hyperplasia" OR "Graves" OR "Hashimoto") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('gi_disease', '("celiac disease" OR "pancreatitis" OR "peptic ulcer" OR "gastrinoma" OR "Hirschsprung" OR "pancreatic insufficiency" OR "primary sclerosing cholangitis") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('mitochondrial_disease', '("MELAS" OR "Leigh syndrome" OR "mitochondrial" OR "oxidative phosphorylation" OR "respiratory chain" OR "LHON" OR "Kearns-Sayre" OR "MERRF") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neurodevelopmental_disease', '("autism" OR "Rett syndrome" OR "Fragile X" OR "Angelman" OR "intellectual disability" OR "microcephaly" OR "lissencephaly" OR "holoprosencephaly") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
]

# Phase C: Keyword-based queries for high-value disease proteins
KEYWORD_QUERIES = [
    ('disease_mutation', '(keyword:"Disease mutation" OR annotation:(type:"natural variant" disease)) AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('oncogene', '(keyword:Oncogene OR keyword:"Tumor suppressor") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('cancer_driver', '(keyword:"Proto-oncogene" OR keyword:"Tumor suppressor" OR cc_function:"tumor suppressor") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
]

ALL_QUERIES = DISEASE_QUERIES_EXPANDED + NEW_DISEASE_QUERIES + KEYWORD_QUERIES


def resilient_get(url, timeout=60, max_retries=3):
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


def fetch_uniprot_accessions(query, max_per_query=200):
    """Fetch UniProt accessions with pagination support."""
    all_accessions = []
    url_base = f'{UNIPROT_API}?query={requests.utils.quote(query)}&size=500&format=tsv&fields=accession,length,protein_name'
    seen = set()

    # UniProt pagination: use cursor-based pagination via next link header
    next_url = url_base
    page = 0
    max_pages = 4  # Max 4 pages × 500 = 2000 per query

    while next_url and page < max_pages and len(all_accessions) < max_per_query:
        r = requests.get(next_url, timeout=30)
        if r.status_code != 200:
            print(f'    HTTP {r.status_code}, stopping pagination')
            break

        lines = r.text.strip().split('\n')
        for line in lines[1:]:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[0]:
                acc = parts[0]
                if acc not in seen:
                    seen.add(acc)
                    all_accessions.append((acc, parts[2] if len(parts) > 2 else ''))

        # Check for next page link
        next_url = None
        link_header = r.headers.get('Link', '')
        for link in link_header.split(','):
            if 'rel="next"' in link:
                next_url = link.split(';')[0].strip(' <>')
                break
        page += 1

    return all_accessions[:max_per_query]


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
    p = argparse.ArgumentParser(description='Build expanded disease protein dataset v2')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset_disease_v2'),
                   help='Output LMDB directory')
    p.add_argument('--max_per_query', type=int, default=50,
                   help='Max proteins per disease category')
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

    # Also check for v1 disease entries to avoid overlap
    v1_path = PROJECT_DIR / 'data' / 'confidence_dataset_disease' / 'confidence_disease.lmdb'
    if v1_path.exists():
        import lmdb as _lmdb
        env = _lmdb.open(str(v1_path), readonly=True, lock=False)
        with env.begin() as txn:
            n = pickle.loads(txn.get(b'__len__'))
            for idx in range(n):
                entry = pickle.loads(txn.get(f'{idx:08d}'.encode()))
                processed_ids.add(entry['pdb_id'])
        env.close()
        print(f'Added {n} v1 accessions to skip list')

    all_accessions = []
    seen = set()

    for category, query in ALL_QUERIES:
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
    print(f'  Total unique accessions to process: {len(all_accessions)}')
    print(f'  Skipping {len(processed_ids)} already processed')
    print(f'{"="*60}\n')

    if not all_accessions:
        print('All accessions already processed!')
        sys.exit(0)

    n_failed = 0
    n_pae_miss = 0
    n_skipped_length = 0

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

        # Length check
        if len(plddt_list) < 30 or len(plddt_list) > 500:
            print(f'SKIP (length {len(plddt_list)} out of range)')
            n_skipped_length += 1
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
            'source': f'Disease_v2_{category}',
            'is_idp': False,
        }
        entries.append(entry)
        print(f'OK (L={n_res}, pLDDT={plddt_tensor.mean():.3f}, pTM={iptm:.3f})')

        # Incremental save every 50 proteins
        if len(entries) % 50 == 0:
            import shutil as _shutil
            import lmdb as _lmdb
            db_path_tmp = str(output_dir / 'confidence_disease.lmdb')
            if os.path.exists(db_path_tmp):
                _shutil.rmtree(db_path_tmp)
            sample = pickle.dumps(entries[0])
            map_size = max(len(sample) * len(entries) * 3 + 10 * 1024 * 1024, 500 * 1024 * 1024)
            map_size = ((map_size + 4095) // 4096) * 4096
            env_tmp = _lmdb.open(db_path_tmp, map_size=map_size)
            with env_tmp.begin(write=True) as txn:
                for j, e in enumerate(entries):
                    txn.put(f'{j:08d}'.encode(), pickle.dumps(e))
                txn.put(b'__len__', pickle.dumps(len(entries)))
            env_tmp.close()
            print(f'  [saved {len(entries)} entries so far]')

    print(f'\n{"="*60}')
    print(f'  Results: {len(entries)}/{len(all_accessions)} succeeded')
    print(f'  Failed: {n_failed},  Skipped (length): {n_skipped_length},  Missing PAE: {n_pae_miss}')
    print(f'{"="*60}')

    if not entries:
        print('No entries! Check network/API access.')
        sys.exit(1)

    # Final save
    import lmdb
    db_path = str(output_dir / 'confidence_disease.lmdb')
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    sample = pickle.dumps(entries[0])
    map_size = max(len(sample) * len(entries) * 3 + 10 * 1024 * 1024, 500 * 1024 * 1024)
    map_size = ((map_size + 4095) // 4096) * 4096
    print(f'Final LMDB map_size: {map_size / 1024 / 1024:.0f} MB for {len(entries)} entries')
    env = lmdb.open(db_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()

    # Category breakdown
    cat_counts = {}
    for e in entries:
        cat = e.get('source', 'unknown')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    summary = {
        'n_entries': len(entries),
        'n_failed': n_failed,
        'n_pae_miss': n_pae_miss,
        'n_skipped_length': n_skipped_length,
        'categories': cat_counts,
        'source': f'EBI AlphaFold DB v{args.af2_version}',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved {len(entries)} disease-protein entries to {db_path}')
    print(f'Categories: {json.dumps(cat_counts, indent=2)}')
    print('Done!')


if __name__ == '__main__':
    main()
