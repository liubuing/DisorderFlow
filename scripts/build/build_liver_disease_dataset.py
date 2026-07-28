#!/usr/bin/env python
"""
Build liver/hepatic disease protein dataset for BFN confidence training.

Covers 25+ liver disease categories: metabolic, genetic, infectious,
autoimmune, and oncologic liver conditions.

Modeled on build_brain_disease_dataset.py — queries UniProt, downloads
AlphaFold DB structures, preprocesses into BFN training format.
"""
import sys, os, json, pickle, time, io, tempfile, re
from pathlib import Path

if sys.platform == 'win32':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import requests
import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from build_utils import AA3_TO_1, AA_LETTERS, compute_ptm_from_pae, resilient_get, load_pdb_sequence, save_lmdb

AFDB_BASE = 'https://alphafold.ebi.ac.uk/files'
UNIPROT_API = 'https://rest.uniprot.org/uniprotkb/search'

# 10 broad liver disease categories (consolidated for API efficiency)
LIVER_DISEASE_QUERIES = [
    # 1. Protein misfolding / aggregation (AAT deficiency, amyloidosis)
    ('liver_misfolding_aggregation', '("alpha-1 antitrypsin" OR "SERPINA1" OR "transthyretin amyloidosis" OR "TTR" OR "APOA1 amyloidosis" OR "fibrinogen amyloidosis" OR "FGA" OR "lysozyme amyloidosis" OR "LYZ" OR "gelsolin amyloidosis" OR "GSN") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true) AND (fragment:false)'),

    # 2. Metal metabolism (Wilson disease, hemochromatosis)
    ('liver_metal_metabolism', '("Wilson disease" OR "ATP7B" OR "ceruloplasmin" OR "CP" OR "hemochromatosis" OR "HFE" OR "HAMP" OR "hepcidin" OR "TFR2" OR "HJV" OR "ferroportin" OR "SLC40A1") AND (organism_id:9606) AND (length:[50 TO 1500]) AND (reviewed:true) AND (fragment:false)'),

    # 3. Metabolic fatty liver / NAFLD-NASH / cirrhosis / fibrosis
    ('liver_metabolic_fibrosis', '("NAFLD" OR "NASH" OR "fatty liver" OR "PNPLA3" OR "TM6SF2" OR "GCKR" OR "MBOAT7" OR "HSD17B13" OR "cirrhosis" OR "liver fibrosis" OR "TGFB1" OR "PDGFRB" OR "COL1A1" OR "LOXL2" OR "TIMP1" OR "CTGF" OR "hepatic stellate") AND (organism_id:9606) AND (length:[50 TO 600]) AND (reviewed:true) AND (fragment:false)'),

    # 4. Glycogen storage + congenital glycosylation
    ('liver_glycogen_glycosylation', '("glycogen storage disease" OR "G6PC" OR "SLC37A4" OR "AGL" OR "GBE1" OR "PYGL" OR "PHKA2" OR "PHKB" OR "von Gierke" OR "McArdle" OR "congenital disorder of glycosylation" OR "PMM2" OR "MPI" OR "ALG6" OR "DPM1") AND (organism_id:9606) AND (length:[50 TO 900]) AND (reviewed:true) AND (fragment:false)'),

    # 5. Cholestasis / bile (PFIC, Alagille, Dubin-Johnson, Crigler-Najjar, Gilbert)
    ('liver_cholestasis_bile', '("progressive familial intrahepatic cholestasis" OR "PFIC" OR "ATP8B1" OR "ABCB11" OR "ABCB4" OR "BSEP" OR "Alagille syndrome" OR "JAG1" OR "Crigler-Najjar" OR "Gilbert syndrome" OR "UGT1A1" OR "Dubin-Johnson" OR "ABCC2" OR "NR1H4" OR "TJP2") AND (organism_id:9606) AND (length:[50 TO 1500]) AND (reviewed:true) AND (fragment:false)'),

    # 6. Autoimmune liver (AIH, PBC, PSC)
    ('liver_autoimmune', '("autoimmune hepatitis" OR "primary biliary cholangitis" OR "PBC" OR "primary sclerosing cholangitis" OR "PSC" OR "DLAT" OR "SP100" OR "GP210" OR "NUP210" OR "IgG4-related") AND (organism_id:9606) AND (length:[50 TO 2000]) AND (reviewed:true) AND (fragment:false)'),

    # 7. Viral hepatitis (B, C, D, E) + host factors
    ('liver_viral_hepatitis', '("hepatitis B" OR "HBV" OR "hepatitis C" OR "HCV" OR "CD81" OR "SCARB1" OR "CLDN1" OR "OCLN" OR "IFNL3" OR "IFNL4" OR "hepatitis E" OR "HEV" OR "hepatitis D" OR "HDV") AND (organism_id:9606) AND (length:[50 TO 700]) AND (reviewed:true) AND (fragment:false)'),

    # 8. Liver tumors (HCC, cholangiocarcinoma)
    ('liver_tumors', '("hepatocellular carcinoma" OR "HCC" OR "hepatoma" OR "liver cancer" OR "AFP" OR "GPC3" OR "CTNNB1" OR "ARID1A" OR "cholangiocarcinoma" OR "bile duct cancer" OR "FGFR2" OR "IDH1" OR "BAP1") AND (organism_id:9606) AND (length:[50 TO 2500]) AND (reviewed:true) AND (fragment:false)'),

    # 9. Porphyria + drug metabolism / DILI
    ('liver_porphyria_dili', '("acute intermittent porphyria" OR "HMBS" OR "UROD" OR "CPOX" OR "PPOX" OR "FECH" OR "drug-induced liver injury" OR "DILI" OR "CYP2E1" OR "CYP3A4" OR "NAT2" OR "GSTT1" OR "GSTM1" OR "NQO1") AND (organism_id:9606) AND (length:[50 TO 600]) AND (reviewed:true) AND (fragment:false)'),

    # 10. Inborn errors: urea cycle, organic acidemia, amino acid, fructose, galactose, tyrosinemia, citrin, maple syrup, etc.
    ('liver_inborn_errors', '("urea cycle disorder" OR "OTC" OR "ornithine transcarbamylase" OR "CPS1" OR "ASS1" OR "ASL" OR "ARG1" OR "tyrosinemia" OR "FAH" OR "HPD" OR "hereditary fructose intolerance" OR "ALDOB" OR "galactosemia" OR "GALT" OR "citrin deficiency" OR "SLC25A13" OR "maple syrup urine disease" OR "BCKDHA" OR "BCKDHB" OR "propionic acidemia" OR "PCCA" OR "PCCB" OR "methylmalonic acidemia" OR "MUT" OR "isovaleric acidemia" OR "IVD" OR "glutaryl-CoA" OR "GCDH" OR "alkaptonuria" OR "HGD" OR "biotinidase" OR "BTD") AND (organism_id:9606) AND (length:[50 TO 750]) AND (reviewed:true) AND (fragment:false)'),
]


def fetch_uniprot_accessions(query, max_per_query=80):
    url = f'{UNIPROT_API}?query={requests.utils.quote(query)}&size={max_per_query}&format=tsv&fields=accession,length,protein_name'
    r = requests.get(url, timeout=20)
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


def process_one_entry(acc, name, category, af2_version):
    """Process a single protein entry. Returns (entry_dict, None) on success, (None, 'reason') on failure."""
    # Download mmCIF
    url = f'{AFDB_BASE}/AF-{acc}-F1-model_v{af2_version}.cif'
    r = resilient_get(url, timeout=60)
    if r.status_code != 200:
        return None, f'SKIP (no AF2, status={r.status_code})'

    cif_text = r.text
    plddt_list, mmcif_seq = extract_plddt_from_mmcif(cif_text)
    if not plddt_list:
        return None, 'SKIP (no pLDDT)'

    tmp_pdb = None
    try:
        tmp_pdb = mmcif_to_temp_pdb(cif_text, acc)
        batch, seq, n_res = preprocess_pdb_for_bfn(tmp_pdb)
    except Exception as e:
        if tmp_pdb and os.path.exists(tmp_pdb):
            os.unlink(tmp_pdb)
        return None, f'SKIP (preprocess: {e})'

    if tmp_pdb and os.path.exists(tmp_pdb):
        os.unlink(tmp_pdb)

    if batch is None:
        return None, 'SKIP (batch is None)'

    if len(plddt_list) < n_res:
        plddt_list += [0.5] * (n_res - len(plddt_list))
    plddt_tensor = torch.tensor(plddt_list[:n_res], dtype=torch.float32)

    # Download PAE
    pae_matrix = None
    pae_url = f'{AFDB_BASE}/AF-{acc}-F1-predicted_aligned_error_v{af2_version}.json'
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

    pae_miss = pae_matrix is None
    if pae_miss:
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
        'category': category,
    }
    status = f'OK (L={n_res}, pLDDT={plddt_tensor.mean():.3f}, pTM={iptm:.3f})'
    return (entry, pae_miss), status


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    p = argparse.ArgumentParser(description='Build liver disease protein dataset')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset_liver_disease'),
                   help='Output LMDB directory')
    p.add_argument('--max_per_query', type=int, default=60, help='Max proteins per disease category')
    p.add_argument('--af2_version', type=int, default=6, help='AFDB version')
    p.add_argument('--resume', action='store_true', help='Resume from existing LMDB')
    p.add_argument('--workers', type=int, default=8, help='Concurrent download workers')
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    processed_ids = set()
    entries = []
    if args.resume:
        import lmdb as _lmdb
        db_path = str(output_dir / 'liver_disease.lmdb')
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

    for i, (category, query) in enumerate(LIVER_DISEASE_QUERIES):
        print(f'\n{"="*60}')
        print(f'  {category}: searching UniProt...')
        print(f'   Query: {query[:120]}...')
        print(f'{"="*60}')
        if i > 0:
            time.sleep(1.0)  # Rate-limit UniProt API
        try:
            results = fetch_uniprot_accessions(query, max_per_query=args.max_per_query)
        except Exception as e:
            print(f'  Search failed: {e}')
            continue
        new_results = [(acc, name) for acc, name in results if acc not in seen]
        for acc, name in new_results:
            seen.add(acc)
            all_accessions.append((acc, name, category))
        print(f'  Found {len(results)} results, {len(new_results)} new (total unique: {len(seen)})')

    all_accessions = [(acc, name, cat) for acc, name, cat in all_accessions if acc not in processed_ids]
    print(f'\n{"="*60}')
    print(f'  Total unique accessions to process: {len(all_accessions)}')
    print(f'  Already processed (skipping): {len(processed_ids)}')
    print(f'  Workers: {args.workers}')
    print(f'{"="*60}\n')

    if not all_accessions:
        print('All accessions already processed!')
        sys.exit(0)

    n_failed = 0
    n_pae_miss = 0
    n_completed = 0
    n_total = len(all_accessions)
    lock = threading.Lock()
    save_interval = 50

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, (acc, name, category) in enumerate(all_accessions):
            fut = executor.submit(process_one_entry, acc, name, category, args.af2_version)
            futures[fut] = (i + 1, acc, name, category)

        for fut in as_completed(futures):
            idx, acc, name, category = futures[fut]
            try:
                result, status = fut.result()
            except Exception as e:
                with lock:
                    n_failed += 1
                    n_completed += 1
                print(f'[{n_completed}/{n_total}] [{category}] {acc}: FAIL ({e})')
                continue

            with lock:
                n_completed += 1
                if result is None:
                    n_failed += 1
                    print(f'[{n_completed}/{n_total}] [{category}] {acc}: {status}')
                else:
                    entry, pae_miss = result
                    if pae_miss:
                        n_pae_miss += 1
                    entries.append(entry)
                    print(f'[{n_completed}/{n_total}] [{category}] {acc}: {status}')

                if len(entries) % save_interval == 0 and entries:
                    save_lmdb(str(output_dir / 'liver_disease.lmdb'), entries)
                    print(f'  [saved {len(entries)} entries at {n_completed}/{n_total}]')

    print(f'\n{"="*60}')
    print(f'  Results: {len(entries)}/{len(all_accessions)} succeeded')
    print(f'  Failed: {n_failed},  Missing PAE: {n_pae_miss}')
    print(f'{"="*60}')

    if not entries:
        print('No entries!')
        sys.exit(1)

    save_lmdb(str(output_dir / 'liver_disease.lmdb'), entries)

    cat_counts = {}
    for e in entries:
        cat = e.get('category', 'unknown')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    summary = {
        'n_entries': len(entries),
        'n_failed': n_failed,
        'n_pae_miss': n_pae_miss,
        'categories': cat_counts,
        'source': f'EBI AlphaFold DB v{args.af2_version}',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'Liver/hepatic disease focused dataset for BFN confidence training (25+ categories)',
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved {len(entries)} liver-disease protein entries to {output_dir / "liver_disease.lmdb"}')
    print(f'Categories: {json.dumps(cat_counts, indent=2)}')
    print('Done!')


if __name__ == '__main__':
    main()
