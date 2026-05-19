#!/usr/bin/env python
"""
Build brain/neurological disease protein dataset for BFN confidence training.

Expanded brain disease categories with higher coverage of neurodegenerative,
neurodevelopmental, brain tumor, epilepsy, and psychiatric disorder proteins.
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

# Expanded brain/neurological disease queries
BRAIN_DISEASE_QUERIES = [
    # === Neurodegenerative (expanded) ===
    ('neuro_alzheimer', '("Alzheimer disease" OR "amyloid" OR "APP" OR "PSEN1" OR "PSEN2" OR "APOE" OR "tauopathy") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_parkinson', '("Parkinson disease" OR "SNCA" OR "LRRK2" OR "PARK7" OR "PINK1" OR "PRKN" OR "alpha-synuclein" OR "parkin") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_als_ftd', '("amyotrophic lateral sclerosis" OR "frontotemporal dementia" OR "TARDBP" OR "TDP-43" OR "FUS" OR "C9orf72" OR "SOD1" OR "UBQLN2" OR "TBK1") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_huntington', '("Huntington disease" OR "HTT" OR "huntingtin" OR "polyglutamine" OR "CAG repeat" OR "spinocerebellar ataxia" OR "ATXN1" OR "ATXN2" OR "ATXN3" OR "ATXN7") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_prion', '("prion disease" OR "Creutzfeldt-Jakob" OR "PRNP" OR "fatal familial insomnia" OR "Gerstmann-Straussler" OR "PRND" OR "transmissible spongiform") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_msa_dlb', '("multiple system atrophy" OR "Lewy body dementia" OR "dementia with Lewy" OR "DLB" OR "alpha-synuclein" OR "COQ2" OR "progressive supranuclear palsy" OR "MAPT") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),

    # === Brain tumors ===
    ('brain_glioma', '("glioma" OR "glioblastoma" OR "astrocytoma" OR "oligodendroglioma" OR "IDH1" OR "IDH2" OR "EGFR" OR "TP53" OR "PTEN" OR "MGMT" OR "ATRX" OR "CIC" OR "FUBP1") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('brain_meningioma', '("meningioma" OR "NF2" OR "merlin" OR "schwannomin" OR "SMARCB1" OR "SMARCE1" OR "AKT1" OR "SMO" OR "TRAF7") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('brain_medulloblastoma', '("medulloblastoma" OR "SHH" OR "PTCH1" OR "SUFU" OR "WNT" OR "CTNNB1" OR "APC" OR "MYC" OR "MYCN" OR "OTX2") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('brain_neurofibromatosis', '("neurofibromatosis" OR "NF1" OR "neurofibromin" OR "SPRED1" OR "Legius syndrome") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),

    # === Neurodevelopmental ===
    ('neuro_autism', '("autism spectrum disorder" OR "SHANK3" OR "NLGN3" OR "NLGN4X" OR "NRXN1" OR "CNTNAP2" OR "SYNGAP1" OR "CHD8" OR "SCN2A" OR "ADNP" OR "ANK2" OR "PTEN") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_rett', '("Rett syndrome" OR "MECP2" OR "CDKL5" OR "FOXG1" OR "MEF2C") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_fragile_x', '("fragile X" OR "FMR1" OR "FMR2" OR "FXR1" OR "FXR2") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_angelman', '("Angelman syndrome" OR "UBE3A" OR "GABRB3" OR "Prader-Willi" OR "SNRPN" OR "NDN") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_tuberous', '("tuberous sclerosis" OR "TSC1" OR "TSC2" OR "hamartin" OR "tuberin" OR "mTOR") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),

    # === Epilepsy / Channelopathies ===
    ('neuro_epilepsy', '("epilepsy" OR "Dravet syndrome" OR "Lennox-Gastaut" OR "SCN1A" OR "SCN2A" OR "SCN8A" OR "KCNQ2" OR "KCNQ3" OR "GABRG2" OR "CHRNA4" OR "CHRNB2") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_channelopathy', '("episodic ataxia" OR "familial hemiplegic migraine" OR "CACNA1A" OR "CACNB4" OR "KCNA1" OR "KCND3" OR "ATP1A2" OR "SCN1A") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),

    # === Demyelinating / White matter ===
    ('neuro_demyelinating', '("multiple sclerosis" OR "MOG" OR "MBP" OR "PLP1" OR "leukodystrophy" OR "adrenoleukodystrophy" OR "ABCD1" OR "Krabbe disease" OR "GALC" OR "metachromatic leukodystrophy" OR "ARSA" OR "Alexander disease" OR "GFAP") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),

    # === Neuromuscular / Neuropathy ===
    ('neuro_neuromuscular', '("Charcot-Marie-Tooth" OR "PMP22" OR "MPZ" OR "GJB1" OR "MFN2" OR "GDAP1" OR "hereditary neuropathy" OR "spinal muscular atrophy" OR "SMN1" OR "NAIP" OR "IGHMBP2") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_myasthenia', '("myasthenia gravis" OR "CHRNE" OR "CHRNA1" OR "RAPSN" OR "DOK7" OR "congenital myasthenic" OR "MUSK" OR "LRP4") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_dystrophy', '("Duchenne muscular dystrophy" OR "Becker muscular dystrophy" OR "DMD" OR "dystrophin" OR "myotonic dystrophy" OR "DMPK" OR "CNBP" OR "FSHD" OR "DUX4" OR "facioscapulohumeral") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),

    # === Psychiatric genetics ===
    ('psych_schizophrenia', '("schizophrenia" OR "DISC1" OR "NRG1" OR "ERBB4" OR "DTNBP1" OR "COMT" OR "CACNA1C" OR "TCF4" OR "ZNF804A") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('psych_bipolar', '("bipolar disorder" OR "ANK3" OR "CACNA1C" OR "SYNE1" OR "ODZ4" OR "TRANK1" OR "NCAN") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('psych_depression', '("major depressive disorder" OR "SLC6A4" OR "HTR2A" OR "BDNF" OR "FKBP5" OR "CRHR1" OR "TPH2") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),

    # === Other neurological ===
    ('neuro_migraine', '("migraine" OR "familial hemiplegic" OR "CACNA1A" OR "ATP1A2" OR "SCN1A" OR "NOTCH3" OR "CSD") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_caa_stroke', '("cerebral amyloid angiopathy" OR "CADASIL" OR "NOTCH3" OR "HTRA1" OR "COL4A1" OR "COL4A2" OR "CARASIL" OR "MELAS" OR "mitochondrial encephalopathy" OR "POLG" OR "MT-") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_narcolepsy', '("narcolepsy" OR "hypocretin" OR "HCRT" OR "HCRTR1" OR "HCRTR2" OR "orexin" OR "cataplexy") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
    ('neuro_nbnia', '("Niemann-Pick" OR "Gaucher" OR "Tay-Sachs" OR "Sandhoff" OR "Batten disease" OR "neuronal ceroid lipofuscinosis" OR "NPC1" OR "NPC2" OR "GBA" OR "HEXA" OR "HEXB" OR "CLN3") AND (organism_id:9606) AND (length:[50 TO 400]) AND (reviewed:true) AND (fragment:false)'),
]


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


def fetch_uniprot_accessions(query, max_per_query=80):
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
    p = argparse.ArgumentParser(description='Build brain disease protein dataset')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_dataset_brain_disease'),
                   help='Output LMDB directory')
    p.add_argument('--max_per_query', type=int, default=80, help='Max proteins per disease category')
    p.add_argument('--af2_version', type=int, default=6, help='AFDB version')
    p.add_argument('--resume', action='store_true', help='Resume from existing LMDB')
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    processed_ids = set()
    entries = []
    if args.resume:
        import lmdb as _lmdb
        db_path = str(output_dir / 'brain_disease.lmdb')
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

    for category, query in BRAIN_DISEASE_QUERIES:
        print(f'\n{"="*60}')
        print(f'  {category}: searching UniProt...')
        print(f'   Query: {query[:100]}...')
        print(f'{"="*60}')
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
            print('SKIP (no AF2)')
            n_failed += 1
            continue
        cif_text = r.text

        plddt_list, mmcif_seq = extract_plddt_from_mmcif(cif_text)
        if not plddt_list:
            print('SKIP (no pLDDT)')
            n_failed += 1
            continue

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

        if len(plddt_list) < n_res:
            plddt_list += [0.5] * (n_res - len(plddt_list))
        plddt_tensor = torch.tensor(plddt_list[:n_res], dtype=torch.float32)

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
        entries.append(entry)
        print(f'OK (L={n_res}, pLDDT={plddt_tensor.mean():.3f}, pTM={iptm:.3f})')

        if len(entries) % 30 == 0:
            import lmdb as _lmdb
            db_path_tmp = str(output_dir / 'brain_disease.lmdb')
            if os.path.exists(db_path_tmp):
                import shutil as _shutil
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

    import lmdb
    db_path = str(output_dir / 'brain_disease.lmdb')
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
        'note': 'Brain/neurological disease focused dataset for BFN confidence training',
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved {len(entries)} brain-disease protein entries to {db_path}')
    print(f'Categories: {json.dumps(cat_counts, indent=2)}')
    print('Done!')


if __name__ == '__main__':
    main()
