#!/usr/bin/env python
"""
Build comprehensive IDP dataset for BFN confidence training.

Strategy:
  1. Fetch ALL DisProt entries (3199 IDPs) —source of IDP ground truth
  2. Query UniProt with disease keywords —mark disease association
  3. Process ALL DisProt entries + manual disease IDP lists (max coverage)
  4. Download AF2 PDBs, fetch per-residue disorder labels via EBI Proteins API
  5. Preprocess through BFN pipeline, save as LMDB

Key fix over v3: Uses EBI Proteins API (ebi.ac.uk) for disorder labels,
NOT the broken mobidb.org API.

Usage:
  python scripts/build/build_disease_idp_dataset.py
  python scripts/build/build_disease_idp_dataset.py --max_proteins 500
  python scripts/build/build_disease_idp_dataset.py --skip_disprot  # use only manual lists
"""

import sys, os, json, pickle, time, argparse, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import requests

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # scripts/build/ -> project root
sys.path.insert(0, str(PROJECT_DIR))

from build_utils import (
    AA3_TO_1, AA_LETTERS, compute_ptm_from_pae,
    resilient_get, preprocess_pdb_for_bfn, save_lmdb,
    split_and_save_entries
)

# ── Paths ──
DEFAULT_OUTPUT = PROJECT_DIR / 'data' / 'confidence_idp_disease'
V10_TRAIN = PROJECT_DIR / 'data' / 'confidence_merged_v10' / 'confidence_train.lmdb'
V10_VAL = PROJECT_DIR / 'data' / 'confidence_merged_v10' / 'confidence_val.lmdb'
AF2_CACHE = PROJECT_DIR / 'data' / 'idp_pdb_cache'
PDB_CACHE_DIR = PROJECT_DIR / 'data' / 'idp_disease_pdb_cache'

# ── UniProt REST API ──
UNIPROT_SEARCH = 'https://rest.uniprot.org/uniprotkb/search'

# ── Disease queries for UniProt (expanded from build_disease_dataset.py) ──
DISEASE_QUERIES = [
    ('neurodegeneration',
     '(Alzheimer OR Parkinson OR "Huntington" OR "amyotrophic lateral sclerosis" OR '
     '"prion" OR "frontotemporal dementia" OR "spinocerebellar ataxia" OR '
     '"Charcot-Marie-Tooth" OR "Lewy body" OR "multiple system atrophy" OR '
     '"corticobasal degeneration" OR "hereditary spastic paraplegia")'),
    ('cancer',
     '("tumor suppressor" OR oncogene OR "tyrosine kinase" OR '
     '("transcription factor" AND cancer) OR proto-oncogene OR '
     '"cell cycle" OR "DNA repair" AND cancer OR "breast cancer" OR '
     '"colorectal cancer" OR "lung cancer" OR "prostate cancer" OR '
     '"leukemia" OR "lymphoma" OR "melanoma" OR "glioblastoma")'),
    ('neurodevelopmental',
     '("autism" OR "schizophrenia" OR "bipolar disorder" OR "epilepsy" OR '
     '"intellectual disability" OR "Rett syndrome" OR "Fragile X" OR '
     '"Angelman syndrome" OR "Prader-Willi")'),
    ('cardiovascular',
     '(cardiomyopathy OR arrhythmia OR "long QT" OR "Brugada syndrome" OR '
     '"Marfan syndrome" OR "Noonan syndrome" OR "hypertension" OR '
     '"atherosclerosis" OR "heart failure" OR "aortic aneurysm")'),
    ('metabolic',
     '("diabetes" OR "obesity" OR "metabolic syndrome" OR "hypercholesterolemia" OR '
     '"lysosomal storage" OR "Gaucher" OR "Pompe" OR "Fabry" OR '
     '"glycogen storage" OR "mitochondrial disorder")'),
    ('immune_inflammatory',
     '("autoimmune" OR "rheumatoid arthritis" OR "lupus" OR "multiple sclerosis" OR '
     '"Crohn\'s disease" OR "ulcerative colitis" OR "psoriasis" OR '
     '"immunodeficiency" OR "severe combined immunodeficiency" OR "asthma")'),
    ('muscle_bone',
     '("muscular dystrophy" OR "osteogenesis imperfecta" OR "myopathy" OR '
     '"Ehlers-Danlos" OR "osteoporosis" OR "dwarfism" OR '
     '"amyotrophic lateral sclerosis" OR "spinal muscular atrophy" OR "sarcopenia")'),
    ('eye_neuro',
     '("retinitis pigmentosa" OR "macular degeneration" OR "cataract" OR '
     '"glaucoma" OR "Leber congenital amaurosis" OR "optic atrophy" OR '
     '"Usher syndrome" OR "Stargardt disease")'),
    ('blood_coagulation',
     '("hemophilia" OR "thalassemia" OR "sickle cell" OR "anemia" OR '
     '"thrombophilia" OR "von Willebrand" OR "myelodysplastic")'),
    ('liver_kidney',
     '(hepatitis OR cirrhosis OR hepatocellular OR NAFLD OR "Wilson disease" OR '
     '"alpha-1 antitrypsin" OR nephritis OR nephrotic OR "polycystic kidney" OR '
     '"renal failure" OR nephropathy OR "Alport syndrome")'),
    ('aging',
     '("progeria" OR "Werner syndrome" OR "telomere" OR "premature aging" OR '
     '"Cockayne syndrome" OR "xeroderma pigmentosum" OR "Bloom syndrome")'),
    ('infectious_host',
     '("virus receptor" OR "viral entry" OR "HIV" OR "influenza" OR '
     '"SARS-CoV-2" OR "Ebola" OR "malaria" OR "tuberculosis" OR '
     '"host-pathogen interaction")'),
]

# ── Manual disease IDP lists ──

# From misfolding_knowledge_base.py (IDP targets)
MISFOLDING_IDPS = [
    ('P05067', 'APP', 'Amyloid-beta —Alzheimer\'s'),
    ('P10636', 'MAPT', 'Tau —Alzheimer\'s'),
    ('P37840', 'SNCA', 'Alpha-synuclein —Parkinson\'s'),
    ('Q13148', 'TARDBP', 'TDP-43 —ALS/FTD'),
    ('P35637', 'FUS', 'FUS —ALS'),
    ('P42858', 'HTT', 'Huntingtin —Huntington\'s'),
    ('P10997', 'IAPP', 'Amylin —Type 2 Diabetes'),
    ('Q96LT7', 'C9orf72', 'C9orf72 —ALS/FTD'),
]

# From build_idp_dataset_v2.py KNOWN_IDPS (expanded)
KNOWN_IDPS = [
    # Neurodegeneration
    ('P10636', 'MAPT', 'Tau —Alzheimer\'s disease'),
    ('P37840', 'SNCA', 'Alpha-synuclein —Parkinson\'s disease'),
    ('P05067', 'APP', 'Amyloid-beta precursor —Alzheimer\'s'),
    ('P04156', 'PRNP', 'Prion protein —CJD'),
    ('Q13148', 'TARDBP', 'TDP-43 —ALS/FTD'),
    ('P35637', 'FUS', 'FUS —ALS'),
    # Cancer-related
    ('P04637', 'TP53', 'p53 tumor suppressor'),
    ('P01106', 'MYC', 'c-Myc oncogene'),
    ('P38398', 'BRCA1', 'Breast cancer type 1'),
    ('P38936', 'CDKN1A', 'p21/Waf1 —CDK inhibitor'),
    ('P46527', 'CDKN1B', 'p27/Kip1 —CDK inhibitor'),
    # Transcription factors
    ('P16220', 'CREB1', 'cAMP response element-binding protein'),
    ('Q92793', 'CREBBP', 'CREB-binding protein'),
    ('Q09472', 'EP300', 'p300 histone acetyltransferase'),
    ('Q00987', 'MDM2', 'E3 ubiquitin-protein ligase Mdm2'),
    # RNA-binding proteins
    ('P09651', 'HNRNPA1', 'hnRNP A1'),
    ('P07910', 'HNRNPC', 'hnRNP C1/C2'),
    ('P52597', 'HNRNPF', 'hnRNP F'),
    # Membrane-associated
    ('P02686', 'MBP', 'Myelin basic protein'),
    ('P02649', 'APOE', 'Apolipoprotein E'),
    # Cell cycle / signaling
    ('Q13541', 'EIF4EBP1', '4E-BP1 —translation repressor'),
    ('P62993', 'GRB2', 'Growth factor receptor-bound protein 2'),
    ('P27986', 'PIK3R1', 'PI3K regulatory subunit alpha'),
    # Structural / nuclear
    ('P07305', 'H1-0', 'Histone H1.0'),
    ('P16403', 'H1-2', 'Histone H1.2'),
    ('P06748', 'NPM1', 'Nucleophosmin'),
    ('P24928', 'POLR2A', 'RNA polymerase II CTD'),
    # Immune
    ('P05112', 'IL4', 'Interleukin-4'),
    ('P60568', 'IL2', 'Interleukin-2'),
    # Additional well-studied IDPs
    ('Q8WZ42', 'TTN', 'Titin'),
    ('P20936', 'RASA1', 'Ras GTPase-activating protein 1'),
    ('Q92574', 'TSC1', 'Hamartin'),
    ('P49841', 'GSK3B', 'Glycogen synthase kinase-3 beta'),
    ('P08047', 'SP1', 'Transcription factor Sp1'),
    ('Q04206', 'RELA', 'NF-kB p65 subunit'),
    ('P18848', 'ATF4', 'ATF-4'),
    ('P01100', 'FOS', 'c-Fos proto-oncogene'),
    ('P05412', 'JUN', 'c-Jun'),
    ('Q16665', 'HIF1A', 'Hypoxia-inducible factor 1-alpha'),
    ('P04626', 'ERBB2', 'Receptor tyrosine-protein kinase erbB-2'),
    # Low-complexity / prion-like domains
    ('Q8N9N5', 'RBM14', 'RNA-binding protein 14'),
    ('Q99700', 'ATXN2', 'Ataxin-2'),
    ('P54253', 'ATXN1', 'Ataxin-1'),
    ('O00592', 'PODXL', 'Podocalyxin'),
    # Additional known IDPs
    ('P35579', 'MYH9', 'Myosin-9'),
    ('Q15149', 'PLEC', 'Plectin'),
    ('P19338', 'NCL', 'Nucleolin'),
    ('P08238', 'HSP90AB1', 'Hsp90'),
    ('P07900', 'HSP90AA1', 'Hsp90 alpha'),
    ('Q00653', 'NFKB2', 'NF-kB p100'),
    ('P19838', 'NFKB1', 'NF-kB p105'),
    ('Q15648', 'MED1', 'Mediator subunit 1'),
    ('Q9UQB8', 'BAIAP2', 'IRSp53'),
    ('Q9NQX3', 'GPHN', 'Gephyrin'),
    ('Q05513', 'PRKCZ', 'Protein kinase C zeta'),
    ('Q02930', 'CREB5', 'CREB5'),
    ('Q9H4A3', 'WNK1', 'WNK1 kinase'),
    ('O14965', 'AURKA', 'Aurora kinase A'),
    ('P53350', 'PLK1', 'Polo-like kinase 1'),
    ('Q96J02', 'ITCH', 'E3 ubiquitin ligase ITCH'),
    ('Q9Y4E8', 'USP15', 'Ubiquitin carboxyl-terminal hydrolase 15'),
    ('Q9UJY1', 'HSPB8', 'Heat shock protein beta-8'),
    ('P02545', 'LMNA', 'Lamin A/C'),
    ('Q14103', 'HNRNPD', 'hnRNP D0'),
    ('Q99729', 'HNRNPAB', 'hnRNP A/B'),
    ('P51991', 'HNRNPA3', 'hnRNP A3'),
    ('Q13151', 'HNRNPA0', 'hnRNP A0'),
    ('O43390', 'HNRNPR', 'hnRNP R'),
    ('O60506', 'SYNCRIP', 'hnRNP Q'),
    ('Q9H6Z4', 'RANBP3', 'Ran-binding protein 3'),
    ('Q9Y6Y1', 'CAMTA1', 'Calmodulin-binding transcription activator 1'),
    ('Q9Y2X7', 'GIT1', 'ARF GTPase-activating protein GIT1'),
]


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Phase 1: Collect UniProt IDs
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def fetch_disprot_accessions():
    """Fetch ALL DisProt entries via paginated API. Returns {uniprot_acc: info}."""
    accessions = {}
    page = 1
    while True:
        url = f'https://disprot.org/api/search?show_ambiguous=false&page_size=500&page={page}'
        print(f'  DisProt page {page}...', end=' ', flush=True)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f'HTTP {r.status_code}')
                break
            data = r.json()
            items = data.get('data', [])
            if not items:
                print('done (empty page)')
                break
            for item in items:
                acc = item.get('acc', '')
                if acc:
                    accessions[acc] = {
                        'sequence': item.get('sequence', ''),
                        'length': item.get('length', 0),
                        'name': item.get('name', ''),
                        'organism': item.get('organism', ''),
                        'disorder_content': item.get('disorder_content', None),
                        'genes': item.get('genes', []),
                    }
            print(f'{len(items)} entries (total: {len(accessions)})')
            page += 1
            time.sleep(0.3)
        except Exception as e:
            print(f'error: {e}')
            break
    print(f'  Total DisProt entries: {len(accessions)}')
    return accessions


def fetch_uniprot_disease_ids(max_per_query=500):
    """Query UniProt for disease-associated proteins. Returns set of UniProt accessions."""
    all_ids = set()
    for category, query in DISEASE_QUERIES:
        full_query = f'({query}) AND (organism_id:9606) AND (reviewed:true) AND (fragment:false)'
        url = (f'{UNIPROT_SEARCH}?query={requests.utils.quote(full_query)}'
               f'&size={max_per_query}&format=tsv&fields=accession')
        print(f'  {category}: querying...', end=' ', flush=True)
        try:
            r = requests.get(url, timeout=30)
            lines = r.text.strip().split('\n')
            ids = set()
            for line in lines[1:]:  # skip header
                parts = line.split('\t')
                if parts and parts[0]:
                    ids.add(parts[0])
            all_ids.update(ids)
            print(f'{len(ids)} accessions')
        except Exception as e:
            print(f'error: {e}')
        time.sleep(0.5)
    print(f'  Total unique UniProt disease accessions: {len(all_ids)}')
    return all_ids


def load_existing_v10_ids():
    """Load all UniProt IDs already in v10 merged dataset."""
    existing = set()
    for lmdb_path in [V10_TRAIN, V10_VAL]:
        if not lmdb_path.exists():
            continue
        import lmdb
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            n = pickle.loads(txn.get(b'__len__'))
            for i in range(n):
                entry = pickle.loads(txn.get(f'{i:08d}'.encode()))
                existing.add(entry.get('pdb_id', ''))
        env.close()
    print(f'  Existing v10 entries: {len(existing)}')
    return existing


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Phase 2: Download & Process
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def fetch_disorder_regions(uniprot_id, seq_len):
    """Fetch per-residue disorder mask from EBI Proteins API (WORKING).

    Returns (mask, source) where mask is (seq_len,) float32 array.
    source: 'ebi_mobidb_lite' or 'failed'.
    """
    import urllib.request as ur
    for attempt in range(2):
        try:
            url = f'https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}'
            req = ur.Request(url, headers={'Accept': 'application/json'})
            with ur.urlopen(req, timeout=30) as f:
                data = json.loads(f.read())

            features = data.get('features', [])
            mask = np.zeros(seq_len, dtype=np.float32)
            found = False
            for feat in features:
                if feat['type'] == 'REGION' and 'disorder' in feat.get('description', '').lower():
                    start = int(feat['begin']) - 1
                    end = min(int(feat['end']), seq_len)
                    mask[start:end] = 1.0
                    found = True
            if found:
                return mask, 'ebi_mobidb_lite'
            break
        except Exception as e:
            if attempt < 1:
                time.sleep(1)
    return None, 'failed'


def af2_plddt_fallback(af2_plddt, threshold=50.0):
    """Use AF2 pLDDT < threshold as disorder indicator."""
    plddt = np.array(af2_plddt, dtype=np.float32)
    mask = (plddt < threshold).astype(np.float32)
    return mask, 'af2_plddt_fallback'


def download_af2_pdb(acc, cache_dir, version=6):
    """Download AF2 PDB from EBI AlphaFold DB. Returns Path or None."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f'AF-{acc}-F1-model_v{version}.pdb'
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    url = f'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{version}.pdb'
    r = resilient_get(url, timeout=60)
    if r is None or r.status_code != 200:
        return None
    content = r.content
    if not content or not any(content[:200].startswith(p) for p in
                              (b'HEADER', b'ATOM', b'HETATM', b'REMARK', b'TITLE', b'COMPND')):
        return None
    with open(out_path, 'wb') as f:
        f.write(content)
    return out_path


def fetch_af2_confidence(acc, version=6):
    """Fetch pLDDT, pTM, ipTM, PAE from EBI AlphaFold DB. Returns dict or None."""
    # pLDDT from mmCIF
    cif_url = f'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{version}.cif'
    cif_r = resilient_get(cif_url, timeout=60)
    if cif_r is None or cif_r.status_code != 200:
        return None

    cif_text = cif_r.text
    plddt = _parse_plddt_from_cif(cif_text)
    sequence = _parse_seq_from_cif(cif_text)

    # PAE from JSON
    pae_url = f'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-predicted_aligned_error_v{version}.json'
    pae_r = resilient_get(pae_url, timeout=60)
    pae_matrix = None
    ptm = 0.0
    if pae_r is not None and pae_r.status_code == 200:
        pae_data = pae_r.json()
        if isinstance(pae_data, list):
            pae_matrix = pae_data[0]['predicted_aligned_error']
            pae_matrix = np.array(pae_matrix, dtype=np.float32)
        elif isinstance(pae_data, dict):
            pae = pae_data.get('predicted_aligned_error',
                               pae_data.get('pae', [[]]))
            pae_matrix = np.array(pae, dtype=np.float32)
        ptm = compute_ptm_from_pae(pae_matrix) if pae_matrix is not None else 0.0

    if plddt is None or len(plddt) == 0:
        return None

    iptm = ptm  # approximate
    return {
        'plddt': plddt,
        'ptm': ptm,
        'iptm': iptm,
        'pae': pae_matrix,
        'sequence': sequence,
    }


def _parse_plddt_from_cif(cif_text):
    """Parse per-residue pLDDT from mmCIF text (B-factor of CA atoms)."""
    plddt = []
    import re
    lines = cif_text.split('\n')
    in_atom = False
    current_chain = None
    current_res = None
    for line in lines:
        if line.startswith('_atom_site.'):
            in_atom = True
            continue
        if not in_atom or not line.strip():
            continue
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < 21:
            continue
        try:
            group = parts[0]  # ATOM or HETATM
            if group != 'ATOM':
                continue
            atom_id = parts[1]
            label_atom = parts[3]
            if label_atom != 'CA':
                continue
            label_comp = parts[5]
            label_asym = parts[6]  # chain
            label_seq = parts[8]   # residue number
            bfactor = parts[14]    # B-factor = pLDDT * 100

            if label_comp not in AA3_TO_1:
                continue

            new_res = (label_asym, label_seq)
            if new_res != current_res:
                plddt.append(float(bfactor) / 100.0)
                current_res = new_res
                current_chain = label_asym
        except (ValueError, IndexError):
            continue
    return plddt


def _parse_seq_from_cif(cif_text):
    """Parse sequence from mmCIF text."""
    seq = []
    import re
    lines = cif_text.split('\n')
    in_atom = False
    current_res = None
    for line in lines:
        if line.startswith('_atom_site.'):
            in_atom = True
            continue
        if not in_atom or not line.strip() or line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < 21:
            continue
        try:
            if parts[0] != 'ATOM' or parts[3] != 'CA':
                continue
            label_comp = parts[5]
            label_asym = parts[6]
            label_seq = parts[8]
            res_key = (label_asym, label_seq)
            if res_key != current_res:
                if label_comp in AA3_TO_1:
                    seq.append(AA3_TO_1[label_comp])
                else:
                    seq.append('X')
                current_res = res_key
        except (ValueError, IndexError):
            continue
    return ''.join(seq)


def preprocess_af2_pdb(pdb_path):
    """Preprocess AF2 single-chain PDB into BFN-compatible batch dict.

    Mirrors build_idp_dataset_v3.py's approach: auto-detect chain ID,
    then use mask_region + merge_protein + patch_protein transforms.
    """
    from Bio.PDB import PDBParser
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.transforms import get_transform

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', str(pdb_path))
    model = structure[0]
    chains_info = {}
    for chain in model:
        seq = []
        for res in chain:
            if res.get_resname() in AA3_TO_1:
                seq.append(AA3_TO_1[res.get_resname()])
        if seq:
            chains_info[chain.id] = ''.join(seq)
    if not chains_info:
        return None, None, 0
    chain_id = list(chains_info.keys())[0]

    structure_data = preprocess_protein_structure(str(pdb_path), chain_ids=[chain_id])
    if structure_data is None:
        return None, None, 0

    chain_data = structure_data['chains'][0]['data']
    aa_indices = chain_data['aa']
    seq = ''.join(AA_LETTERS[a] if 0 <= a < 20 else 'X' for a in aa_indices.cpu())
    n_res = len(seq)

    transform = get_transform([
        {'type': 'mask_region', 'regions': {chain_id: list(range(n_res))}},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    data = transform(structure_data)
    batch = recursive_to(data, 'cpu')
    batch['generate_flag'] = torch.zeros(batch['aa'].shape[0], dtype=torch.bool)

    batch_clean = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_clean[k] = v.cpu()
        elif isinstance(v, (int, float, str, bool)):
            batch_clean[k] = v
        elif isinstance(v, (list, tuple)):
            batch_clean[k] = v
        else:
            batch_clean[k] = str(v)

    return batch_clean, seq, n_res


def process_one_protein(acc, pdb_cache, is_disease=False):
    """Download AF2 structure + disorder labels for one protein.
    Returns entry dict or None.
    """
    # 1. Download AF2 PDB
    pdb_path = download_af2_pdb(acc, pdb_cache)
    if pdb_path is None:
        return None

    # 2. Preprocess through BFN (v3-style: single-chain aware)
    try:
        batch, seq_str, n_res = preprocess_af2_pdb(str(pdb_path))
    except Exception:
        return None
    if batch is None or n_res < 20 or n_res > 2000:
        return None

    seq_chars = list(seq_str)
    if len(seq_chars) != n_res:
        return None

    # 3. Fetch AF2 confidence data
    conf = fetch_af2_confidence(acc)
    if conf is None:
        return None
    plddt_arr = conf['plddt']
    if len(plddt_arr) != n_res:
        min_len = min(len(plddt_arr), n_res)
        if min_len < 5:
            return None
        plddt_arr = plddt_arr[:min_len]
        seq_chars = seq_chars[:min_len]
        n_res = min_len

    # 4. Fetch disorder labels from EBI (WORKING API)
    disorder_mask, disorder_source = fetch_disorder_regions(acc, n_res)
    if disorder_mask is None:
        disorder_mask, disorder_source = af2_plddt_fallback(plddt_arr)

    disorder_fraction = float(disorder_mask.mean()) if disorder_mask is not None else -1.0

    # 5. Build entry
    pae = conf['pae']
    pae_tensor = torch.tensor(pae, dtype=torch.float32) if pae is not None else torch.zeros(n_res, n_res)

    source_tag = 'IDP_disease' if is_disease else 'IDP_DisProt'

    entry = {
        'pdb_id': acc,
        'sequence': ''.join(seq_chars),
        'batch': batch,
        'length': n_res,
        'af2_plddt': torch.tensor(plddt_arr[:n_res], dtype=torch.float32),
        'af2_iptm': torch.tensor(conf['iptm'], dtype=torch.float32),
        'af2_pae_matrix': pae_tensor,
        'af2_ptm': torch.tensor(conf['ptm'], dtype=torch.float32),
        'is_idp': True,
        'source': source_tag,
        'disorder_mask': torch.tensor(disorder_mask[:n_res], dtype=torch.float32),
        'disorder_fraction': disorder_fraction,
        'disorder_source': disorder_source,
        'disorder_content': None,
    }
    return entry


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Main
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def main():
    parser = argparse.ArgumentParser(description='Build disease-associated IDP dataset')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='Output directory')
    parser.add_argument('--max_proteins', type=int, default=None, help='Max proteins to process')
    parser.add_argument('--skip_disprot', action='store_true', help='Skip DisProt, use only manual lists')
    parser.add_argument('--workers', type=int, default=4, help='Parallel download workers')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Collect candidate UniProt IDs ──
    print('=' * 60)
    print('Phase 1: Collecting candidate UniProt IDs')
    print('=' * 60)

    # 1a. Manual lists
    manual_ids = set()
    for acc, gene, desc in MISFOLDING_IDPS + KNOWN_IDPS:
        manual_ids.add(acc)
    print(f'Manual curated IDs: {len(manual_ids)}')

    # 1b. DisProt API (all IDPs)
    disprot_ids = set()
    if not args.skip_disprot:
        disprot_info = fetch_disprot_accessions()
        disprot_ids = set(disprot_info.keys())
        print(f'DisProt IDs: {len(disprot_ids)}')
    else:
        disprot_info = {}

    # 1c. UniProt disease queries
    print('\nQuerying UniProt for disease-associated proteins...')
    uniprot_disease_ids = fetch_uniprot_disease_ids()

    # 1d. ALL DisProt IDs are IDPs —process all of them
    # Mark disease-associated ones with special source tag
    disease_idp_ids = disprot_ids & uniprot_disease_ids if disprot_ids else set()
    print(f'\nDisProt 鈭?UniProt-disease: {len(disease_idp_ids)} disease-associated IDPs')

    # 1e. Take ALL DisProt IDs + manual IDs = comprehensive IDP coverage
    all_candidates = disprot_ids | manual_ids
    print(f'Total candidates (all DisProt + manual): {len(all_candidates)}')
    print(f'  - Disease-associated subset: {len(disease_idp_ids)}')

    # 1f. Deduplicate against existing v10
    existing = load_existing_v10_ids()
    new_ids = sorted(all_candidates - existing)
    print(f'New IDs (not in v10): {len(new_ids)}')

    if args.max_proteins:
        new_ids = new_ids[:args.max_proteins]
        print(f'Limited to {args.max_proteins} proteins')

    if len(new_ids) == 0:
        print('No new IDs to process. Done.')
        return

    # ── Step 2: Download & process ──
    print(f'\n{"=" * 60}')
    print(f'Phase 2: Processing {len(new_ids)} proteins')
    print(f'{"=" * 60}')

    # Save candidate list for reference
    with open(output_dir / 'candidate_ids.json', 'w') as f:
        json.dump({'n_candidates': len(new_ids), 'ids': new_ids}, f, indent=2)

    entries = []
    n_success = 0
    n_failed = 0
    n_ebi_disorder = 0
    n_fallback_disorder = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_protein, acc, PDB_CACHE_DIR,
                                   acc in disease_idp_ids or acc in manual_ids): acc
                   for acc in new_ids}
        for i, future in enumerate(as_completed(futures)):
            acc = futures[future]
            try:
                entry = future.result()
            except Exception as e:
                print(f'[{i+1}/{len(new_ids)}] {acc}: exception {e}')
                n_failed += 1
                continue

            if entry is not None:
                entries.append(entry)
                n_success += 1
                src = entry.get('disorder_source', 'unknown')
                if src == 'ebi_mobidb_lite':
                    n_ebi_disorder += 1
                elif src == 'af2_plddt_fallback':
                    n_fallback_disorder += 1
                status = f'OK ({entry["length"]}aa, disorder={entry["disorder_fraction"]:.2f})'
            else:
                n_failed += 1
                status = 'FAILED'

            if (i + 1) % 10 == 0 or i == 0:
                print(f'[{i+1}/{len(new_ids)}] {acc}: {status} | '
                      f'success={n_success} fail={n_failed} '
                      f'ebi={n_ebi_disorder} fallback={n_fallback_disorder}')

    print(f'\nDone processing. Success: {n_success}, Failed: {n_failed}')
    print(f'Disorder labels: EBI={n_ebi_disorder}, AF2-fallback={n_fallback_disorder}')

    if n_success == 0:
        print('No successful entries. Exiting.')
        return

    # ── Step 3: Save LMDB ──
    print(f'\n{"=" * 60}')
    print(f'Phase 3: Saving {n_success} entries')
    print(f'{"=" * 60}')

    split_and_save_entries(entries, output_dir)
    print(f'Output: {output_dir}')

    # Summary
    summary = {
        'n_entries': n_success,
        'n_failed': n_failed,
        'n_ebi_disorder': n_ebi_disorder,
        'n_fallback_disorder': n_fallback_disorder,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sources': {
            'disprot_disease': len(disease_idp_ids),
            'manual_curated': len(manual_ids),
        },
    }
    with open(output_dir / 'build_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
