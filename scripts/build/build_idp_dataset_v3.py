#!/usr/bin/env python
"""Build IDP confidence dataset v3 — MobiDB per-residue disorder + expanded coverage.

Improvements over v2:
  - MobiDB API integration for per-residue disorder consensus annotations
  - Increased DisProt API coverage (10 pages × 500)
  - Per-residue `disorder_mask` tensor in LMDB entries
  - More literature-curated IDPs
  - Higher default max_proteins (300)

Usage:
  python build_idp_dataset_v3.py --output ./data/confidence_idp_v3 --max_proteins 300 --disprot_pages 10
"""

import sys, os, json, time, argparse, pickle, io
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import requests
from urllib3 import Retry
from requests.adapters import HTTPAdapter

PROJECT_DIR = Path(__file__).parent.parent

# ── Well-known IDPs from literature ──
KNOWN_IDPS = [
    # Neurodegeneration
    ('P10636', 'MAPT', 'Tau — Alzheimer\'s disease'),
    ('P37840', 'SNCA', 'Alpha-synuclein — Parkinson\'s disease'),
    ('P05067', 'APP', 'Amyloid-beta precursor — Alzheimer\'s'),
    ('P04156', 'PRNP', 'Prion protein — CJD'),
    ('Q13148', 'TARDBP', 'TDP-43 — ALS/FTD'),
    ('P35637', 'FUS', 'FUS — ALS'),
    ('Q9UH65', 'SETX', 'Senataxin — ALS4'),
    ('Q99720', 'SIGMAR1', 'Sigma-1 receptor — ALS16'),
    ('O00499', 'BIN1', 'Myc box-dependent-interacting protein 1 — Alzheimer\'s risk'),
    ('Q9Y6H1', 'CHCHD2', 'Coiled-coil-helix-coiled-coil-helix domain containing 2 — Parkinson\'s'),
    # Cancer-related
    ('P04637', 'TP53', 'p53 tumor suppressor'),
    ('P01106', 'MYC', 'c-Myc oncogene'),
    ('P38398', 'BRCA1', 'Breast cancer type 1'),
    ('P38936', 'CDKN1A', 'p21/Waf1 — cyclin-dependent kinase inhibitor'),
    ('P46527', 'CDKN1B', 'p27/Kip1 — cyclin-dependent kinase inhibitor'),
    ('Q15303', 'ERBB4', 'Receptor tyrosine-protein kinase erbB-4'),
    ('Q13501', 'SQSTM1', 'Sequestosome-1 / p62 — autophagy receptor'),
    # Transcription factors
    ('P16220', 'CREB1', 'cAMP response element-binding protein'),
    ('Q92793', 'CREBBP', 'CREB-binding protein'),
    ('Q09472', 'EP300', 'p300 histone acetyltransferase'),
    ('Q00987', 'MDM2', 'E3 ubiquitin-protein ligase Mdm2'),
    # RNA-binding proteins
    ('P09651', 'HNRNPA1', 'hnRNP A1'),
    ('P07910', 'HNRNPC', 'hnRNP C1/C2'),
    ('P52597', 'HNRNPF', 'hnRNP F'),
    ('Q99729', 'HNRNPAB', 'hnRNP A/B'),
    ('P51991', 'HNRNPA3', 'hnRNP A3'),
    ('Q13151', 'HNRNPA0', 'hnRNP A0'),
    ('O43390', 'HNRNPR', 'hnRNP R'),
    ('O60506', 'SYNCRIP', 'hnRNP Q — SYNCRIP'),
    ('Q14103', 'HNRNPD', 'hnRNP D0 — AU-rich element RNA-binding'),
    # Membrane-associated
    ('P02686', 'MBP', 'Myelin basic protein'),
    ('P02649', 'APOE', 'Apolipoprotein E'),
    # Cell cycle / signaling
    ('Q13541', 'EIF4EBP1', '4E-BP1 — translation repressor'),
    ('P62993', 'GRB2', 'Growth factor receptor-bound protein 2'),
    ('P27986', 'PIK3R1', 'PI3K regulatory subunit alpha'),
    ('P49841', 'GSK3B', 'Glycogen synthase kinase-3 beta'),
    # Structural / nuclear
    ('P07305', 'H1-0', 'Histone H1.0 — linker histone'),
    ('P16403', 'H1-2', 'Histone H1.2'),
    ('P16401', 'H1-5', 'Histone H1.5'),
    ('P06748', 'NPM1', 'Nucleophosmin'),
    ('P24928', 'POLR2A', 'RNA polymerase II CTD'),
    ('P19338', 'NCL', 'Nucleolin — highly disordered'),
    ('P02545', 'LMNA', 'Lamin A/C — partially disordered'),
    # Immune
    ('P05112', 'IL4', 'Interleukin-4'),
    ('P60568', 'IL2', 'Interleukin-2'),
    # Additional well-studied IDPs
    ('Q8WZ42', 'TTN', 'Titin — giant sarcomeric protein'),
    ('P20936', 'RASA1', 'Ras GTPase-activating protein 1'),
    ('Q92574', 'TSC1', 'Hamartin — TSC complex'),
    ('P08047', 'SP1', 'Transcription factor Sp1'),
    ('Q04206', 'RELA', 'NF-kB p65 subunit'),
    ('P18848', 'ATF4', 'Cyclic AMP-dependent transcription factor ATF-4'),
    ('P01100', 'FOS', 'c-Fos proto-oncogene'),
    ('P05412', 'JUN', 'c-Jun transcription factor'),
    ('Q16665', 'HIF1A', 'Hypoxia-inducible factor 1-alpha'),
    ('P04626', 'ERBB2', 'Receptor tyrosine-protein kinase erbB-2'),
    # Low-complexity / prion-like domains
    ('Q8N9N5', 'RBM14', 'RNA-binding protein 14'),
    ('Q99700', 'ATXN2', 'Ataxin-2'),
    ('P54253', 'ATXN1', 'Ataxin-1'),
    ('O00592', 'PODXL', 'Podocalyxin'),
    # Additional known IDPs
    ('P35579', 'MYH9', 'Myosin-9 — contains disordered tail'),
    ('Q15149', 'PLEC', 'Plectin — large disordered regions'),
    ('P08238', 'HSP90AB1', 'Hsp90 — disordered linkers'),
    ('P07900', 'HSP90AA1', 'Hsp90 alpha — disordered linkers'),
    ('Q00653', 'NFKB2', 'NF-kB p100 subunit'),
    ('P19838', 'NFKB1', 'NF-kB p105 subunit'),
    ('Q15648', 'MED1', 'Mediator of RNA pol II subunit 1'),
    ('Q9UQB8', 'BAIAP2', 'IRSp53 — inverse BAR domain protein'),
    ('Q9NQX3', 'GPHN', 'Gephyrin — scaffold protein'),
    ('Q05513', 'PRKCZ', 'Protein kinase C zeta'),
    ('Q02930', 'CREB5', 'CREB5 transcription factor'),
    ('Q9H4A3', 'WNK1', 'WNK1 kinase — disordered'),
    ('O14965', 'AURKA', 'Aurora kinase A — disordered N-term'),
    ('P53350', 'PLK1', 'Polo-like kinase 1 — disordered'),
    ('Q96J02', 'ITCH', 'E3 ubiquitin ligase ITCH'),
    ('Q9Y4E8', 'USP15', 'Ubiquitin carboxyl-terminal hydrolase 15'),
    ('Q9UJY1', 'HSPB8', 'Heat shock protein beta-8'),
    ('Q9H6Z4', 'RANBP3', 'Ran-binding protein 3'),
    ('Q9Y6Y1', 'CAMTA1', 'Calmodulin-binding transcription activator 1'),
    ('Q9Y2X7', 'GIT1', 'ARF GTPase-activating protein GIT1'),
    # Expanded neurodegenerative disease related
    ('P42858', 'HTT', 'Huntingtin — Huntington\'s disease'),
    ('Q9UQN3', 'CHMP2B', 'Charged multivesicular body protein 2b — FTD'),
    ('Q96CV9', 'OPTN', 'Optineurin — ALS/glaucoma'),
    ('Q9NQW7', 'XPNPEP1', 'Xaa-Pro aminopeptidase 1'),
    ('P52565', 'ARHGDIA', 'Rho GDP-dissociation inhibitor 1'),
    # Transcription/regulatory (expanded)
    ('P10275', 'AR', 'Androgen receptor'),
    ('P03372', 'ESR1', 'Estrogen receptor'),
    ('Q92731', 'ESR2', 'Estrogen receptor beta'),
    ('P06401', 'PGR', 'Progesterone receptor'),
    ('P35222', 'CTNNB1', 'Catenin beta-1 — Wnt signaling'),
    ('Q13315', 'ATM', 'Ataxia telangiectasia mutated — DNA damage'),
    ('P51587', 'BRCA2', 'Breast cancer type 2 susceptibility protein'),
    ('Q15831', 'STK11', 'Serine/threonine-protein kinase 11'),
    ('Q92934', 'BAD', 'BCL2 associated agonist of cell death'),
    # Additional IDPs with well-characterized disordered regions
    ('P49736', 'MCM2', 'DNA replication licensing factor MCM2'),
    ('P33992', 'MCM5', 'DNA replication licensing factor MCM5'),
    ('Q14566', 'MCM6', 'DNA replication licensing factor MCM6'),
    ('P33993', 'MCM7', 'DNA replication licensing factor MCM7'),
    ('O43463', 'SUV39H1', 'Histone-lysine N-methyltransferase SUV39H1'),
    ('Q96T23', 'RSF1', 'Remodeling and spacing factor 1'),
    ('Q12824', 'SMARCB1', 'SWI/SNF-related matrix-associated actin-dependent regulator'),
    ('P51532', 'SMARCA4', 'Transcription activator BRG1'),
    ('Q9HBD4', 'BCL11A', 'B-cell lymphoma/leukemia 11A'),
    ('Q9H165', 'BCL11B', 'B-cell lymphoma/leukemia 11B'),
    ('Q9UJU6', 'DBNL', 'Drebrin-like protein'),
    ('Q9UHB6', 'LIMA1', 'LIM domain and actin-binding protein 1'),
    ('Q16643', 'DBN1', 'Drebrin'),
    ('Q13813', 'SPTAN1', 'Spectrin alpha chain, non-erythrocytic 1'),
    ('Q01082', 'SPTBN1', 'Spectrin beta chain, non-erythrocytic 1'),
    ('P11142', 'HSPA8', 'Heat shock cognate 71 kDa protein'),
    ('P0DMV8', 'HSPA1A', 'Heat shock 70 kDa protein 1A'),
    ('P34932', 'HSPA4', 'Heat shock 70 kDa protein 4'),
]


def parse_args():
    p = argparse.ArgumentParser(description='Build IDP confidence dataset v3')
    p.add_argument('--output', default=str(PROJECT_DIR / 'data' / 'confidence_idp_v3'),
                   help='Output LMDB directory')
    p.add_argument('--max_proteins', type=int, default=300, help='Max IDPs to process')
    p.add_argument('--min_len', type=int, default=30, help='Min sequence length')
    p.add_argument('--max_len', type=int, default=500, help='Max sequence length')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--afdb_version', type=int, default=6, help='AlphaFold DB version (4 or 6)')
    p.add_argument('--disprot_pages', type=int, default=10, help='Number of DisProt API pages to fetch (500/page)')
    p.add_argument('--pdb_cache', default=str(PROJECT_DIR / 'data' / 'idp_pdb_cache'),
                   help='Cache directory for AF2 PDB files')
    p.add_argument('--skip_mobidb', action='store_true', help='Skip MobiDB lookup (faster, no per-residue mask)')
    return p.parse_args()


def resilient_get(url, timeout=30):
    """HTTP GET with retry."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session.get(url, timeout=timeout)


def fetch_disprot_accessions(n_pages=10):
    """Fetch IDP accessions from DisProt API (multiple pages)."""
    accessions = {}
    for page in range(1, n_pages + 1):
        try:
            url = f'https://disprot.org/api/search?show_ambiguous=false&page_size=500&page={page}'
            resp = resilient_get(url, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                entries = data.get('data', [])
                if not entries:
                    break
                for entry in entries:
                    acc = entry.get('acc')
                    if acc:
                        accessions[acc] = {
                            'sequence': entry.get('sequence', ''),
                            'length': entry.get('length', 0),
                            'name': entry.get('name', ''),
                            'organism': entry.get('organism', ''),
                            'disorder_content': entry.get('disorder_content', None),
                        }
                print(f'  Page {page}: {len(entries)} entries (total unique: {len(accessions)})')
            else:
                print(f'  Page {page}: HTTP {resp.status_code}, stopping')
                break
        except Exception as e:
            print(f'  Page {page}: Error — {e}')
            break
    return accessions


def fetch_mobidb_disorder(uniprot_id):
    """Fetch per-residue disorder consensus from MobiDB. Returns (mask, fraction) or (None, None)."""
    try:
        url = f'https://mobidb.org/api/download/{uniprot_id}'
        resp = resilient_get(url, timeout=60)
        if resp.status_code != 200:
            return None, None

        data = resp.json()

        # MobiDB consensus is stored as region annotations
        # Format: {"mobidb_consensus": {"disorder": {"predictors": [{"regions": [[start,end],...]}]}}}
        consensus = data.get('mobidb_consensus', {})
        disorder_data = consensus.get('disorder', {})
        predictors = disorder_data.get('predictors', [])

        if not predictors:
            # Fallback: try other known structure
            disorder_data = consensus.get('disorder', consensus)
            if isinstance(disorder_data, dict):
                predictors = disorder_data.get('predictors', [])

        regions = []
        for pred in predictors:
            if pred.get('type') == 'disorder' or 'disorder' in pred.get('name', '').lower():
                regions = pred.get('regions', [])
                break

        if not regions and predictors:
            # Take first predictor's regions
            regions = predictors[0].get('regions', [])

        # Get sequence to determine length
        sequence = data.get('sequence', '')
        seq_len = len(sequence)

        if not regions or seq_len == 0:
            return None, None

        # Build per-residue mask from region annotations
        disorder_mask = np.zeros(seq_len, dtype=np.float32)
        for region in regions:
            start, end = region[0], region[1]
            # MobiDB uses 1-based indexing
            start = max(0, start - 1)
            end = min(seq_len, end)
            disorder_mask[start:end] = 1.0

        fraction = float(disorder_mask.mean())
        return disorder_mask, fraction

    except Exception as e:
        return None, None


def fetch_uniprot_sequence(accession):
    """Fetch amino acid sequence from UniProt."""
    try:
        resp = resilient_get(f'https://rest.uniprot.org/uniprotkb/{accession}.fasta')
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            seq = ''.join(line.strip() for line in lines if not line.startswith('>'))
            return seq
    except Exception:
        pass
    return None


def download_af2_pdb(accession, cache_dir, version=6):
    """Download AF2 predicted PDB from EBI AlphaFold DB. Returns Path or None."""
    pdb_path = Path(cache_dir) / f'AF-{accession}-F1-model_v{version}.pdb'
    if pdb_path.exists():
        return pdb_path

    for v in [version, 4]:
        url = f'https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{v}.pdb'
        try:
            resp = resilient_get(url, timeout=120)
            if resp.status_code == 200:
                pdb_path = Path(cache_dir) / f'AF-{accession}-F1-model_v{v}.pdb'
                pdb_path.write_text(resp.text)
                return pdb_path
        except Exception:
            continue
    return None


def fetch_af2_confidence(accession, version=6):
    """Fetch AF2 pLDDT and PAE from EBI AlphaFold DB. Returns dict."""
    result = {'plddt_array': None, 'pae_matrix': None, 'iptm': None, 'ptm': None}

    for v in [version, 4]:
        try:
            cif_url = f'https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{v}.cif'
            resp = resilient_get(cif_url)
            if resp.status_code == 200:
                plddt_vals = []
                in_atom_site = False
                for line in resp.text.split('\n'):
                    if '_atom_site.' in line:
                        in_atom_site = True
                        continue
                    if in_atom_site and line.startswith('ATOM'):
                        parts = line.split()
                        if len(parts) > 16 and parts[3] == 'CA':
                            try:
                                bfactor = float(parts[14])
                                plddt_vals.append(bfactor / 100.0)
                            except ValueError:
                                continue
                    elif in_atom_site and not line.startswith('ATOM'):
                        break
                if plddt_vals:
                    result['plddt_array'] = plddt_vals
                break
        except Exception:
            continue

    for v in [version, 4]:
        try:
            pae_url = f'https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-predicted_aligned_error_v{v}.json'
            resp = resilient_get(pae_url)
            if resp.status_code == 200:
                pae_data = resp.json()
                pae_list = pae_data[0]['predicted_aligned_error'] if isinstance(pae_data, list) else pae_data.get('predicted_aligned_error')
                if pae_list:
                    result['pae_matrix'] = pae_list
                    result['ptm'] = _compute_ptm_from_pae(pae_list)
                break
        except Exception:
            continue

    result['iptm'] = result['ptm']
    return result


def _compute_ptm_from_pae(pae):
    """Compute pTM from PAE matrix."""
    pa = np.array(pae)
    L = pa.shape[0]
    d0 = max(1.24 * (L - 15) ** (1.0 / 3) - 1.8, 0.5)
    tm_scores = 1.0 / (1.0 + (pa / d0) ** 2)
    return float(np.max(np.mean(tm_scores, axis=1)))


def preprocess_af2_pdb(pdb_path):
    """Preprocess AF2 PDB into BFN-compatible batch dict."""
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.transforms import get_transform

    from Bio.PDB import PDBParser
    aa3_to_1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
        'TRP': 'W', 'TYR': 'Y',
    }
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', str(pdb_path))
        model = structure[0]
        chains_info = {}
        for chain in model:
            seq = []
            for res in chain:
                if res.get_resname() in aa3_to_1:
                    seq.append(aa3_to_1[res.get_resname()])
            if seq:
                chains_info[chain.id] = ''.join(seq)
        if not chains_info:
            return None, None, 0
        chain_id = list(chains_info.keys())[0]
    except Exception:
        return None, None, 0

    try:
        structure_data = preprocess_protein_structure(str(pdb_path), chain_ids=[chain_id])
        if structure_data is None:
            return None, None, 0
    except Exception:
        return None, None, 0

    chain_data = structure_data['chains'][0]['data']
    aa_indices = chain_data['aa']
    AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
    seq = ''.join(AA_LETTERS[a] if 0 <= a < 20 else 'X' for a in aa_indices.cpu())
    n_res = len(seq)

    try:
        transform = get_transform([
            {'type': 'mask_region', 'regions': {chain_id: list(range(n_res))}},
            {'type': 'merge_protein'},
            {'type': 'patch_protein'},
        ])
        data = transform(structure_data)
        batch = recursive_to(data, 'cpu')
    except Exception:
        return None, None, 0

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


def save_lmdb(entries, db_path):
    """Save entries to LMDB."""
    import lmdb
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    if not entries:
        print('  No entries to save!')
        return

    sample = pickle.dumps(entries[0])
    est_size = max(len(sample) * len(entries) * 3 + 10 * 1024 * 1024, 20 * 1024 * 1024)
    env = lmdb.open(str(db_path), map_size=est_size)
    with env.begin(write=True) as txn:
        for j, entry in enumerate(entries):
            txn.put(f'{j:08d}'.encode(), pickle.dumps(entry))
        txn.put(b'__len__', pickle.dumps(len(entries)))
    env.close()
    print(f'  Saved {len(entries)} entries to LMDB: {db_path}')


# ── Main ──
if __name__ == '__main__':
    args = parse_args()
    output_dir = Path(args.output)
    pdb_cache_dir = Path(args.pdb_cache)
    pdb_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect IDP accessions
    accessions_map = {}

    for acc, gene, desc in KNOWN_IDPS:
        if acc not in accessions_map:
            accessions_map[acc] = {'gene': gene, 'description': desc, 'source': 'literature'}
    print(f'Literature-curated IDPs: {len(accessions_map)}')

    print('Fetching DisProt entries...')
    disprot = fetch_disprot_accessions(n_pages=args.disprot_pages)
    n_new = 0
    for acc, info in disprot.items():
        if acc not in accessions_map:
            accessions_map[acc] = {
                'gene': info.get('name', ''),
                'description': f"{info.get('name','')} ({info.get('organism','')})",
                'source': 'DisProt',
                'disprot_seq': info.get('sequence', ''),
                'disprot_length': info.get('length', 0),
                'disorder_content': info.get('disorder_content'),
            }
            n_new += 1
    print(f'Total unique IDP accessions: {len(accessions_map)} ({n_new} new from DisProt)')

    sorted_accessions = sorted(accessions_map.keys(),
                               key=lambda a: (0 if accessions_map[a].get('source') == 'literature' else 1, a))
    sorted_accessions = sorted_accessions[:args.max_proteins]

    # 2. Process each IDP
    entries = []
    n_processed = 0
    n_skipped_len = 0
    n_skipped_nopdb = 0
    n_skipped_preproc = 0
    n_mobidb_ok = 0
    n_total = len(sorted_accessions)

    for i, acc in enumerate(sorted_accessions):
        info = accessions_map[acc]
        gene_name = info.get('gene', '?')
        print(f'[{i+1}/{n_total}] {acc} ({gene_name}): ', end='', flush=True)

        seq = info.get('disprot_seq', '')
        if seq:
            n_res = info.get('disprot_length', len(seq))
        else:
            seq = fetch_uniprot_sequence(acc)
            n_res = len(seq) if seq else 0

        if not seq:
            print('FAILED (no sequence)')
            continue

        n_res = len(seq)
        if n_res < args.min_len:
            print(f'SKIPPED (too short: {n_res}aa)')
            n_skipped_len += 1
            continue
        if n_res > args.max_len:
            print(f'SKIPPED (too long: {n_res}aa)')
            n_skipped_len += 1
            continue

        # MobiDB per-residue disorder
        disorder_mask = None
        disorder_fraction = None
        if not args.skip_mobidb:
            disorder_mask_arr, disorder_frac = fetch_mobidb_disorder(acc)
            if disorder_mask_arr is not None:
                n_mobidb_ok += 1
                disorder_mask = disorder_mask_arr
                disorder_fraction = disorder_frac

        pdb_path = download_af2_pdb(acc, str(pdb_cache_dir), version=args.afdb_version)
        if not pdb_path:
            print('SKIPPED (no AF2 PDB available)')
            n_skipped_nopdb += 1
            continue

        batch, batch_seq, batch_nres = preprocess_af2_pdb(pdb_path)
        if batch is None:
            print('SKIPPED (preprocessing failed)')
            n_skipped_preproc += 1
            continue

        af2 = fetch_af2_confidence(acc, version=args.afdb_version)
        plddt = af2.get('plddt_array')
        if not plddt:
            print(f'SKIPPED (no AF2 confidence data)')
            n_skipped_nopdb += 1
            continue

        avg_plddt = np.mean(plddt)
        if len(plddt) < batch_nres:
            plddt = list(plddt) + [0.5] * (batch_nres - len(plddt))
        plddt = plddt[:batch_nres]

        pae = af2.get('pae_matrix')
        af2_pae_tensor = torch.zeros(batch_nres, batch_nres)
        if pae is not None:
            pae_np = np.array(pae)
            if pae_np.shape[0] >= batch_nres and pae_np.shape[1] >= batch_nres:
                af2_pae_tensor = torch.from_numpy(pae_np[:batch_nres, :batch_nres]).float()

        # Truncate/pad disorder_mask to match batch_nres
        if disorder_mask is not None:
            if len(disorder_mask) < batch_nres:
                disorder_mask = np.pad(disorder_mask, (0, batch_nres - len(disorder_mask)), mode='constant', constant_values=0.5)
            disorder_mask = disorder_mask[:batch_nres]
            disorder_tensor = torch.from_numpy(disorder_mask.copy()).float()
        else:
            disorder_tensor = torch.full((batch_nres,), -1.0)  # -1 = unknown

        entry = {
            'pdb_id': acc,
            'sequence': seq[:batch_nres],
            'batch': batch,
            'length': batch_nres,
            'af2_plddt': torch.tensor(plddt, dtype=torch.float32),
            'af2_iptm': torch.tensor(af2.get('iptm') or af2.get('ptm') or 0.5, dtype=torch.float32),
            'af2_pae_matrix': af2_pae_tensor,
            'af2_ptm': af2.get('ptm'),
            'is_idp': True,
            'source': f'IDP_{info.get("source", "unknown")}',
            'disorder_mask': disorder_tensor,
            'disorder_fraction': disorder_fraction if disorder_fraction is not None else -1.0,
            'disorder_content': info.get('disorder_content'),
        }
        entries.append(entry)
        n_processed += 1

        mobi_flag = ' [MobiDB]' if disorder_mask is not None else ''
        print(f'{batch_nres}aa, avg_pLDDT={avg_plddt:.3f}{mobi_flag}')

    print(f'\nProcessed: {n_processed} | MobiDB: {n_mobidb_ok} | '
          f'Skipped (len): {n_skipped_len} | '
          f'Skipped (no AF2): {n_skipped_nopdb} | Skipped (preproc): {n_skipped_preproc}')

    # 3. Save
    if entries:
        save_lmdb(entries, str(output_dir))

        avg_plddts = [np.mean(e['af2_plddt'].numpy() if isinstance(e['af2_plddt'], torch.Tensor) else e['af2_plddt']) for e in entries]
        disorder_fracs = [e['disorder_fraction'] for e in entries if e.get('disorder_fraction', -1) >= 0]
        n_with_mobidb = sum(1 for e in entries if e.get('disorder_fraction', -1) >= 0)

        summary = {
            'n_entries': len(entries),
            'n_with_mobidb': n_with_mobidb,
            'source': f'DisProt ({n_new} new) + literature ({len(KNOWN_IDPS)} curated) IDPs + MobiDB per-residue',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'avg_plddt': float(np.mean(avg_plddts)) if avg_plddts else 0,
            'min_plddt': float(np.min(avg_plddts)) if avg_plddts else 0,
            'max_plddt': float(np.max(avg_plddts)) if avg_plddts else 0,
            'avg_disorder_fraction': float(np.mean(disorder_fracs)) if disorder_fracs else -1,
        }
        with open(output_dir / 'idp_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'Average AF2 pLDDT across {len(avg_plddts)} IDPs: {np.mean(avg_plddts):.3f}')
        print(f'  Min: {np.min(avg_plddts):.3f} | Max: {np.max(avg_plddts):.3f}')
        if disorder_fracs:
            print(f'Average MobiDB disorder fraction: {np.mean(disorder_fracs):.3f}')
            print(f'  Entries with per-residue disorder: {n_with_mobidb}/{len(entries)}')

    print(f'\nDone! Results saved to {output_dir}')
