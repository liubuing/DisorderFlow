#!/usr/bin/env python
"""Build an IDP (Intrinsically Disordered Protein) dataset for BFN confidence training.

Sources:
  - DisProt (https://disprot.org) — curated database of experimentally validated IDPs
  - Literature-curated well-known IDPs

Usage:
  python build_idp_dataset.py --output ./data/confidence_idp --max_proteins 100
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

PROJECT_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description='Build IDP confidence dataset')
    p.add_argument('--output', default=str(PROJECT_DIR / 'data' / 'confidence_idp'),
                   help='Output LMDB directory')
    p.add_argument('--max_proteins', type=int, default=100, help='Max IDPs to process')
    p.add_argument('--min_len', type=int, default=30, help='Min sequence length')
    p.add_argument('--max_len', type=int, default=500, help='Max sequence length (IDPs can be long)')
    p.add_argument('--seed', type=int, default=2026, help='Random seed')
    p.add_argument('--resume', action='store_true', help='Skip already-processed proteins')
    return p.parse_args()


def resilient_get(url, timeout=30):
    """HTTP GET with retry."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session.get(url, timeout=timeout)


# ── Well-known IDPs from literature ──
# Each entry: (uniprot_accession, gene_name, description)
KNOWN_IDPS = [
    # Neurodegeneration
    ('P10636', 'MAPT', 'Tau — Alzheimer\'s disease'),
    ('P37840', 'SNCA', 'Alpha-synuclein — Parkinson\'s disease'),
    ('P05067', 'APP', 'Amyloid-beta precursor — Alzheimer\'s'),
    ('P04156', 'PRNP', 'Prion protein — CJD'),
    ('Q13148', 'TARDBP', 'TDP-43 — ALS/FTD'),
    ('P35637', 'FUS', 'FUS — ALS'),
    # Cancer-related
    ('P04637', 'TP53', 'p53 tumor suppressor'),
    ('P01106', 'MYC', 'c-Myc oncogene'),
    ('P38398', 'BRCA1', 'Breast cancer type 1'),
    ('P38936', 'CDKN1A', 'p21/Waf1 — cyclin-dependent kinase inhibitor'),
    ('P46527', 'CDKN1B', 'p27/Kip1 — cyclin-dependent kinase inhibitor'),
    # Transcription factors
    ('P16220', 'CREB1', 'cAMP response element-binding protein'),
    ('Q92793', 'CREBBP', 'CREB-binding protein'),
    ('Q09472', 'EP300', 'p300 histone acetyltransferase'),
    ('Q00987', 'MDM2', 'E3 ubiquitin-protein ligase Mdm2'),
    # RNA-binding proteins (often disordered)
    ('P09651', 'HNRNPA1', 'hnRNP A1'),
    ('P07910', 'HNRNPC', 'hnRNP C1/C2'),
    ('P52597', 'HNRNPF', 'hnRNP F'),
    # Membrane-associated
    ('P02686', 'MBP', 'Myelin basic protein'),
    ('P02649', 'APOE', 'Apolipoprotein E'),
    # Cell cycle / signaling
    ('Q13541', 'EIF4EBP1', '4E-BP1 — translation repressor'),
    ('P62993', 'GRB2', 'Growth factor receptor-bound protein 2'),
    ('P27986', 'PIK3R1', 'PI3K regulatory subunit alpha'),
    # Structural / nuclear
    ('P07305', 'H1-0', 'Histone H1.0 — linker histone'),
    ('P16403', 'H1-2', 'Histone H1.2'),
    ('P06748', 'NPM1', 'Nucleophosmin'),
    ('P24928', 'POLR2A', 'RNA polymerase II CTD'),
    # Immune
    ('P05112', 'IL4', 'Interleukin-4'),
    ('P60568', 'IL2', 'Interleukin-2'),
    # Additional well-studied IDPs
    ('Q8WZ42', 'TTN', 'Titin — giant sarcomeric protein (partial disorder)'),
    ('P20936', 'RASA1', 'Ras GTPase-activating protein 1'),
    ('Q92574', 'TSC1', 'Hamartin — TSC complex'),
    ('P49841', 'GSK3B', 'Glycogen synthase kinase-3 beta'),
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
]


def fetch_disprot_accessions():
    """Try to fetch IDP accessions from DisProt API."""
    accessions = set()
    try:
        # DisProt REST API — get all entries
        resp = resilient_get('https://disprot.org/api/search?release=latest&show_ambiguous=false&page_size=500')
        if resp.status_code == 200:
            data = resp.json()
            for entry in data.get('results', []):
                acc = entry.get('acc')
                if acc:
                    accessions.add(acc)
            print(f'  Fetched {len(accessions)} entries from DisProt API')
    except Exception as e:
        print(f'  DisProt API unavailable: {e}')
    return accessions


def fetch_uniprot_sequence(accession):
    """Fetch amino acid sequence from UniProt."""
    try:
        resp = resilient_get(f'https://rest.uniprot.org/uniprotkb/{accession}.fasta')
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            seq = ''.join(line.strip() for line in lines if not line.startswith('>'))
            return seq
    except Exception as e:
        print(f'    UniProt fetch failed for {accession}: {e}')
    return None


def fetch_af2_data(accession, version=6):
    """Fetch AF2 pLDDT and PAE from EBI AlphaFold DB."""
    result = {'plddt_array': None, 'pae_matrix': None, 'iptm': None, 'ptm': None}

    # Try mmCIF for per-residue pLDDT
    try:
        cif_url = f'https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{version}.cif'
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
                        # B-factor / 100 = pLDDT
                        try:
                            bfactor = float(parts[14])
                            plddt_vals.append(bfactor / 100.0)
                        except ValueError:
                            continue
                elif in_atom_site and not line.startswith('ATOM'):
                    break
            if plddt_vals:
                result['plddt_array'] = plddt_vals
    except Exception as e:
        pass

    # Try PAE JSON
    try:
        pae_url = f'https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-predicted_aligned_error_v{version}.json'
        resp = resilient_get(pae_url)
        if resp.status_code == 200:
            pae_data = resp.json()
            pae_list = pae_data[0]['predicted_aligned_error'] if isinstance(pae_data, list) else pae_data.get('predicted_aligned_error')
            if pae_list:
                result['pae_matrix'] = pae_list
                # Compute pTM from PAE
                result['ptm'] = _compute_ptm_from_pae(pae_list)
    except Exception as e:
        pass

    # ipTM from summary (approximate with pTM)
    result['iptm'] = result['ptm']
    return result


def _compute_ptm_from_pae(pae):
    """Compute pTM from PAE matrix."""
    pa = np.array(pae)
    L = pa.shape[0]
    d0 = max(1.24 * (L - 15) ** (1.0 / 3) - 1.8, 0.5)
    tm_scores = 1.0 / (1.0 + (pa / d0) ** 2)
    return float(np.max(np.mean(tm_scores, axis=1)))


def preprocess_for_bfn(sequence):
    """Create BFN-compatible batch from sequence using the standard pipeline."""
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.transforms import get_transform
    from antibodydesignbfn.utils.train import recursive_to

    # For IDPs, we don't have experimental structures — use AF2 predicted structure
    # We need to create a minimal structure. Instead, we download the PDB from AFDB.
    return None  # Deferred — done after AF2 data fetch


def save_lmdb(entries, db_path):
    """Save entries to LMDB."""
    os.makedirs(db_path, exist_ok=True)
    try:
        import lmdb
        map_size = max(len(pickle.dumps(entries[0])) * len(entries) * 3 + 10 * 1024 * 1024,
                       10 * 1024 * 1024)
        env = lmdb.open(db_path, map_size=map_size)
        with env.begin(write=True) as txn:
            for i, entry in enumerate(entries):
                txn.put(f'{i:08d}'.encode(), pickle.dumps(entry))
            txn.put(b'__len__', pickle.dumps(len(entries)))
        env.close()
        print(f'  Saved {len(entries)} entries to LMDB: {db_path}')
    except ImportError:
        meta = {'n_entries': len(entries)}
        with open(os.path.join(db_path, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        for i, entry in enumerate(entries):
            with open(os.path.join(db_path, f'{i:08d}.pkl'), 'wb') as f:
                pickle.dump(entry, f)
        print(f'  Saved {len(entries)} entries to pickle dir: {db_path}')


# ── Main ──
if __name__ == '__main__':
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect IDP accessions
    accessions_set = set()

    # 1. Well-known IDPs
    for acc, gene, desc in KNOWN_IDPS:
        accessions_set.add(acc)
    print(f'Literature-curated IDPs: {len(accessions_set)}')

    # 2. Try DisProt API
    disprot_accs = fetch_disprot_accessions()
    accessions_set.update(disprot_accs)
    print(f'Total unique IDP accessions: {len(accessions_set)}')

    # Limit
    all_accessions = sorted(accessions_set)[:args.max_proteins]

    entries = []
    n_processed = 0
    n_skipped = 0

    for i, acc in enumerate(all_accessions):
        print(f'[{i+1}/{len(all_accessions)}] {acc}: ', end='', flush=True)

        # Fetch sequence from UniProt
        seq = fetch_uniprot_sequence(acc)
        if not seq:
            print('FAILED (no sequence)')
            continue

        n_res = len(seq)
        if n_res < args.min_len:
            print(f'SKIPPED (too short: {n_res}aa)')
            n_skipped += 1
            continue
        if n_res > args.max_len:
            print(f'SKIPPED (too long: {n_res}aa)')
            n_skipped += 1
            continue

        # Fetch AF2 data
        af2 = fetch_af2_data(acc)
        plddt = af2.get('plddt_array')
        if not plddt:
            print(f'SKIPPED (no AF2 data)')
            n_skipped += 1
            continue

        avg_plddt = np.mean(plddt)
        # IDPs typically have low pLDDT — that's expected
        if avg_plddt < 0.3:
            print(f'{n_res}aa, avg_pLDDT={avg_plddt:.3f} ★ DISORDERED')
        else:
            print(f'{n_res}aa, avg_pLDDT={avg_plddt:.3f}')

        entry = {
            'pdb_id': acc,
            'sequence': seq,
            'length': n_res,
            'af2_plddt': plddt,
            'af2_iptm': af2.get('iptm'),
            'af2_pae_matrix': af2.get('pae_matrix'),
            'af2_ptm': af2.get('ptm'),
            'is_idp': True,
            'source': 'IDP_database',
        }
        entries.append(entry)
        n_processed += 1

    print(f'\nProcessed: {n_processed} | Skipped: {n_skipped}')

    if entries:
        # Save summary
        summary = {
            'n_entries': len(entries),
            'source': 'DisProt + literature-curated IDPs',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(output_dir / 'idp_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save as pickle dataset (not LMDB with batches — these are metadata entries)
        save_lmdb(entries, str(output_dir))

        # Print average pLDDT (should be low for IDPs)
        avg_plddts = [np.mean(e['af2_plddt']) for e in entries if e['af2_plddt']]
        if avg_plddts:
            print(f'Average AF2 pLDDT across {len(avg_plddts)} IDPs: {np.mean(avg_plddts):.3f}')
            print(f'  Min: {np.min(avg_plddts):.3f} | Max: {np.max(avg_plddts):.3f}')
            print(f'  (Lower pLDDT = more disordered; typical folded protein > 0.7)')

    print(f'\nDone! Results saved to {output_dir}')
