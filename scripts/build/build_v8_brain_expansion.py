#!/usr/bin/env python
"""
Build V8 brain disease expansion dataset with ipTM > 0.6 quality filter.

Key improvements over V6:
  1. ipTM > 0.6 quality filter applied at download time
  2. Targets underrepresented categories (< 30 entries in V6)
  3. Adds new disease areas: neuroinflammation, repeat expansion disorders,
     cerebral small vessel disease, brain iron accumulation
  4. Uses AFDB v7 (newer AF3 models where available)
"""
import sys, os, json, pickle, time, io, tempfile
from pathlib import Path

if sys.platform == 'win32':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import numpy as np
import torch

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from build_utils import (AA3_TO_1, AA_LETTERS, compute_ptm_from_pae,
                         resilient_get, load_pdb_sequence, save_lmdb)

AFDB_BASE = 'https://alphafold.ebi.ac.uk/files'
UNIPROT_API = 'https://rest.uniprot.org/uniprotkb/search'

# V8 expansion: target underrepresented brain disease categories + new areas
V8_BRAIN_QUERIES = [
    # === Deeper coverage of underrepresented categories from V6 (< 30 entries) ===
    ('v8_prion_deep', '("prion" OR "PRNP" OR "PRND" OR "PRNDP" OR "SPRN" OR "shadow of prion" OR "Doppel") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_channelopathy_deep', '("ion channel" OR "channelopathy" OR "CACNA1A" OR "CACNA1B" OR "CACNA1G" OR "KCNA2" OR "KCNC1" OR "KCNB1" OR "KCND3" OR "KCNMA1" OR "SCN1B" OR "SCN9A" OR "CACNB4" OR "KCNA1" OR "CACNA1H") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_msa_dlb_deep', '("multiple system atrophy" OR "dementia with Lewy bodies" OR "DLB" OR "alpha synuclein" OR "GBA" OR "LRRK2" OR "SNCAIP" OR "synphilin" OR "PARK7" OR "UCHL1" OR "ATP13A2" OR "PLA2G6") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_rett_deep', '("Rett syndrome" OR "MECP2" OR "CDKL5" OR "FOXG1" OR "MEF2C" OR "TCF4" OR "Pitt Hopkins" OR "Angelman" OR "UBE3A" OR "GABRA5") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_depression_deep', '("major depressive disorder" OR "depression" OR "SLC6A4" OR "HTR1A" OR "HTR2A" OR "BDNF" OR "FKBP5" OR "CRHR1" OR "TPH2" OR "GRIA1" OR "GRIN2B" OR "DRD4") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_bipolar_deep', '("bipolar disorder" OR "manic depressive" OR "ANK3" OR "CACNA1C" OR "SYNE1" OR "TRANK1" OR "NCAN" OR "CACNB2" OR "GRM3" OR "ITIH4" OR "PBRM1") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_narcolepsy_deep', '("narcolepsy" OR "hypocretin" OR "HCRT" OR "HCRTR1" OR "HCRTR2" OR "orexin" OR "cataplexy" OR "TCRA" OR "P2RY11" OR "TNFSF4") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_myasthenia_deep', '("myasthenia gravis" OR "congenital myasthenic syndrome" OR "CHRNE" OR "CHRNA1" OR "CHRNB1" OR "CHRND" OR "RAPSN" OR "DOK7" OR "MUSK" OR "LRP4" OR "AGRN" OR "GFPT1") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    # === New brain disease categories for V8 ===
    ('v8_neuroinflammation', '("neuroinflammation" OR "microglia" OR "TREM2" OR "TYROBP" OR "CD33" OR "TLR4" OR "NLRP3" OR "inflammasome" OR "CSF1R" OR "CX3CR1" OR "ITGAM" OR "P2RY12" OR "neuroinflammatory") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_repeat_expansion', '("repeat expansion" OR "C9orf72" OR "FMR1" OR "ATXN1" OR "ATXN2" OR "ATXN3" OR "ATXN7" OR "ATXN10" OR "ATN1" OR "TBP" OR "CACNA1A" OR "PPP2R2B" OR "NOP56" OR "DMPK" OR "CNBP" OR "C9ORF72" OR "RFC1" OR "CANVAS") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_small_vessel_disease', '("cerebral small vessel disease" OR "CADASIL" OR "NOTCH3" OR "HTRA1" OR "COL4A1" OR "COL4A2" OR "CARASIL" OR "FOXC1" OR "PITX2" OR "TREX1" OR "RNASEH2A") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_brain_iron', '("neurodegeneration with brain iron accumulation" OR "NBIA" OR "PANK2" OR "PLA2G6" OR "FA2H" OR "C19orf12" OR "WDR45" OR "COASY" OR "ATP13A2" OR "DCAF17" OR "CP" OR "FTL" OR "ferritinopathy") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_cerebral_palsy', '("cerebral palsy" OR "neonatal encephalopathy" OR "hypoxic ischemic encephalopathy" OR "KCNQ2" OR "SCN2A" OR "GNAO1" OR "STXBP1" OR "KCNT1" OR "SLC2A1" OR "neonatal epilepsy") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_leukodystrophy', '("leukodystrophy" OR "leukodystrophies" OR "adrenoleukodystrophy" OR "ABCD1" OR "GALC" OR "Krabbe" OR "ARSA" OR "GFAP" OR "Alexander disease" OR "EIF2B1" OR "EIF2B2" OR "EIF2B3" OR "EIF2B4" OR "EIF2B5" OR "vanishing white matter") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    # === Protein aggregation / misfolding (cross-cutting) ===
    ('v8_amyloidosis_neuro', '("cerebral amyloid" OR "amyloid beta" OR "APP" OR "BACE1" OR "BACE2" OR "PSEN1" OR "PSEN2" OR "ADAM10" OR "clusterin" OR "CLU" OR "PICALM" OR "BIN1" OR "CD2AP" OR "EPHA1" OR "CR1" OR "SORL1" OR "ABCA7" OR "TREM2" OR "PLD3") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    ('v8_tauopathy', '("tauopathy" OR "MAPT" OR "tau protein" OR "GSK3B" OR "CDK5" OR "PP2A" OR "PIN1" OR "FKBP4" OR "FKBP5" OR "CHIP" OR "STUB1" OR "HSP90" OR "BAG2" OR "progressive supranuclear palsy" OR "corticobasal degeneration" OR "Pick disease") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),

    # === Cerebrovascular ===
    ('v8_cerebrovascular', '("stroke" OR "cerebrovascular" OR "ischemic stroke" OR "hemorrhagic stroke" OR "NOTCH3" OR "COL4A1" OR "HTRA1" OR "FOXF2" OR "ZFHX3" OR "HDAC9" OR "PITX2") AND (organism_id:9606) AND (length:[50 TO 500]) AND (reviewed:true)'),
]


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
    seq = ''.join(AA_LETTERS[int(a)] if 0 <= int(a) < 20 else 'X' for a in aa_indices.cpu())
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
    p = argparse.ArgumentParser(description='Build V8 brain disease expansion dataset')
    p.add_argument('--output_dir', default=str(PROJECT_DIR / 'data' / 'confidence_v8_brain_expansion'),
                   help='Output LMDB directory')
    p.add_argument('--max_per_query', type=int, default=80, help='Max proteins per query')
    p.add_argument('--af2_version', type=int, default=7, help='AFDB version (7 = AF3)')
    p.add_argument('--iptm_threshold', type=float, default=0.6,
                   help='Minimum ipTM for quality filtering')
    p.add_argument('--resume', action='store_true', help='Resume from existing LMDB')
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    processed_ids = set()
    entries = []
    if args.resume:
        import lmdb as _lmdb
        db_path = str(output_dir / 'v8_brain_expansion.lmdb')
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

    for category, query in V8_BRAIN_QUERIES:
        print(f'\n{"="*60}')
        print(f'  {category}: searching UniProt...')
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
    print(f'  Quality threshold: ipTM > {args.iptm_threshold}')
    print(f'{"="*60}\n')

    if not all_accessions:
        print('All accessions already processed!')
        sys.exit(0)

    n_failed = 0
    n_low_quality = 0

    for i, (acc, name, category) in enumerate(all_accessions):
        print(f'[{i+1}/{len(all_accessions)}] [{category}] {acc}: {name[:60]}...', end=' ', flush=True)

        # Download mmCIF + PAE with version fallback (v7 -> v6 -> v5 -> v4)
        cif_text = None
        pae_matrix = None
        for v_fallback in [args.af2_version, 6, 5, 4]:
            url = f'{AFDB_BASE}/AF-{acc}-F1-model_v{v_fallback}.cif'
            r = resilient_get(url, timeout=60)
            if r.status_code != 200:
                continue
            cif_text = r.text
            pae_url = f'{AFDB_BASE}/AF-{acc}-F1-predicted_aligned_error_v{v_fallback}.json'
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
            if pae_matrix is not None:
                break

        if cif_text is None or pae_matrix is None:
            print('SKIP (no AF2 structure/PAE)')
            n_failed += 1
            continue

        pae_t = torch.tensor(pae_matrix, dtype=torch.float32)
        if pae_t.dim() != 2:
            print('SKIP (bad PAE)')
            n_failed += 1
            continue

        iptm = compute_ptm_from_pae(pae_t.cpu().numpy())

        if iptm < args.iptm_threshold:
            print(f'SKIP (ipTM={iptm:.3f} < {args.iptm_threshold})')
            n_low_quality += 1
            continue

        # Passed quality check — now do full processing
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

        if pae_t.shape[0] < n_res:
            pae_t = torch.nn.functional.pad(pae_t, (0, n_res - pae_t.shape[0], 0, n_res - pae_t.shape[1]))
        pae_t = pae_t[:n_res, :n_res]

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
            'af2_iptm': torch.tensor(iptm, dtype=torch.float32),
            'af2_pae_matrix': pae_t.cpu(),
            'category': category,
            'source': 'v8_brain_expansion',
        }
        entries.append(entry)
        plddt_mean = plddt_tensor.mean().item()
        print(f'OK (L={n_res}, pLDDT={plddt_mean:.3f}, ipTM={iptm:.3f})')

        if len(entries) % 30 == 0:
            save_lmdb(str(output_dir / 'v8_brain_expansion.lmdb'), entries)
            print(f'  [saved {len(entries)} entries]')

    print(f'\n{"="*60}')
    print(f'  Results: {len(entries)} succeeded')
    print(f'  Failed: {n_failed}')
    print(f'  Low quality (ipTM < {args.iptm_threshold}): {n_low_quality}')
    print(f'{"="*60}')

    if not entries:
        print('No entries passed quality filter! Try lowering --iptm_threshold.')
        sys.exit(1)

    save_lmdb(str(output_dir / 'v8_brain_expansion.lmdb'), entries)

    cat_counts = {}
    for e in entries:
        cat = e.get('category', 'unknown')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    summary = {
        'n_entries': len(entries),
        'n_failed': n_failed,
        'n_low_quality': n_low_quality,
        'iptm_threshold': args.iptm_threshold,
        'categories': cat_counts,
        'source': f'EBI AlphaFold DB v{args.af2_version}',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved {len(entries)} quality-filtered entries to {output_dir / "v8_brain_expansion.lmdb"}')
    print(f'Categories: {json.dumps(cat_counts, indent=2)}')
    print('Done!')


if __name__ == '__main__':
    main()
