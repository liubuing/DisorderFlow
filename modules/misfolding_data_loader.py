#!/usr/bin/env python
"""Data loader for misfolding disease targets.

Handles:
 - Downloading PDB structures from RCSB and EBI AlphaFold DB
 - Extracting structured regions from full-length proteins
 - Local caching and validation
 - Preparing target structures for the misfolding pipeline

Usage:
  from misfolding_data_loader import prepare_misfolding_target, fetch_alphafold_structure
"""

import sys
import os
import time
import json
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / 'data' / 'misfolding_targets'
CACHE_FILE = DATA_DIR / 'target_cache.json'

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache():
    """Load the local target cache."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache):
    """Save the local target cache."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


def fetch_alphafold_structure(uniprot_id, output_dir=None, version=4):
    """Download AF2-predicted PDB from EBI AlphaFold DB.

    Args:
        uniprot_id: UniProt accession (e.g. 'P05067')
        output_dir: output directory (defaults to data/misfolding_targets/)
        version: AFDB version (4 or 6)

    Returns:
        Path to downloaded PDB file, or None on failure.
    """
    if output_dir is None:
        output_dir = DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_name = f'AF-{uniprot_id}-F1-model_v{version}.pdb'
    pdb_path = output_dir / pdb_name

    if pdb_path.exists() and pdb_path.stat().st_size > 1000:
        return pdb_path

    url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb'
    try:
        urllib.request.urlretrieve(url, str(pdb_path))
        size = pdb_path.stat().st_size
        if size < 1000:
            pdb_path.unlink()
            return None
        return pdb_path
    except Exception:
        # Try v4 fallback
        if version != 4:
            return fetch_alphafold_structure(uniprot_id, output_dir, version=4)
        return None


def fetch_rcsb_structure(pdb_id, output_dir=None):
    """Download experimental PDB structure from RCSB.

    Args:
        pdb_id: PDB identifier (e.g. '2NAO')
        output_dir: output directory

    Returns:
        Path to downloaded PDB file, or None on failure.
    """
    if output_dir is None:
        output_dir = DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_name = f'{pdb_id}.pdb'
    pdb_path = output_dir / pdb_name

    if pdb_path.exists() and pdb_path.stat().st_size > 1000:
        return pdb_path

    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    try:
        urllib.request.urlretrieve(url, str(pdb_path))
        size = pdb_path.stat().st_size
        if size < 1000:
            pdb_path.unlink()
            return None
        return pdb_path
    except Exception:
        return None


def is_nmr_structure(pdb_path):
    """Check if a PDB file is an NMR ensemble (multiple MODEL records)."""
    try:
        with open(pdb_path, 'r') as f:
            model_count = 0
            for line in f:
                if line.startswith('MODEL '):
                    model_count += 1
                    if model_count > 1:
                        return True
                if model_count > 1:
                    break
            return False
    except (IOError, OSError):
        return False


def extract_first_model_from_nmr(pdb_path, output_path=None):
    """Extract the first model from an NMR ensemble PDB, writing a single-model PDB.

    Returns the path to the extracted file.
    """
    if output_path is None:
        p = Path(pdb_path)
        output_path = p.parent / f'{p.stem}_model1.pdb'

    with open(pdb_path, 'r') as fin, open(output_path, 'w') as fout:
        in_model = False
        for line in fin:
            if line.startswith('MODEL '):
                model_num = int(line.split()[1])
                if model_num == 1:
                    in_model = True
                else:
                    break
            elif line.startswith('ENDMDL'):
                in_model = False
                break
            elif in_model:
                fout.write(line)
    return output_path


def extract_region_from_pdb(pdb_path, chain_id, start_res, end_res, output_path):
    """Extract a residue range from a PDB file into a new PDB file.

    Uses only ATOM/HETATM records matching the chain and residue range.
    Keeps MODEL/ENDMDL records for multi-model files.
    """
    from pathlib import Path
    output_path = Path(output_path)

    with open(pdb_path, 'r') as fin, open(output_path, 'w') as fout:
        in_region = False
        for line in fin:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                fout.write(line)
                continue
            try:
                line_chain = line[21].strip()
                resseq = int(line[22:26].strip())
            except (ValueError, IndexError):
                continue
            if line_chain == chain_id and start_res <= resseq <= end_res:
                fout.write(line)

    return output_path


def prepare_misfolding_target(disease_key, target_data, conformation_key=None):
    """Download and prepare a misfolding disease target for antibody design.

    Args:
        disease_key: key in MISFOLDING_TARGETS
        target_data: the target dict from misfolding_knowledge_base
        conformation_key: optional specific conformation key (defaults to first)

    Returns:
        dict with keys:
            - pdb_path: Path to local cleaned PDB file
            - chain_id: Chain to use
            - target_name: Human-readable name
            - idp_warning: warning string or None
            - design_note: design guidance
            - resolution: structure resolution
            - success: bool
            - error: error message if failed
    """
    cache = _load_cache()
    cache_key = f'{disease_key}_{conformation_key or "default"}'

    # Check cache first (valid for 7 days)
    if cache_key in cache:
        entry = cache[cache_key]
        pdb_path = Path(entry['pdb_path'])
        if pdb_path.exists():
            return {
                'pdb_path': str(pdb_path),
                'chain_id': entry.get('chain_id', 'A'),
                'target_name': target_data.get('disease_cn', disease_key),
                'idp_warning': target_data.get('idp_warning'),
                'design_note': target_data.get('design_note', ''),
                'resolution': entry.get('resolution', 'unknown'),
                'success': True,
                'error': None,
            }

    # Determine which conformation to use
    confs = target_data.get('conformations', {})
    if not confs:
        return {
            'pdb_path': None, 'chain_id': None,
            'target_name': target_data.get('disease_cn', disease_key),
            'idp_warning': None, 'design_note': '', 'resolution': None,
            'success': False, 'error': 'No conformations defined for this target.',
        }

    if conformation_key is None:
        conformation_key = list(confs.keys())[0]

    conf = confs.get(conformation_key)
    if conf is None:
        return {
            'pdb_path': None, 'chain_id': None,
            'target_name': target_data.get('disease_cn', disease_key),
            'idp_warning': None, 'design_note': '', 'resolution': None,
            'success': False,
            'error': f'Conformation "{conformation_key}" not found.',
        }

    pdb_id = conf['pdb_id']
    chain = conf.get('chain', 'A')
    region = conf.get('region')  # (start, end) or None
    resolution = conf.get('resolution', 'unknown')

    # Download PDB
    pdb_path = fetch_rcsb_structure(pdb_id)
    if pdb_path is None:
        # Try AlphaFold DB as fallback
        uniprot = target_data.get('uniprot_id')
        if uniprot:
            pdb_path = fetch_alphafold_structure(uniprot)
        if pdb_path is None:
            return {
                'pdb_path': None, 'chain_id': chain,
                'target_name': target_data.get('disease_cn', disease_key),
                'idp_warning': None, 'design_note': '', 'resolution': resolution,
                'success': False,
                'error': f'Failed to download PDB {pdb_id} from RCSB.',
            }

    # Handle NMR ensembles
    if is_nmr_structure(pdb_path):
        pdb_path = extract_first_model_from_nmr(pdb_path)

    # Extract structured region if specified
    if region is not None:
        extracted_path = (
            Path(pdb_path).parent
            / f'{Path(pdb_path).stem}_{chain}_{region[0]}-{region[1]}.pdb'
        )
        if not extracted_path.exists():
            extract_region_from_pdb(pdb_path, chain, region[0], region[1],
                                    extracted_path)
        pdb_path = extracted_path

    # Update cache
    cache[cache_key] = {
        'pdb_path': str(pdb_path),
        'chain_id': chain,
        'disease_key': disease_key,
        'conformation_key': conformation_key,
        'pdb_id': pdb_id,
        'resolution': resolution,
        'timestamp': time.time(),
    }
    _save_cache(cache)

    return {
        'pdb_path': str(pdb_path),
        'chain_id': chain,
        'target_name': target_data.get('disease_cn', disease_key),
        'idp_warning': target_data.get('idp_warning'),
        'design_note': target_data.get('design_note', ''),
        'resolution': resolution,
        'success': True,
        'error': None,
    }


def batch_prepare_all_targets(targets_dict, max_per_disease=3):
    """Pre-download all target structures. Call once offline.

    Args:
        targets_dict: MISFOLDING_TARGETS dict from misfolding_knowledge_base
        max_per_disease: max conformations per target

    Returns:
        dict of {disease_key: [result_dict, ...]}
    """
    results = {}
    for key, target in targets_dict.items():
        results[key] = []
        confs = list(target.get('conformations', {}).keys())[:max_per_disease]
        for conf_key in confs:
            result = prepare_misfolding_target(key, target, conf_key)
            results[key].append(result)
            if result['success']:
                print(f'  [{key}/{conf_key}] OK: {result["pdb_path"]}')
            else:
                print(f'  [{key}/{conf_key}] FAIL: {result["error"]}')
            time.sleep(0.3)
    return results
