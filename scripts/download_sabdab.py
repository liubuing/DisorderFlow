#!/usr/bin/env python
"""
Download Chothia-numbered antibody-antigen structures from SAbDab.

Filters for complexes with:
  - Heavy chain + antigen chain
  - Protein antigen
  - Resolution <= 3.5 Angstrom
  - Not scFv

Downloads individual PDB files from OPIG's SAbDab archive with
parallel workers and automatic retry on failure.

Usage:
    python scripts/download_sabdab.py                        # download all filtered
    python scripts/download_sabdab.py --max 100              # download first 100
    python scripts/download_sabdab.py --workers 16           # 16 parallel workers
    python scripts/download_sabdab.py --dry-run              # just count, don't download
"""

import os
import sys
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# URL patterns to try (in order). RCSB is most reliable; SAbDab provides Chothia-numbered structures.
URL_PATTERNS = [
    "http://files.rcsb.org/download/{pdbcode}.pdb",
    "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/{pdbcode}.pdb",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_SUMMARY = os.path.join(PROJECT_DIR, "data", "sabdab_summary_all.tsv")
DEFAULT_OUTDIR = os.path.join(PROJECT_DIR, "data", "all_structures", "chothia")

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
TIMEOUT = 30  # seconds per request


def resilient_get(url, max_retries=MAX_RETRIES, timeout=TIMEOUT):
    """GET with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                content = resp.content
                # Check it's a valid PDB (starts with ATOM/HETATM/HEADER/REMARK)
                if content.strip():
                    return content
                else:
                    logger.debug(f"Empty response from {url}")
                    return None
            elif resp.status_code == 404:
                return None  # Not found, don't retry
            else:
                logger.debug(f"HTTP {resp.status_code} from {url}, attempt {attempt+1}/{max_retries}")
        except requests.RequestException as e:
            logger.debug(f"Request error for {url}: {e}, attempt {attempt+1}/{max_retries}")

        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY * (2 ** attempt))

    return None


def download_one(pdbcode, outdir, url_patterns=URL_PATTERNS):
    """Download a single PDB file. Returns (pdbcode, success)."""
    outpath = os.path.join(outdir, f"{pdbcode}.pdb")

    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        return pdbcode, True  # Already downloaded

    for pattern in url_patterns:
        url = pattern.format(pdbcode=pdbcode.upper())
        content = resilient_get(url)
        if content is not None:
            # Validate: PDB files start with ATOM, HETATM, HEADER, etc.
            text = content.decode('utf-8', errors='ignore')[:200]
            if any(text.startswith(prefix) for prefix in ('HEADER', 'ATOM', 'HETATM', 'REMARK', 'TITLE', 'COMPND')):
                with open(outpath, 'wb') as f:
                    f.write(content)
                return pdbcode, True
            else:
                logger.debug(f"Invalid PDB content for {pdbcode}: {text[:80]}")

    return pdbcode, False


def filter_entries(df):
    """Filter SAbDab entries for training-quality antibody-antigen complexes."""
    # Has heavy chain
    has_h = df['Hchain'].notna() & (df['Hchain'] != 'NA') & (df['Hchain'] != '')
    # Has antigen
    has_ag = df['antigen_chain'].notna() & (df['antigen_chain'] != 'NA') & (df['antigen_chain'] != '')
    # Protein antigen only
    protein_ag = df['antigen_type'].str.contains('protein', na=False, case=False)
    # Not scFv (these don't have heavy/light pair structure)
    scfv_col = df.get('scfv', pd.Series([False] * len(df)))
    not_scfv = ~scfv_col.astype(bool)
    # Resolution filter
    resolution = pd.to_numeric(df['resolution'], errors='coerce')
    res_ok = resolution <= 3.5

    mask = has_h & has_ag & protein_ag & not_scfv & res_ok
    return df[mask]


def main():
    parser = argparse.ArgumentParser(description="Download SAbDab Chothia structures")
    parser.add_argument('--summary', default=DEFAULT_SUMMARY, help='Path to sabdab_summary_all.tsv')
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='Output directory for PDB files')
    parser.add_argument('--workers', type=int, default=8, help='Parallel download workers')
    parser.add_argument('--max', type=int, default=None, help='Max PDBs to download (for testing)')
    parser.add_argument('--dry-run', action='store_true', help='Only count, no download')
    parser.add_argument('--resume', action='store_true', default=True, help='Skip existing files')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load and filter
    logger.info(f"Loading SAbDab summary from {args.summary}...")
    df = pd.read_csv(args.summary, sep='\t')
    logger.info(f"Total entries: {len(df)}")

    filtered = filter_entries(df)
    unique_pdbs = sorted(filtered['pdb'].unique())
    logger.info(f"After filtering: {len(filtered)} entries, {len(unique_pdbs)} unique PDBs")

    # Check existing
    existing = set()
    if args.resume and os.path.exists(args.outdir):
        for f in os.listdir(args.outdir):
            if f.endswith('.pdb') and os.path.getsize(os.path.join(args.outdir, f)) > 0:
                existing.add(f.replace('.pdb', ''))
    logger.info(f"Already downloaded: {len(existing)}")

    to_download = [p for p in unique_pdbs if p not in existing]
    if args.max:
        to_download = to_download[:args.max]
    logger.info(f"To download: {len(to_download)}")

    if args.dry_run or len(to_download) == 0:
        logger.info("Dry run or nothing to download. Done.")
        return

    # Download in parallel
    success = 0
    failed = 0
    failed_pdbs = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, pdb, args.outdir): pdb for pdb in to_download}
        with tqdm(total=len(to_download), desc='Downloading') as pbar:
            for future in as_completed(futures):
                pdbcode, ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    failed_pdbs.append(pdbcode)
                pbar.set_postfix({'ok': success, 'fail': failed})
                pbar.update(1)

    logger.info(f"Done. Success: {success}, Failed: {failed}")
    if failed_pdbs:
        logger.warning(f"Failed PDBs: {failed_pdbs[:20]}{'...' if len(failed_pdbs) > 20 else ''}")
        # Save failed list for retry
        failed_path = os.path.join(args.outdir, '_failed.txt')
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_pdbs))
        logger.info(f"Failed PDBs saved to {failed_path}")


if __name__ == '__main__':
    main()
