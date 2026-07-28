#!/usr/bin/env python
"""Download TfR / Transferrin reference structures for the TfR design pipeline.

Fetches:
  - 6GZV  : human TfR1 in complex with Transferrin (PDB)  → on-target epitope ref
  - 1A8E  : apo human Transferrin (PDB)                   → off-target (neg design)
  - (optional) AlphaFold DB uniprot P0DB08 (TfR1)         → full-length disordered check

Falls back gracefully (writes a README noting which downloads failed) so the
pipeline does not hard-crash when offline — callers use DEFAULT_TFR_PDB / mock.

Usage:
  python scripts/download_tfr_targets.py [--out data/tfr_targets]
  python scripts/download_tfr_targets.py --smoke   # fetch 1 small file to verify connectivity
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

# PDB IDs → target filenames. These are the canonical references cited in the
# research plan (docs/海参肽-TfR纳米抗体BBB递送系统_研究方案.md).
TARGETS = {
    "6GZV": "6GZV.pdb",   # TfR1 - Transferrin - Fab complex (on-target)
    "1A8E": "1A8E.pdb",   # apo Transferrin (off-target / negative design)
}

PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str, out_path: str, timeout: int = 60) -> bool:
    """Download a single PDB file from RCSB. Returns True on success."""
    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DisorderFlow/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000 or b"<!DOCTYPE" in data[:200]:
            print(f"  [skip] {pdb_id}: unexpected response (not a PDB file)")
            return False
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  [ok]   {pdb_id} → {out_path} ({len(data)} bytes)")
        return True
    except Exception as e:  # noqa: BLE001 — network errors are non-fatal here
        print(f"  [fail] {pdb_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download TfR/Transferrin reference PDBs")
    parser.add_argument("--out", default=os.path.join("data", "tfr_targets"),
                        help="Output directory (default: data/tfr_targets)")
    parser.add_argument("--smoke", action="store_true",
                        help="Download only 1A8E to verify connectivity, then exit")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading TfR targets to: {os.path.abspath(args.out)}")

    targets = {"1A8E": TARGETS["1A8E"]} if args.smoke else TARGETS

    successes, failures = [], []
    for pdb_id, fname in targets.items():
        out_path = os.path.join(args.out, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f"  [exists] {pdb_id} → {out_path}")
            successes.append(pdb_id)
            continue
        if download_pdb(pdb_id, out_path):
            successes.append(pdb_id)
        else:
            failures.append(pdb_id)

    # Write a status README so the pipeline can detect partial downloads.
    status_path = os.path.join(args.out, "README.txt")
    with open(status_path, "w") as f:
        f.write("TfR design target structures\n")
        f.write("=" * 40 + "\n")
        f.write(f"Downloaded:    {successes}\n")
        f.write(f"Failed/missing:{failures}\n\n")
        f.write("If downloads failed, fetch manually from RCSB:\n")
        for pdb_id, fname in TARGETS.items():
            f.write(f"  {pdb_id}: https://www.rcsb.org/structure/{pdb_id} → {fname}\n")
        f.write("\nThe TfR pipeline degrades gracefully: when a structure is "
                "missing, the negative-design step is skipped and design "
                "proceeds on-target only.\n")

    print(f"\nDone. Success: {successes}. Failed: {failures}.")
    print(f"Status: {status_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
