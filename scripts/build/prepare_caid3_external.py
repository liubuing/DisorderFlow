#!/usr/bin/env python
"""Download and freeze official CAID3 reference files before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://caid.idpcentral.org/assets/sections/challenge/static/references/3"
REFERENCES = (
    "disorder_nox.fasta",
    "disorder_pdb.fasta",
    "binding.fasta",
    "binding_idr.fasta",
    "linker.fasta",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def count_records(data):
    return sum(line.startswith(b">") for line in data.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "data/caid3/frozen_raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Frozen manifest already exists: {manifest_path}")

    files = []
    for name in REFERENCES:
        url = f"{BASE_URL}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if not data.startswith(b">"):
            raise ValueError(f"Unexpected CAID3 payload: {url}")
        path = out_dir / name
        path.write_bytes(data)
        files.append(
            {
                "name": name,
                "url": url,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
                "records": count_records(data),
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "frozen_raw_before_label_analysis",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "method_frozen_before_download": True,
        "checkpoint_sha256": "ed9984504d799187b7660c1d40c3dc68c01a0ce9b3cf92e25af1273195931252",
        "calibration_sha256": "166cb02ae8eca4da4d1f041b9e2b0020457188a1a7b6e0141041213985d1da5a",
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
