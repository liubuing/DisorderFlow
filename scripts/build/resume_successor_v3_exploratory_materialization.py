#!/usr/bin/env python
"""Resume exploratory materialization from an interrupted local CIF directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build import materialize_successor_v3_confirmatory as materializer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--source-cif-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = ROOT / args.source_cif_dir

    def reuse_download(row, destination, retries=3):
        del retries
        source = source_dir / f"{row['instance']}.cif"
        if not source.is_file():
            raise FileNotFoundError(f"Interrupted materialization lacks {source}")
        shutil.copy2(source, destination)

    materializer.download_crop = reuse_download
    sys.argv = [
        "materialize_successor_v3_confirmatory.py",
        "--discovery", str(args.discovery),
        "--output-dir", str(args.output_dir),
    ]
    materializer.main()


if __name__ == "__main__":
    main()
