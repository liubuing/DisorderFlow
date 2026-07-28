#!/usr/bin/env python3
"""Upload all DisorderFlow datasets to Hugging Face.

Usage:
    python scripts/upload_to_hf.py

Uploads data/ directory to liubuing/disorderflow (dataset repo).
Large files (>10MB) are automatically tracked with Git LFS.
Supports resume if interrupted.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from huggingface_hub import HfApi

REPO_ID = "liubuing/disorderflow"
REPO_TYPE = "dataset"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# File patterns to track with LFS (large binary/data files)
LFS_PATTERNS = [
    "*.mdb",
    "*.lmdb",
    "*.pt",
    "*.pth",
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.tar.gz",
    "*.zip",
    "*.h5",
    "*.hdf5",
    "*.pdb",
    "*.cif",
    "*.tsv",
    "*.json",
]


def main():
    api = HfApi()

    # Verify auth
    try:
        user = api.whoami()["name"]
        print(f"[auth] Logged in as: {user}")
    except Exception as e:
        print(f"[error] Not authenticated: {e}")
        print("Run: hf auth login")
        sys.exit(1)

    # Verify repo exists
    try:
        api.repo_info(REPO_ID, repo_type=REPO_TYPE)
        print(f"[repo] Target: {REPO_ID} ({REPO_TYPE})")
    except Exception:
        print(f"[repo] Creating {REPO_ID} ...")
        api.create_repo(REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)

    # Set LFS tracking patterns
    print("[lfs] Setting LFS patterns...")
    for pattern in LFS_PATTERNS:
        try:
            api.preupload_lfs_files(REPO_ID, [pattern], repo_type=REPO_TYPE)
        except Exception:
            pass  # Pattern may already be set

    # Upload
    print(f"[upload] Source: {DATA_DIR}")
    print(f"[upload] This will take a long time for ~220GB of data.")
    print(f"[upload] Progress will be shown below. Ctrl+C to pause (resumable).\n")

    start = time.time()
    try:
        api.upload_folder(
            folder_path=str(DATA_DIR),
            path_in_repo="data",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\n[paused] Interrupted after {elapsed/60:.1f} min. Re-run to resume.")
        sys.exit(0)
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[error] {e}")
        print(f"[info] Elapsed: {elapsed/60:.1f} min. Re-run to resume from where it stopped.")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n[done] Upload complete in {elapsed/60:.1f} min")
    print(f"[done] View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
