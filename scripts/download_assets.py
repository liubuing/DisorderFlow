#!/usr/bin/env python3
"""Download external assets required by DisorderFlow.

Usage:
    python scripts/download_assets.py [--all | --mpnn | --hdock | --checkpoints]

Assets downloaded:
  - ProteinMPNN model weights (from GitHub)
  - HDOCKlite v1.1 (from Huang Lab)
  - BFN checkpoints (from Hugging Face)
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress indication."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {desc or dest.name} already exists")
        return
    print(f"  [download] {desc or dest.name}")
    print(f"             {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  [done] saved to {dest}")


def setup_proteinmpnn():
    """Clone ProteinMPNN repo and keep only inference code (no weights in git)."""
    mpnn_dir = ROOT / "ProteinMPNN"
    print("\n=== ProteinMPNN ===")

    if (mpnn_dir / "protein_mpnn_utils.py").exists():
        print("  [skip] ProteinMPNN code already present")
    else:
        print("  [clone] dauparas/ProteinMPNN ...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/dauparas/ProteinMPNN.git", str(mpnn_dir)],
            check=True,
        )

    # Download vanilla weights
    weights_dir = mpnn_dir / "vanilla_model_weights"
    v_48_020 = weights_dir / "v_48_020.pt"
    download_file(
        "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt",
        v_48_020,
        "vanilla v_48_020.pt",
    )

    # Download CA-only weights
    ca_dir = mpnn_dir / "ca_model_weights"
    ca_v_48_020 = ca_dir / "v_48_020.pt"
    download_file(
        "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/ca_model_weights/v_48_020.pt",
        ca_v_48_020,
        "CA-only v_48_020.pt",
    )

    # Download soluble weights
    sol_dir = mpnn_dir / "soluble_model_weights"
    sol_v_48_020 = sol_dir / "v_48_020.pt"
    download_file(
        "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/soluble_model_weights/v_48_020.pt",
        sol_v_48_020,
        "soluble v_48_020.pt",
    )
    print("  [ok] ProteinMPNN ready")


def setup_hdock():
    """Download HDOCKlite v1.1 for Linux."""
    print("\n=== HDOCKlite v1.1 ===")
    hdock_dir = ROOT / "HDOCKlite-v1.1"

    if (hdock_dir / "hdock").exists():
        print("  [skip] HDOCKlite already present")
        return

    tarball = ROOT / "HDOCKlite.tar.gz"
    download_file(
        "http://huanglab.phys.hust.edu.cn/software/hdocklite/HDOCKlite-v1.1.tar.gz",
        tarball,
        "HDOCKlite-v1.1.tar.gz",
    )

    print("  [extract] ...")
    import tarfile
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(ROOT)
    print("  [ok] HDOCKlite ready (Linux only)")


def setup_checkpoints():
    """Download BFN model checkpoints from Hugging Face."""
    print("\n=== BFN Checkpoints (Hugging Face) ===")
    ckpt_dir = ROOT / "logs" / "pretrained"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Primary design checkpoint
    best_pt = ckpt_dir / "best.pt"
    download_file(
        "https://huggingface.co/YueHuLab/disorderflow/resolve/main/best.pt",
        best_pt,
        "BFN design checkpoint (best.pt)",
    )
    print("  [ok] Checkpoints ready")
    print(f"  NOTE: Update app_config.yaml -> models.bfn.checkpoint to: {best_pt.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Download DisorderFlow external assets")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--mpnn", action="store_true", help="Download ProteinMPNN")
    parser.add_argument("--hdock", action="store_true", help="Download HDOCKlite")
    parser.add_argument("--checkpoints", action="store_true", help="Download BFN checkpoints")
    args = parser.parse_args()

    if not any(vars(args).values()):
        args.all = True

    print("DisorderFlow Asset Downloader")
    print(f"Project root: {ROOT}")

    if args.all or args.mpnn:
        setup_proteinmpnn()
    if args.all or args.hdock:
        setup_hdock()
    if args.all or args.checkpoints:
        setup_checkpoints()

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
