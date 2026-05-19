"""Upload confidence dataset to HuggingFace — direct foreground execution."""
import sys, os, json, time, shutil

# Auth token
token = os.environ.get("HF_TOKEN", "")

print("=== HF Upload Script ===")
print(f"Python: {sys.executable}")

# Step 1: Login
print("\n[1/5] Logging in...")
import huggingface_hub
huggingface_hub.login(token=token)
print(f"  Logged in as: {huggingface_hub.whoami()['name']}")

# Step 2: Delete test repo if exists
print("\n[2/5] Cleaning up test repo...")
try:
    huggingface_hub.delete_repo("liubuing/bfn-confidence-test", token=token)
    print("  Deleted test repo")
except Exception as e:
    print(f"  (test repo cleanup: {e})")

# Step 3: Create repo
print("\n[3/5] Creating repo...")
try:
    url = huggingface_hub.create_repo(
        "liubuing/bfn-confidence-general-proteins",
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )
    print(f"  Created: {url}")
except Exception as e:
    print(f"  create_repo error: {e}")

# Step 4: Upload files one by one
print("\n[4/5] Uploading files...")
api = huggingface_hub.HfApi(token=token)
repo_id = "liubuing/bfn-confidence-general-proteins"
repo_type = "dataset"

src_dir = "C:/biological/AntibodyDesignBFN-main/AntibodyDesignBFN-main/data/confidence_dataset"
files_to_upload = [
    "dataset_summary.json",
    "confidence_train.lmdb/data.mdb",
    "confidence_train.lmdb/lock.mdb",
    "confidence_val.lmdb/data.mdb",
    "confidence_val.lmdb/lock.mdb",
]

for rel_path in files_to_upload:
    local_path = os.path.join(src_dir, rel_path)
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {rel_path}")
        continue
    size_kb = os.path.getsize(local_path) / 1024
    print(f"  Uploading {rel_path} ({size_kb:.1f} KB)...", end=" ", flush=True)
    try:
        # Use path_in_repo to put files in the right structure
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
        )
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

# Step 5: Upload README
print("\n[5/5] Uploading README...")
readme = """---
license: mit
language: en
tags:
- protein
- confidence-prediction
- plddt
- iptm
- pae
- bfn
- alphaflow2
- biology
size_categories:
- n<1K
pretty_name: BFN Confidence General Proteins
---

# BFN Confidence General Proteins

Dataset for fine-tuning BFN (Bayesian Flow Network) confidence heads on general
single-chain protein structures using AlphaFold2 as teacher.

## Contents

- `confidence_train.lmdb/` — 16 training entries (LMDB)
- `confidence_val.lmdb/` — 4 validation entries (LMDB)
- `dataset_summary.json` — metadata (counts, split ratio, creation date)

## Format

Each LMDB entry is a pickled dictionary:
- `pdb_id`: str — PDB identifier
- `sequence`: str — amino acid sequence
- `batch`: dict — preprocessed BFN-compatible input
- `af2_plddt`: list[float] — per-residue pLDDT [0,1] (teacher)
- `af2_iptm`: float — ipTM score (teacher)
- `af2_pae_matrix`: list[list[float]] — PAE matrix L×L (teacher)

## Usage

```python
from antibodydesignbfn.datasets.confidence_dataset import ConfidenceRegressionDataset

dataset = ConfidenceRegressionDataset("path/to/confidence_train.lmdb")
```

## Source

Built from PDB structures matched to ColabFold (AlphaFold2) predictions.
See `build_lmdb_from_af2.py` for the construction logic.
"""
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type=repo_type,
    token=token,
)
print("  README uploaded")

print("\n=== DONE ===")
print(f"View at: https://huggingface.co/datasets/liubuing/bfn-confidence-general-proteins")
