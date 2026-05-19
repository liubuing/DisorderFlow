"""Upload V6 confidence dataset to HuggingFace — direct foreground execution."""
import sys, os, json, time

token = os.environ.get("HF_TOKEN", "")
repo_id = "liubuing/bfn-confidence-general-proteins"
src_dir = "C:/biological/AntibodyDesignBFN-main/AntibodyDesignBFN-main/data/confidence_merged_v6"

print("=== HF Upload V6 Dataset ===")
print(f"Source: {src_dir}")

# Step 1: Login
print("\n[1/5] Logging in...")
import huggingface_hub
huggingface_hub.login(token=token)
print(f"  Logged in as: {huggingface_hub.whoami()['name']}")

# Step 2: Create repo
print("\n[2/5] Ensuring repo exists...")
try:
    url = huggingface_hub.create_repo(
        repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )
    print(f"  Repo: {url}")
except Exception as e:
    print(f"  create_repo error: {e}")

# Step 3: Upload LMDB files
print("\n[3/5] Uploading dataset files...")
api = huggingface_hub.HfApi(token=token)

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
    size_mb = os.path.getsize(local_path) / (1024*1024)
    print(f"  Uploading {rel_path} ({size_mb:.1f} MB)...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

# Step 4: Upload README
print("\n[4/5] Uploading README...")
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
- antibody
- brain-disease
size_categories:
- 1K<n<10K
pretty_name: BFN Confidence General Proteins (V6)
---

# BFN Confidence General Proteins — V6

Dataset for fine-tuning BFN (Bayesian Flow Network) confidence heads on
general single-chain protein structures using AlphaFold2 as teacher.

**V6** is the largest release: it combines V5 (general proteins + disease proteins)
with brain disease proteins (26 categories), then deduplicates.

## Contents

| Split | Entries | LMDB Size |
|-------|---------|-----------|
| Train | 1,625   | ~2.8 GB   |
| Val   | 407     | ~667 MB   |
| **Total** | **2,032** | **~3.5 GB** |

Subsets:
- Brain disease: 713 train / 170 val
- IDP (intrinsically disordered): 32 train / 5 val
- Duplicates removed during merge: 240

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

## Version History

| Version | Train | Val | Notes |
|---------|-------|-----|-------|
| V6 | 1,625 | 407 | +brain disease (26 cats), deduped |
| V5 | 919 | 230 | +disease v2 |
| V4 | 474 | 121 | expanded general proteins |

## Source

Built from PDB structures matched to ColabFold (AlphaFold2) predictions.
"""
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
    token=token,
)
print("  README uploaded")

# Step 5: Verify
print("\n[5/5] Verifying...")
try:
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    print(f"  Files in repo: {files}")
except Exception as e:
    print(f"  Verify error: {e}")

print("\n=== DONE ===")
print(f"View at: https://huggingface.co/datasets/{repo_id}")
