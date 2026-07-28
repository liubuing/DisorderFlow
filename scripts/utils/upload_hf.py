#!/usr/bin/env python
"""Upload BFN confidence dataset to HuggingFace.

Usage:
  python upload_hf.py --dataset_dir ./data/confidence_merged_v6 --repo_id user/dataset-name
  python upload_hf.py --version v6    # Pre-configured V6 upload
"""
import sys, os, json, time
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_DIR = Path(__file__).parent.parent
TOKEN = os.environ.get("HF_TOKEN", "")
DEFAULT_REPO = "liubuing/bfn-confidence-general-proteins"

README_V1 = """---
license: mit
language: en
tags: [protein, confidence-prediction, plddt, iptm, pae, bfn, alphafold2, biology]
size_categories: [n<1K]
pretty_name: BFN Confidence General Proteins
---

# BFN Confidence General Proteins

Dataset for fine-tuning BFN confidence heads on general single-chain proteins.

## Contents
- `confidence_train.lmdb/` — {n_train} training entries
- `confidence_val.lmdb/` — {n_val} validation entries
- `dataset_summary.json` — metadata

## Format
Each LMDB entry is a pickled dict with `pdb_id`, `sequence`, `batch`, `af2_plddt`, `af2_iptm`, `af2_pae_matrix`.

## Usage
```python
from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset
dataset = ConfidenceRegressionDataset("path/to/confidence_train.lmdb")
```
"""

README_V6 = """---
license: mit
language: en
tags: [protein, confidence-prediction, plddt, iptm, pae, bfn, alphafold2, biology, antibody, brain-disease]
size_categories: [1K<n<10K]
pretty_name: BFN Confidence General Proteins (V6)
---

# BFN Confidence General Proteins -- V6

**V6** is the largest release: V5 + 1,123 brain disease proteins (26 categories), deduplicated.

## Contents
| Split | Entries |
|-------|---------|
| Train | {n_train} |
| Val   | {n_val} |

## Format
Each LMDB entry: `pdb_id`, `sequence`, `batch`, `af2_plddt`, `af2_iptm`, `af2_pae_matrix`.

## Version History
| Version | Train | Val | Notes |
|---------|-------|-----|-------|
| V6 | 1,625 | 407 | +brain disease |
| V5 | 919 | 230 | +disease v2 |
| V4 | 474 | 121 | expanded proteins |
"""

README_V8 = """---
license: mit
language: en
tags: [protein, confidence-prediction, plddt, iptm, pae, bfn, alphafold2, biology, brain-disease, neurodegeneration]
size_categories: [1K<n<10K]
pretty_name: BFN Confidence Brain Disease Proteins (V8)
---

# BFN Confidence Brain Disease Proteins -- V8

**V8** focuses on brain/neurological disease proteins with quality filtering (ipTM > 0.6).

## Key improvements over V6/V7
- Quality filter: all entries have AF2 ipTM > 0.6 (removes ~50% low-quality data)
- Expanded brain disease coverage: 43 categories (V6 had 28)
- New categories: neuroinflammation, repeat expansion disorders, cerebral small vessel disease, brain iron accumulation, leukodystrophy, cerebral palsy
- Liver disease: top 200 by ipTM only (quality supplement, not bulk)

## Contents
| Split | Entries |
|-------|---------|
| Train | {n_train} |
| Val   | {n_val} |

## Format
Each LMDB entry: `pdb_id`, `sequence`, `batch`, `af2_plddt`, `af2_iptm`, `af2_pae_matrix`, `category`.

## Version History
| Version | Train | Val | Notes |
|---------|-------|-----|-------|
| V8 | {n_train} | {n_val} | Quality-filtered brain-focused |
| V7 | 1,866 | 464 | +liver disease (regressed) |
| V6 | 1,625 | 407 | +brain disease |
| V5 | 919 | 230 | +disease v2 |
"""


def upload_files(api, src_dir, files_to_upload, repo_id, repo_type="dataset"):
    for rel_path in files_to_upload:
        local_path = os.path.join(src_dir, rel_path)
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {rel_path}")
            continue
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Uploading {rel_path} ({size_mb:.1f} MB)...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=TOKEN,
            )
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")


def upload_dataset(dataset_dir, repo_id, readme_template=None, private=False):
    """Upload a dataset directory to HuggingFace."""
    import huggingface_hub

    if not TOKEN:
        print("ERROR: HF_TOKEN not set. Run: $env:HF_TOKEN='your_token'")
        sys.exit(1)

    src = Path(dataset_dir)
    if not src.exists():
        print(f"ERROR: Dataset directory not found: {src}")
        sys.exit(1)

    # Load summary
    summary_path = src / 'dataset_summary.json'
    if not summary_path.exists():
        print(f"ERROR: dataset_summary.json not found in {src}")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Dataset: {summary['n_train']} train + {summary['n_val']} val")
    print(f"Target: {repo_id}")

    # Login
    huggingface_hub.login(token=TOKEN)
    user = huggingface_hub.whoami()
    print(f"Logged in as: {user['name']}")

    # Create repo
    try:
        url = huggingface_hub.create_repo(
            repo_id, repo_type="dataset", private=private, exist_ok=True, token=TOKEN)
        print(f"Repo: {url}")
    except Exception as e:
        print(f"create_repo error: {e}")

    # Upload files
    api = huggingface_hub.HfApi(token=TOKEN)
    files_to_upload = [
        "dataset_summary.json",
        "confidence_train.lmdb/data.mdb",
        "confidence_train.lmdb/lock.mdb",
        "confidence_val.lmdb/data.mdb",
        "confidence_val.lmdb/lock.mdb",
    ]
    upload_files(api, str(src), files_to_upload, repo_id)

    # Upload README
    print("  Uploading README.md...", end=" ", flush=True)
    template = readme_template or README_V1
    readme_content = template.format(n_train=summary['n_train'], n_val=summary['n_val'])
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id, repo_type="dataset", token=TOKEN)
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

    print(f"\nDone! https://huggingface.co/datasets/{repo_id}")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Upload confidence dataset to HuggingFace')
    p.add_argument('--dataset_dir', help='Path to dataset directory')
    p.add_argument('--repo_id', help='HuggingFace repo ID')
    p.add_argument('--version', choices=['v1', 'v6', 'v8'], help='Pre-configured version upload')
    p.add_argument('--public', action='store_true', help='Make public')
    args = p.parse_args()

    if args.version == 'v8':
        upload_dataset(
            str(PROJECT_DIR / 'data/confidence_merged_v8'),
            DEFAULT_REPO,
            README_V8,
            private=not args.public)
    elif args.version == 'v6':
        upload_dataset(
            str(PROJECT_DIR / 'data/confidence_merged_v6'),
            DEFAULT_REPO,
            README_V6,
            private=not args.public)
    elif args.version == 'v1':
        upload_dataset(
            str(PROJECT_DIR / 'data/confidence_dataset'),
            DEFAULT_REPO,
            README_V1,
            private=not args.public)
    elif args.dataset_dir and args.repo_id:
        upload_dataset(args.dataset_dir, args.repo_id, private=not args.public)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
