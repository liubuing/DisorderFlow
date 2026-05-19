#!/usr/bin/env python
"""Upload BFN confidence fine-tuning dataset to HuggingFace.

Creates a dataset repository and uploads:
  - confidence_train.lmdb / confidence_val.lmdb (or pickle dirs)
  - dataset_summary.json
  - dataset card (README.md)

Usage:
  1. Login: huggingface-cli login
  2. python upload_to_huggingface.py --dataset_dir ./data/confidence_dataset \
       --repo_id <USERNAME>/bfn-confidence-general-proteins
"""
import sys, os, json, argparse
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def parse_args():
    p = argparse.ArgumentParser(description='Upload confidence dataset to HuggingFace')
    p.add_argument('--dataset_dir', required=True, help='Path to local dataset directory')
    p.add_argument('--repo_id', required=True,
                   help='HuggingFace repo ID, e.g. "username/bfn-confidence-general-proteins"')
    p.add_argument('--private', action='store_true', default=True,
                   help='Create as private dataset (default: True)')
    p.add_argument('--public', action='store_true', help='Make dataset public')
    return p.parse_args()


README_TEMPLATE = """---
license: mit
language:
  - en
tags:
  - biology
  - protein
  - structure-prediction
  - confidence
  - bfn
  - alphafold
size_categories:
  - n<1K
pretty_name: BFN Confidence Fine-Tuning Dataset (General Proteins)
---

# BFN Confidence Fine-Tuning Dataset

Dataset for fine-tuning **Bayesian Flow Network (BFN)** confidence heads (pLDDT, ipTM, PAE)
on general single-chain protein structures.

## Overview

- **Proteins**: {n_proteins} single-chain proteins with diverse folds (all-alpha, all-beta, alpha/beta, alpha+beta)
- **Source**: RCSB PDB, curated subset of 20 small proteins (50-200 residues, resolution < 2.5A)
- **Ground Truth**: Confidence scores predicted by AlphaFold2 (ColabFold) on native sequences
- **Format**: LMDB with pickle-serialized entries

## Data Format

Each LMDB entry is a dict:

```python
{{
    'pdb_id': str,           # PDB identifier (e.g. '1UBQ')
    'sequence': str,         # Amino acid sequence
    'batch': dict,           # BFN-preprocessed batch (tensors: aa, pos_heavyatom, etc.)
    'af2_plddt': Tensor[L],  # Per-residue pLDDT [0, 1] from AF2
    'af2_iptm': Tensor[],    # ipTM score [0, 1] from AF2
    'af2_pae_matrix': Tensor[L, L],  # PAE matrix from AF2
}}
```

## Loading the Dataset

```python
from antibodydesignbfn.datasets.confidence_dataset import ConfidenceRegressionDataset

dataset = ConfidenceRegressionDataset('confidence_train.lmdb')
entry = dataset[0]
# entry contains batch dict + 'af2_plddt', 'af2_iptm', 'af2_pae_matrix'
```

## Training

Used for fine-tuning BFN confidence heads:
- Freeze BFN backbone (~10M params)
- Train confidence heads + feedback embeddings (~134K params)
- MSE loss: pLDDT (per-residue), ipTM (per-complex), PAE (per-residue-pair, normalized by /31)

## Citation

```bibtex
@article{{antibodydesignbfn,
  title={{AntibodyDesignBFN: Antibody Sequence Design with Bayesian Flow Networks}},
  author={{YueHu Lab}},
  year={{2025}},
}}
```

## Split

- Train: {n_train} entries
- Validation: {n_val} entries
"""


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f'ERROR: Dataset directory not found: {dataset_dir}')
        sys.exit(1)

    # Load summary
    summary_path = dataset_dir / 'dataset_summary.json'
    if not summary_path.exists():
        print(f'ERROR: dataset_summary.json not found in {dataset_dir}')
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    print(f'Dataset: {summary["n_train"]} train + {summary["n_val"]} val entries')
    print(f'Target repo: {args.repo_id}')

    # Login check
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        api = HfApi()
        user = api.whoami()
        print(f'Logged in as: {user["name"]}')
    except Exception as e:
        print(f'ERROR: Not logged into HuggingFace: {e}')
        print('Run: huggingface-cli login')
        sys.exit(1)

    # Create repo
    private = not args.public
    try:
        repo_url = create_repo(
            repo_id=args.repo_id,
            repo_type='dataset',
            private=private,
            exist_ok=True,
        )
        print(f'Repository ready: {repo_url}')
    except Exception as e:
        print(f'ERROR creating repo: {e}')
        sys.exit(1)

    # Generate README
    readme_content = README_TEMPLATE.format(
        n_proteins=summary['n_train'] + summary['n_val'],
        n_train=summary['n_train'],
        n_val=summary['n_val'],
    )
    readme_path = dataset_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f'Generated README.md')

    # Upload
    print(f'Uploading files to {args.repo_id}...')
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(dataset_dir),
        repo_type='dataset',
        commit_message=f'Upload BFN confidence dataset ({summary["n_train"]} train + {summary["n_val"]} val)',
    )

    print(f'\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo_id}')


if __name__ == '__main__':
    main()
