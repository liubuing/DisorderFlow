#!/usr/bin/env python
"""Phase 3 Dataset: Antibody-Antigen Complexes for Cross-Chain Fine-Tuning.

Loads preprocessed LMDB data (from preprocess_phase3_dataset.py), applies
CDR masking transforms, and returns batches compatible with BFN training.

Key difference from standard SAbDab dataset:
- Masks ALL CDRs (H1+H2+H3 or L1+L2+L3) for co-design training
- Preserves antigen structure as context (not masked)
- Supports cross-chain feature flow via generate_flag
"""

import os
import random
import pickle
import logging

import lmdb
import torch
from torch.utils.data import Dataset

from ._base import register_dataset

# LMDB read-ahead is bad: must open/close per-worker in DataLoader
# We use a connection factory pattern


class Phase3Dataset(Dataset):
    """Antibody-antigen complex dataset for cross-chain fine-tuning.

    Provides antibody CDR co-design tasks with full antigen context.
    """

    MAP_SIZE = 4 * (1024 * 1024 * 1024)  # 4GB

    def __init__(self, cfg, transform=None):
        super().__init__()
        self.lmdb_path = cfg.lmdb_path
        self.ids_path = cfg.lmdb_path + '-ids'
        self.transform = transform

        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(
                f"Phase 3 LMDB not found: {self.lmdb_path}. "
                "Run scripts/preprocess_phase3_dataset.py first."
            )

        with open(self.ids_path, 'rb') as f:
            self.all_ids = pickle.load(f)

        self.ids = self.all_ids
        logging.info(f"Phase3Dataset loaded {len(self.ids)} entries from {self.lmdb_path}")

    def _connect_db(self):
        # Each worker needs its own connection
        return lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, idx):
        db_conn = self._connect_db()
        try:
            with db_conn.begin() as txn:
                data = pickle.loads(txn.get(self.ids[idx].encode()))
            return data
        finally:
            db_conn.close()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        data = self.get_structure(index)
        if data is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.transform is not None:
            try:
                data = self.transform(data)
            except Exception as e:
                logging.warning(f"Transform failed for {self.ids[index]}: {e}")
                data = None

        if data is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        return data


@register_dataset('phase3')
def get_phase3_dataset(cfg, transform):
    return Phase3Dataset(cfg, transform=transform)
