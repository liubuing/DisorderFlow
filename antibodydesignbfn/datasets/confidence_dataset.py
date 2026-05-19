"""Dataset for BFN confidence head fine-tuning.

Loads pre-computed LMDB entries that pair BFN-preprocessed protein batches
with AF2-derived ground-truth confidence scores (pLDDT, ipTM, PAE).
"""

import os, pickle
import torch
from torch.utils.data import Dataset

from antibodydesignbfn.datasets._base import register_dataset


@register_dataset('confidence_regression')
class ConfidenceRegressionDataset(Dataset):
    """Load BFN batch + AF2 confidence ground truth from LMDB.

    Each entry in the LMDB is a dict with keys:
        pdb_id, sequence, batch, af2_plddt, af2_iptm, af2_pae_matrix

    The batch dict contains standard BFN input tensors (aa, pos_heavyatom,
    mask_heavyatom, generate_flag, etc.) prepared by MaskRegion → MergeProtein
    → PatchProtein transform pipeline.
    """

    def __init__(self, cfg, transform=None, **kwargs):
        super().__init__()
        # Accept both EasyDict/config dict and plain string path
        if isinstance(cfg, str):
            self.db_path = cfg
        else:
            self.db_path = cfg.db_path if hasattr(cfg, 'db_path') else cfg['db_path']
        # transform is ignored — data is already preprocessed in the LMDB

        if os.path.isdir(self.db_path) and not os.path.exists(os.path.join(self.db_path, 'data.mdb')):
            # Pickle directory fallback
            self._use_lmdb = False
            meta_path = os.path.join(self.db_path, 'meta.json')
            if os.path.exists(meta_path):
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                self._length = meta['n_entries']
            else:
                # Count .pkl files
                self._length = len([f for f in os.listdir(self.db_path) if f.endswith('.pkl')])
            self._pkl_dir = self.db_path
        else:
            self._use_lmdb = True
            import lmdb
            self._env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False)
            with self._env.begin() as txn:
                self._length = pickle.loads(txn.get(b'__len__'))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if self._use_lmdb:
            with self._env.begin() as txn:
                key = f'{index:08d}'.encode()
                entry = pickle.loads(txn.get(key))
        else:
            pkl_path = os.path.join(self._pkl_dir, f'{index:08d}.pkl')
            with open(pkl_path, 'rb') as f:
                entry = pickle.load(f)

        # Reconstruct batch and attach AF2 ground truth
        batch = entry['batch']
        # Data is stored without batch dim — PaddingCollate handles stacking
        batch['af2_plddt'] = entry['af2_plddt']
        batch['af2_iptm'] = entry['af2_iptm']
        batch['af2_pae_matrix'] = entry['af2_pae_matrix']
        batch['pdb_id'] = entry.get('pdb_id', '')
        batch['is_idp'] = entry.get('is_idp', False)
        batch['source'] = entry.get('source', '')

        # For confidence regression, all residues are "context" (not generated).
        # This allows the encoder to see full structural information, which is
        # the correct setting for predicting confidence from structure features.
        # The generate_flag from the saved batch is overridden here.
        batch['generate_flag'] = torch.zeros(batch['aa'].shape[0], dtype=torch.bool)

        return batch

    def close(self):
        if self._use_lmdb and hasattr(self, '_env'):
            self._env.close()
