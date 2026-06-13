"""Multi-Conformation Dataset for BFN disorder training.

Loads entries with multiple AF2-predicted conformations + RMSF labels.
On each __getitem__, randomly selects a conformation to expose the model
to structural variability, teaching it to distinguish ordered vs disordered regions
through gradient contradictions.
"""

import random
import pickle
import torch
from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset
from disorderflow.datasets._base import register_dataset


# AlphaFold atomic indices (37-atom representation)
AF2_N_IDX = 0
AF2_CA_IDX = 1
AF2_C_IDX = 2

# BFN atom indices 鈥?N, CA, C are the first 3 in pos_heavyatom
BFN_N_IDX = 0
BFN_CA_IDX = 1
BFN_C_IDX = 2


@register_dataset('conformation_regression')
class ConformationRegressionDataset(ConfidenceRegressionDataset):
    """Dataset that provides multi-conformation entries for disorder training.

    Extends ConfidenceRegressionDataset with:
      - Multi-conformation sampling: randomly picks one conformation per __getitem__
      - RMSF labels: continuous per-residue disorder scores from AF2 ensemble
      - Backbone substitution: replaces N, CA, C positions with AF2-predicted coords
    """

    def __init__(self, cfg, transform=None, **kwargs):
        super().__init__(cfg, transform=transform, **kwargs)
        # Additional config options
        if isinstance(cfg, dict):
            self.fix_backbone = cfg.get('fix_backbone', False)
        else:
            self.fix_backbone = getattr(cfg, 'fix_backbone', False)

    def __getitem__(self, index):
        real_idx = self._valid_indices[index]
        if self._use_lmdb:
            with self._env.begin() as txn:
                key = f'{real_idx:08d}'.encode()
                entry = pickle.loads(txn.get(key))
        else:
            import os
            pkl_path = os.path.join(self._pkl_dir, f'{real_idx:08d}.pkl')
            with open(pkl_path, 'rb') as f:
                entry = pickle.load(f)

        batch = entry['batch']
        n_conformations = entry.get('n_conformations', 1)

        # Randomly select conformation
        if n_conformations > 1 and not self.fix_backbone:
            conf_idx = random.randint(0, n_conformations - 1)
            ca_positions = entry['ca_positions']  # list of (N_res, 3) arrays

            # Replace backbone C伪 in pos_heavyatom
            # ca_positions[conf_idx] is (N_res, 3)
            ca = torch.from_numpy(ca_positions[conf_idx])
            n_res = min(ca.shape[0], batch['pos_heavyatom'].shape[0])
            batch['pos_heavyatom'][:n_res, BFN_CA_IDX, :] = ca[:n_res]

            # Invalidate other backbone atoms for disordered regions
            # (N and C positions are inconsistent with new CA)
            rmsf = entry.get('rmsf', None)
            if rmsf is not None:
                rmsf_t = torch.from_numpy(rmsf[:n_res])
                # For high-RMSF residues, zero out N and C positions
                # to signal structural uncertainty
                high_rmsf = rmsf_t > 2.0  # RMSF > 2脜 = disordered
                batch['pos_heavyatom'][:n_res, BFN_N_IDX][high_rmsf] = 0
                batch['pos_heavyatom'][:n_res, BFN_C_IDX][high_rmsf] = 0
                batch['mask_heavyatom'][:n_res, BFN_N_IDX][high_rmsf] = 0
                batch['mask_heavyatom'][:n_res, BFN_C_IDX][high_rmsf] = 0

        # Attach confidence labels
        batch['af2_plddt'] = entry.get('af2_plddt',
                                       torch.zeros(batch['aa'].shape[0]))
        batch['af2_iptm'] = entry.get('af2_iptm', torch.tensor(0.0))
        batch['af2_pae_matrix'] = entry.get('af2_pae_matrix',
                                            torch.zeros(batch['aa'].shape[0],
                                                        batch['aa'].shape[0]))
        batch['pdb_id'] = entry.get('pdb_id', '')
        batch['is_idp'] = entry.get('is_idp', False)
        batch['source'] = entry.get('source', '')

        # Use RMSF-based continuous disorder label
        if 'disorder_label' in entry:
            disorder_t = torch.from_numpy(entry['disorder_label'])
            batch['disorder_label'] = disorder_t.float()
        elif 'disorder_mask' in batch:
            batch['disorder_label'] = batch['disorder_mask'].float()
        else:
            is_idp = batch.get('is_idp', False)
            if isinstance(is_idp, torch.Tensor):
                is_idp = is_idp.item() if is_idp.numel() == 1 else is_idp[0].item()
            seq_len = batch['aa'].shape[0]
            batch['disorder_label'] = (torch.ones(seq_len) if is_idp
                                       else torch.zeros(seq_len))

        # Override generate_flag for confidence regression
        batch['generate_flag'] = torch.zeros(batch['aa'].shape[0], dtype=torch.bool)

        return batch
