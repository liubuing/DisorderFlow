"""Rank-only flexibility dataset backed by multi-prediction RMSF labels."""

import pickle
import torch
from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset
from disorderflow.datasets._base import register_dataset


@register_dataset('conformation_regression')
class ConformationRegressionDataset(ConfidenceRegressionDataset):
    """Dataset that provides multi-conformation entries for disorder training.

    Extends ConfidenceRegressionDataset with:
      - RMSF labels: continuous per-residue disorder scores from AF2 ensemble
      - Reference geometry: never combines a conformer CA trace with unrelated N/C/O
    """

    def __init__(self, cfg, transform=None, **kwargs):
        super().__init__(cfg, transform=transform, **kwargs)
        # Additional config options
        if isinstance(cfg, dict):
            self.fix_backbone = cfg.get('fix_backbone', False)
            self.random_conformation = cfg.get('random_conformation', True)
            self.geometry_mode = cfg.get('geometry_mode', 'reference_backbone')
        else:
            self.fix_backbone = getattr(cfg, 'fix_backbone', False)
            self.random_conformation = getattr(cfg, 'random_conformation', True)
            self.geometry_mode = getattr(cfg, 'geometry_mode', 'reference_backbone')
        if self.geometry_mode != 'reference_backbone':
            raise ValueError(
                'Only geometry_mode=reference_backbone is supported: the source stores '
                'CA-only conformers, which cannot define valid BFN residue frames')

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

        batch['ensemble_size'] = torch.tensor(n_conformations, dtype=torch.long)
        batch['flexibility_evidence_tier'] = 'af2_multiseed_nmr_rank_calibrated'

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
