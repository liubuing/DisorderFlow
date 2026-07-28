"""Disorder-Augmented Dataset Wrapper (WAY4 V7 P0-fix-B).

Wraps Phase3Dataset to inject per-residue disorder profiles (epitope_disorder_profile)
into each training sample. The per-residue disorder comes from a pre-computed lookup
dict mapping SAbDab sample IDs to disorder arrays.

Used by: train.py with --disorder-lookup flag
"""

import numpy as np
import torch


class DisorderAugmentedDataset:
    """Wraps a Phase3Dataset to add per-residue epitope disorder profiles.

    For each sample:
    - Looks up the pre-computed per-residue disorder array by sample ID
    - Builds epitope_disorder_profile: zeros for antibody, disorder for antigen
    - The profile has shape (L,) where L is the patched sequence length
    """

    def __init__(self, base_dataset, disorder_lookup, sample_ids):
        """
        Args:
            base_dataset: Phase3Dataset instance
            disorder_lookup: dict {sample_id: np.ndarray of per-residue disorder [0,1]}
            sample_ids: list of sample IDs in same order as base_dataset
        """
        self._base = base_dataset
        self._lookup = disorder_lookup
        self._ids = sample_ids

    def __getitem__(self, idx):
        sid = self._ids[idx]
        disorder_arr = self._lookup.get(sid) if self._lookup else None
        if disorder_arr is None and self._lookup:
            disorder_arr = self._lookup.get(str(sid).casefold())

        # Phase3 transforms merge and crop chains. Attach the profile to the raw
        # antigen first so those transforms preserve the exact residue mapping.
        if hasattr(self._base, 'get_structure') and hasattr(self._base, 'transform'):
            structure = self._base.get_structure(idx)
            if structure is not None and structure.get('antigen') is not None:
                antigen = structure['antigen']
                profile = torch.zeros(antigen['aa'].shape[0], dtype=torch.float32)
                if disorder_arr is not None and len(disorder_arr) > 0:
                    n_copy = min(profile.shape[0], len(disorder_arr))
                    profile[:n_copy] = torch.as_tensor(
                        np.asarray(disorder_arr)[:n_copy], dtype=torch.float32)
                antigen['epitope_disorder_profile'] = profile
                data = self._base.transform(structure) if self._base.transform else structure
                if data is not None:
                    return self._add_scalar_mean(data)
                raise RuntimeError(
                    f'Transform removed disorder-augmented sample {sid}')

        data = self._base[idx]

        # Always add profile (all zeros if no disorder data available)
        # This ensures PaddingCollate._get_common_keys finds the field in ALL samples
        fragment_type = data.get('fragment_type')
        L = data['aa'].shape[0]
        profile = torch.zeros(L, dtype=torch.float32)

        if disorder_arr is not None and len(disorder_arr) > 0 and fragment_type is not None:
            ag_mask = (fragment_type == 3)  # Fragment.Antigen = 3
            n_ag = ag_mask.sum().item()
            if n_ag > 0:
                n_copy = min(n_ag, len(disorder_arr))
                ag_indices = ag_mask.nonzero(as_tuple=True)[0]
                profile[ag_indices[:n_copy]] = torch.tensor(
                    disorder_arr[:n_copy], dtype=torch.float32
                )

        data['epitope_disorder_profile'] = profile

        return self._add_scalar_mean(data)

    @staticmethod
    def _add_scalar_mean(data):
        profile = data['epitope_disorder_profile']
        fragment_type = data.get('fragment_type')
        if fragment_type is not None:
            ag_mask = (fragment_type == 3)
            ag_profile = profile[ag_mask]
            ag_mean = ag_profile.mean().item() if ag_profile.numel() > 0 else 0.0
        else:
            ag_mean = profile.mean().item()
        data['epitope_disorder'] = torch.tensor([[ag_mean]], dtype=torch.float32)
        return data

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        # Delegate all other attributes to base dataset
        # This ensures _scaffold_ids, _sample_weights, set_max_residues, etc. work
        return getattr(self._base, name)


class ContrastiveNegativeDataset:
    """Attach a real CDR from another complex as an antigen-mismatched negative."""

    def __init__(self, base_dataset):
        self._base = base_dataset

    def __getitem__(self, idx):
        data = self._base[idx]
        donor_idx = (idx + max(1, len(self) // 2)) % len(self)
        donor = self._base[donor_idx]
        current_idx = torch.where(data['generate_flag'].bool())[0]
        donor_aa = donor['aa'][donor['generate_flag'].bool()]
        negative = data['aa'].clone()
        if len(current_idx) and len(donor_aa):
            source_idx = torch.linspace(
                0, len(donor_aa) - 1, len(current_idx)).round().long()
            negative[current_idx] = donor_aa[source_idx]
        data['contrastive_negative_aa'] = negative
        return data

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)


def load_disorder_lookup(path):
    """Load pre-computed disorder lookup from pickle."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
