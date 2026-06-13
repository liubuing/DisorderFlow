"""Region-based masking for general protein design (no CDR concepts)."""
import torch
from ._base import register_transform


@register_transform('mask_region')
class MaskRegion(object):
    """
    Set generate_flag on specified residue positions for general proteins.

    Args:
        regions: dict mapping chain_id -> list of 0-based residue indices
            e.g. {'A': [10, 11, 12, ..., 25], 'B': [5, 15, 25]}
        anchor_mode: 'flanking' (anchor on neighbors) or 'full' (anchor on all design residues)
    """

    def __init__(self, regions, anchor_mode='flanking'):
        self.regions = regions
        self.anchor_mode = anchor_mode

    def __call__(self, structure):
        chains = structure['chains']

        for chain_entry in chains:
            chain_id = chain_entry['chain_id']
            data = chain_entry['data']

            if chain_id not in self.regions:
                # This chain is context only
                data['generate_flag'] = torch.zeros(data['aa'].shape, dtype=torch.bool)
                data['anchor_flag'] = torch.zeros(data['aa'].shape, dtype=torch.bool)
                continue

            design_indices = self.regions[chain_id]
            n_res = data['aa'].size(0)

            # Validate indices
            valid_indices = [i for i in design_indices if 0 <= i < n_res]
            if len(valid_indices) == 0:
                data['generate_flag'] = torch.zeros(n_res, dtype=torch.bool)
                data['anchor_flag'] = torch.zeros(n_res, dtype=torch.bool)
                continue

            generate_flag = torch.zeros(n_res, dtype=torch.bool)
            generate_flag[torch.tensor(valid_indices, dtype=torch.long)] = True
            data['generate_flag'] = generate_flag

            # Set anchor flag
            anchor_flag = torch.zeros(n_res, dtype=torch.bool)
            if self.anchor_mode == 'flanking':
                left_idx = max(0, min(valid_indices) - 1)
                right_idx = min(n_res - 1, max(valid_indices) + 1)
                anchor_flag[left_idx] = True
                anchor_flag[right_idx] = True
            elif self.anchor_mode == 'full':
                anchor_flag[torch.tensor(valid_indices, dtype=torch.long)] = True
            data['anchor_flag'] = anchor_flag

        return structure
