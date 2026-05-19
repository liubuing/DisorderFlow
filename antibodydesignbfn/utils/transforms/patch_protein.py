"""Generic spatial patching for general protein design (configurable reference)."""
import torch
from ._base import _mask_select_data, register_transform
from ..protein import constants


@register_transform('patch_protein')
class PatchProtein(object):
    """
    Crop structure to a spatial patch around the design region.
    Configurable reference chain for alignment (no heavy-chain hardcoding).
    """

    def __init__(self, initial_patch_size=128, context_size=128, reference_fragment_type=0):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.context_size = context_size
        self.reference_fragment_type = reference_fragment_type

    def _align(self, data):
        from .align_utils import kabsch_rotation

        ca_mask = data['mask_heavyatom'][:, constants.BBHeavyAtom.CA].bool()
        ref_mask = (data['fragment_type'] == self.reference_fragment_type)
        combined_mask = torch.logical_and(ca_mask, ref_mask)
        valid_indices = torch.where(combined_mask)[0]

        if len(valid_indices) < 3:
            return data

        subset = valid_indices[:min(20, len(valid_indices))]
        P_ca = data['pos_heavyatom'][subset, constants.BBHeavyAtom.CA]

        origin = P_ca[0]
        x_vec = P_ca[-1] - origin
        x_vec = x_vec / (torch.norm(x_vec) + 1e-6)
        mid_vec = P_ca[len(P_ca) // 2] - origin
        z_vec = torch.cross(x_vec, mid_vec, dim=0)
        z_vec = z_vec / (torch.norm(z_vec) + 1e-6)
        y_vec = torch.cross(z_vec, x_vec, dim=0)
        R_inv = torch.stack([x_vec, y_vec, z_vec], dim=1)

        pos_centered = data['pos_heavyatom'] - origin.view(1, 1, 3)
        pos_aligned = torch.matmul(pos_centered, R_inv)
        data['pos_heavyatom'] = pos_aligned * data['mask_heavyatom'][:, :, None]
        data['patch_global_origin'] = origin
        data['patch_global_rotation'] = R_inv.t()

        return data

    def __call__(self, data):
        anchor_flag = data.get('anchor_flag', torch.zeros(data['aa'].shape, dtype=torch.bool))
        design_mask = (data['fragment_type'] == self.reference_fragment_type)
        context_mask = (data['fragment_type'] != self.reference_fragment_type)
        n_total = data['aa'].size(0)

        if anchor_flag.sum().item() == 0:
            # No anchor - take all reference chain residues
            data_patch = _mask_select_data(data=data, mask=design_mask)
            if data_patch['aa'].size(0) > 0:
                data_patch['anchor_flag'] = torch.zeros(data_patch['aa'].shape, dtype=torch.bool)
                data_patch['anchor_flag'][0] = True
            data_patch = self._align(data_patch)
            return data_patch

        pos_alpha = data['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
        anchor_points = pos_alpha[anchor_flag]

        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]

        # Select design region residues near anchor
        initial_patch_idx = torch.topk(
            dist_anchor,
            k=min(self.initial_patch_size, n_total),
            largest=False,
        )[1]

        # Select context residues near anchor
        dist_anchor_context = dist_anchor.clone()
        dist_anchor_context[design_mask] = float('+inf')
        context_patch_idx = torch.topk(
            dist_anchor_context,
            k=min(self.context_size, context_mask.sum().item()),
            largest=False, sorted=True
        )[1]

        patch_mask = torch.zeros(n_total, dtype=torch.bool)
        patch_mask[initial_patch_idx] = True
        patch_mask[context_patch_idx] = True
        patch_mask[data['generate_flag']] = True
        patch_mask[anchor_flag] = True

        patch_idx = torch.arange(0, n_total)[patch_mask]
        data_patch = _mask_select_data(data, patch_mask)
        data_patch = self._align(data_patch)
        data_patch['patch_idx'] = patch_idx

        return data_patch
