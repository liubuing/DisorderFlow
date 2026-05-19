"""Generic chain merging for general protein design (no heavy/light/antigen)."""
import torch
from ._base import register_transform


@register_transform('merge_protein')
class MergeProtein(object):
    """Merge all chains into a single flat data structure."""

    def __init__(self):
        super().__init__()

    def _data_attr(self, data, name):
        if name in ('generate_flag', 'anchor_flag') and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        return data[name]

    def __call__(self, structure):
        chains = structure['chains']

        # Assign fragment_type and chain_nb per chain
        for ci, chain_entry in enumerate(chains):
            data = chain_entry['data']
            data['fragment_type'] = torch.full_like(data['aa'], fill_value=ci)
            data['chain_nb'] = torch.full_like(data['aa'], fill_value=ci)

        # Collect all chains
        chain_ids = set()
        for chain_entry in chains:
            chain_ids.update(chain_entry['data']['chain_id'])
        chain_id_to_nb = {c: i for i, c in enumerate(sorted(chain_ids))}

        # Re-assign chain_nb from globally unique mapping
        for chain_entry in chains:
            data = chain_entry['data']
            data['chain_nb'] = torch.LongTensor([
                chain_id_to_nb[c] for c in data['chain_id']
            ])

        # Merge
        list_props = {'chain_id': [], 'icode': []}
        tensor_props = {
            'chain_nb': [], 'resseq': [], 'res_nb': [], 'aa': [],
            'pos_heavyatom': [], 'mask_heavyatom': [],
            'generate_flag': [], 'cdr_flag': [], 'anchor_flag': [],
            'fragment_type': [], 'torsion': [], 'mask_torsion': [],
        }

        for chain_entry in chains:
            data = chain_entry['data']
            for k in list_props.keys():
                list_props[k].append(self._data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(self._data_attr(data, k))

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}

        return {**list_props, **tensor_props}
