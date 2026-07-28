import math
import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate


DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_id': ' ', 
    'icode': ' ',
}

DEFAULT_NO_PADDING = {
    'origin',
    'patch_global_origin',
    'patch_global_rotation',
    'af2_iptm',
}

# Keys that contain 2D matrices (L x L) that should be padded on both dims
MATRIX_KEYS = {
    'af2_pae_matrix',
}

class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            if x.dim() == 0:  # scalar tensor —no padding
                return x
            if x.size(0) > n:
                raise AssertionError(
                    f'Padding error: tensor shape={tuple(x.shape)}, '
                    f'dim0={x.size(0)} > max_length={n}')
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)

        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {}
            for k, v in data.items():
                if k not in keys:
                    continue
                if k in self.no_padding:
                    data_padded[k] = v
                elif k in MATRIX_KEYS and isinstance(v, torch.Tensor) and v.dim() == 2:
                    # Pad 2D matrix (L, L) to (max_length, max_length)
                    L = v.shape[0]
                    padded = torch.zeros(max_length, max_length, dtype=v.dtype, device=v.device)
                    padded[:L, :L] = v
                    data_padded[k] = padded
                else:
                    try:
                        data_padded[k] = self._pad_last(v, max_length, value=self._get_pad_value(k))
                    except AssertionError as e:
                        raise AssertionError(f'Key="{k}" {e}')
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def apply_patch_to_tensor(x_full, x_patch, patch_idx):
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full


class CDRBatchSampler(torch.utils.data.Sampler):
    def __init__(self, indices_by_type, batch_size, shuffle=True):
        self.indices_by_type = indices_by_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = self._make_batches()

    def _make_batches(self):
        all_batches = []
        for c_type, indices in self.indices_by_type.items():
            if self.shuffle:
                random_indices = torch.randperm(len(indices)).tolist()
                indices = [indices[i] for i in random_indices]
            
            # Split indices into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
        
        if self.shuffle:
            np.random.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class CompleteGroupBatchSampler(torch.utils.data.Sampler):
    """Pack whole evidence groups without splitting members across batches."""

    def __init__(self, dataset, max_batch_records, shuffle=True, seed=0):
        groups = [tuple(group) for group in dataset.group_indices]
        if not groups:
            raise ValueError("CompleteGroupBatchSampler requires non-empty groups")
        largest = max(len(group) for group in groups)
        if largest > max_batch_records:
            raise ValueError(
                f"Largest group ({largest}) exceeds batch capacity ({max_batch_records})")
        self.groups = groups
        self.max_batch_records = int(max_batch_records)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _batches(self):
        groups = list(self.groups)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[index] for index in order]
        batches = []
        current = []
        for group in groups:
            if current and len(current) + len(group) > self.max_batch_records:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return len(self._batches())
