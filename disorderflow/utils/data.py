"""数据整理与批采样工具：变长样本 padding 拼批、LMDB 记录哈希、CDR/证据组分批采样器。"""
import math
import hashlib
import lmdb
import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate


# 各字段 padding 时使用的填充值：
# 'aa' 用 21（20 种标准氨基酸之外的"未知/填充"类别）；
# 'chain_id'/'icode' 是字符字段，用空格填充
DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_id': ' ', 
    'icode': ' ',
}

# 这些字段是整体属性（原点向量/旋转矩阵/标量），不随残基数量 padding
DEFAULT_NO_PADDING = {
    'origin',
    'patch_global_origin',
    'patch_global_rotation',
    'af2_iptm',
}


def update_lmdb_record_digest(digest, key, value):
    # 以 (键长, 键, 值长, 值) 的确定性字节布局喂入哈希，
    # 避免不同键值对拼接产生歧义
    digest.update(len(key).to_bytes(8, "big"))
    digest.update(key)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def lmdb_records_sha256(path):
    # 计算整个 LMDB 中全部 (key, value) 记录的 SHA-256，
    # 用于训练前的"冻结数据血统"校验（与 manifest 中记录的哈希比对）
    digest = hashlib.sha256()
    environment = lmdb.open(
        str(path), readonly=True, lock=False, readahead=False,
        meminit=False, subdir=True)
    try:
        with environment.begin() as transaction:
            for key, value in transaction.cursor():
                update_lmdb_record_digest(digest, key, value)
    finally:
        environment.close()
    return digest.hexdigest()

# Keys that contain 2D matrices (L x L) that should be padded on both dims
MATRIX_KEYS = {
    'af2_pae_matrix',
}

class PaddingCollate(object):
    """把不等长的样本 dict 拼成一个 batch：统一 pad 到最大长度并生成 mask。"""

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        # length_ref_key: 以哪个键的第 0 维长度作为 pad 基准（默认残基序列 'aa'）；
        # eight: 把最大长度向上取整到 8 的倍数，利于 GPU tensor core 对齐加速
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        # 在第 0 维末尾补 value 至长度 n；标量张量原样返回，超长直接断言失败
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
        # True 表示真实残基，False 表示 padding 位置
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        # 只保留所有样本共有的键：缺键的样本会导致整个字段被丢弃，
        # 因此各数据集必须保证关键字段在每个样本上都存在
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
    # 把 patch 上预测/生成的结果按索引写回全长张量（clone 避免原地修改入参）
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full


class CDRBatchSampler(torch.utils.data.Sampler):
    """按 CDR 类型分组组批：每个 batch 只含同一类型（如 H_CDR3）的样本，
    避免不同 CDR 长度/任务混批；不足一个 batch 的尾部样本被丢弃。"""

    def __init__(self, indices_by_type, batch_size, shuffle=True):
        # indices_by_type: {CDR 类型: 样本索引列表}，通常来自 LMDB 的 meta_indices.pkl
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

    # 保证同一证据组（如同一目标的多次测量）完整落在同一个 batch 内，
    # 供组内对比/排序损失使用；打乱时用 seed+epoch 派生确定性随机流
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
        batches = self._batches()
        self.epoch += 1
        yield from batches

    def __len__(self):
        return len(self._batches())
