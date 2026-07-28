"""Attach uniform state-contrast scalar metadata to structural samples."""

import torch
from torch.utils.data import Dataset


class StateContrastMetadataDataset(Dataset):
    """Wrap an aligned structural dataset and expose group-aware batch fields."""

    def __init__(self, dataset, records):
        if len(dataset) != len(records):
            raise ValueError("Structural samples and StateContrast records must be aligned")
        self.dataset = dataset
        self.records = list(records)
        group_ids = sorted({record["group_id"] for record in self.records})
        self._group_index = {group_id: index for index, group_id in enumerate(group_ids)}
        self._scaffold_ids = [self._group_index[record["group_id"]] for record in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        record = self.records[index]
        sample.update({
            "contrastive_group_id": torch.tensor(
                self._group_index[record["group_id"]], dtype=torch.long),
            "contrastive_rank": torch.tensor(
                record["targets"]["contrastive_rank"], dtype=torch.float32),
            "contrastive_weight": torch.tensor(
                record["targets"]["training_weight"], dtype=torch.float32),
        })
        return sample
