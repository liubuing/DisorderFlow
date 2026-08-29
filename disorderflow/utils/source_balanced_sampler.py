"""Source-balanced batching that never splits a contrastive evidence group."""

from __future__ import annotations

import torch


class SourceBalancedCompleteGroupBatchSampler(torch.utils.data.Sampler):
    """Round-robin complete groups across sources with deterministic epochs."""

    def __init__(self, dataset, max_batch_records, shuffle=True, seed=0):
        groups = [tuple(group) for group in dataset.group_indices]
        sources = list(dataset.group_source_ids)
        if len(groups) != len(sources) or not groups:
            raise ValueError("Each complete group requires one source id")
        if max(map(len, groups)) > int(max_batch_records):
            raise ValueError("A complete group exceeds batch capacity")
        self.groups = groups
        self.sources = sources
        self.max_batch_records = int(max_batch_records)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _ordered_groups(self):
        by_source = {}
        for group, source in zip(self.groups, self.sources, strict=True):
            by_source.setdefault(source, []).append(group)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        source_names = sorted(by_source, key=str)
        if self.shuffle:
            source_names = [source_names[i] for i in torch.randperm(
                len(source_names), generator=generator).tolist()]
            for source in source_names:
                rows = by_source[source]
                order = torch.randperm(len(rows), generator=generator).tolist()
                by_source[source] = [rows[i] for i in order]
        ordered = []
        depth = 0
        while True:
            added = False
            for source in source_names:
                if depth < len(by_source[source]):
                    ordered.append(by_source[source][depth])
                    added = True
            if not added:
                return ordered
            depth += 1

    def _batches(self):
        batches, current = [], []
        for group in self._ordered_groups():
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
