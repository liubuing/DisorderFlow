"""Homology-clustered dataset splitting utilities."""

import random
import shutil
import subprocess
import tempfile
from pathlib import Path


def mmseqs_cluster(entries, sequence_field, min_identity=0.3, coverage=0.8,
                   executable='mmseqs', threads=4):
    """Return entry-index to MMseqs representative mapping for one sequence field."""
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"MMseqs2 executable '{executable}' was not found; refusing an unsafe random split")

    sequences = [(index, str(entry.get(sequence_field, '')).strip().upper())
                 for index, entry in enumerate(entries)]
    sequences = [(index, sequence) for index, sequence in sequences if sequence]
    clusters = {}
    if not sequences:
        return clusters

    with tempfile.TemporaryDirectory(prefix='disorderflow_mmseqs_') as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / 'sequences.fasta'
        with open(fasta, 'w', encoding='ascii') as handle:
            for index, sequence in sequences:
                handle.write(f'>{index}\n{sequence}\n')
        prefix = tmp_path / 'clusters'
        work = tmp_path / 'work'
        command = [
            executable, 'easy-cluster', str(fasta), str(prefix), str(work),
            '--min-seq-id', str(min_identity), '-c', str(coverage), '--cov-mode', '0',
            '--threads', str(threads),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 clustering failed: {result.stderr[-1000:]}")
        with open(f'{prefix}_cluster.tsv', encoding='utf-8') as handle:
            for line in handle:
                representative, member = line.rstrip().split('\t')[:2]
                clusters[int(member)] = representative
    return clusters


def annotate_homology_clusters(entries, field_thresholds, coverage=0.8,
                                executable='mmseqs'):
    """Annotate entries in place with ``<field>_cluster_id`` values."""
    for field, threshold in field_thresholds.items():
        mapping = mmseqs_cluster(entries, field, threshold, coverage, executable)
        for index, representative in mapping.items():
            entries[index][f'{field}_cluster_id'] = f'{field}:{representative}'
    return entries


def grouped_train_val_split(entries, val_ratio=0.1, seed=2022,
                            cluster_fields=(), fixed_val_ids=(), fixed_train_ids=()):
    """Split connected homology groups without allowing any group across splits."""
    parent = list(range(len(entries)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owners = {}
    for index, entry in enumerate(entries):
        pdb_id = str(entry.get('pdb_id', entry.get('id', ''))).casefold()
        keys = [('pdb_id', pdb_id)] if pdb_id else []
        keys.extend((field, entry.get(field)) for field in cluster_fields if entry.get(field))
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    groups = {}
    for index in range(len(entries)):
        groups.setdefault(find(index), []).append(index)

    fixed = {str(value).casefold() for value in fixed_val_ids}
    fixed_train = {str(value).casefold() for value in fixed_train_ids}
    val_roots = {
        root for root, indices in groups.items()
        if any(str(entries[index].get('pdb_id', entries[index].get('id', ''))).casefold()
               in fixed for index in indices)
    }
    train_roots = {
        root for root, indices in groups.items()
        if any(str(entries[index].get('id', entries[index].get('pdb_id', ''))).casefold()
               in fixed_train for index in indices)
    }
    conflict = val_roots & train_roots
    if conflict:
        raise ValueError(f'{len(conflict)} groups are fixed to both train and validation')
    candidates = [
        root for root in groups
        if root not in val_roots and root not in train_roots
    ]
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda root: len(groups[root]))
    target = max(len(val_roots), round(len(entries) * val_ratio))
    current = sum(len(groups[root]) for root in val_roots)
    for root in candidates:
        if current >= target:
            break
        val_roots.add(root)
        current += len(groups[root])

    val_indices = {index for root in val_roots for index in groups[root]}
    train = [entry for index, entry in enumerate(entries) if index not in val_indices]
    val = [entry for index, entry in enumerate(entries) if index in val_indices]
    return train, val
