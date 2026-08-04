import pytest

from disorderflow.utils.homology_split import grouped_train_val_split


def test_fixed_train_id_keeps_entire_group_out_of_dev():
    entries = [
        {'id': 'seen', 'pdb_id': 'p1', 'cluster': 'a'},
        {'id': 'linked', 'pdb_id': 'p2', 'cluster': 'a'},
        {'id': 'fresh', 'pdb_id': 'p3', 'cluster': 'b'},
    ]
    train, dev = grouped_train_val_split(
        entries, val_ratio=0.34, seed=1, cluster_fields=('cluster',),
        fixed_train_ids=('seen',))
    assert {item['id'] for item in train} == {'seen', 'linked'}
    assert {item['id'] for item in dev} == {'fresh'}


def test_fixed_train_and_val_conflict_fails():
    entries = [{'id': 'same', 'pdb_id': 'p1', 'cluster': 'a'}]
    with pytest.raises(ValueError, match='both train and validation'):
        grouped_train_val_split(
            entries, cluster_fields=('cluster',), fixed_val_ids=('p1',),
            fixed_train_ids=('same',))
