from disorderflow.utils.homology_split import grouped_train_val_split


def test_grouped_split_is_casefolded_and_cluster_disjoint():
    entries = [
        {'id': '7CH5', 'antigen_cluster': 'ag:1'},
        {'id': '7ch5', 'antigen_cluster': 'ag:2'},
        {'id': 'other', 'antigen_cluster': 'ag:1'},
        {'id': 'validation', 'antigen_cluster': 'ag:3'},
    ]
    train, val = grouped_train_val_split(
        entries, val_ratio=0.25, seed=1,
        cluster_fields=['antigen_cluster'], fixed_val_ids=['validation'])

    train_ids = {entry['id'].casefold() for entry in train}
    val_ids = {entry['id'].casefold() for entry in val}
    train_clusters = {entry['antigen_cluster'] for entry in train}
    val_clusters = {entry['antigen_cluster'] for entry in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_clusters.isdisjoint(val_clusters)
    assert 'validation' in val_ids


def test_audit_only_antibody_clusters_do_not_bridge_antigen_groups():
    entries = [
        {'id': 'a', 'antigen_cluster': 'ag:1', 'vh_cluster': 'vh:shared'},
        {'id': 'b', 'antigen_cluster': 'ag:1', 'vh_cluster': 'vh:other'},
        {'id': 'c', 'antigen_cluster': 'ag:2', 'vh_cluster': 'vh:shared'},
        {'id': 'd', 'antigen_cluster': 'ag:2', 'vh_cluster': 'vh:third'},
    ]
    train, val = grouped_train_val_split(
        entries, val_ratio=0.5, seed=1,
        cluster_fields=['antigen_cluster'])

    assert {entry['antigen_cluster'] for entry in train}.isdisjoint(
        {entry['antigen_cluster'] for entry in val})
    assert {entry['vh_cluster'] for entry in train} & {
        entry['vh_cluster'] for entry in val
    } == {'vh:shared'}
