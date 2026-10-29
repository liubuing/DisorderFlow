"""同源拆分测试：验证按抗原簇分组的 train/val 隔离与固定验证集语义。

守护 disorderflow.utils.homology_split.grouped_train_val_split：
ID 大小写折叠后仍不重叠、同源簇绝不跨 train/val（防同源泄漏）、
fixed_val_ids 必须落在验证集；
仅抗体侧共享的簇（vh_cluster）允许跨拆分——只有抗原簇才是隔离边界。
"""
from disorderflow.utils.homology_split import grouped_train_val_split


# 簇隔离与 ID 规范化：'7CH5'/'7ch5' 视为同一 ID 不得分居两侧，
# 抗原簇严格不跨拆分，固定验证 ID 强制进 val——防止同源泄漏虚高评估。
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


# 隔离边界只取抗原簇：抗体可变区簇（vh_cluster）跨拆分共享是允许的——
# 防止把抗体同源性也当作隔离条件而过度切分、或反之把抗原簇当非隔离条件造成泄漏。
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
