"""同源聚类拆分的契约测试：守护固定训练 ID 把其整个同源簇隔离在 dev 之外，
以及同一 ID 同时出现在固定训练与固定验证时的冲突拒绝——保证拆分溯源无歧义。"""
import pytest

from disorderflow.utils.homology_split import grouped_train_val_split


# 簇级隔离：指定 fixed_train_ids 后，其同簇成员必须全部留在 train、另一簇进入 dev——
# 防止固定种子样本的同源伙伴泄漏进验证集。
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


# 冲突拒绝：一个 ID 同时被固定进 train 与 val 必须报错，防止拆分配置自相矛盾时静默产生脏拆分。
def test_fixed_train_and_val_conflict_fails():
    entries = [{'id': 'same', 'pdb_id': 'p1', 'cluster': 'a'}]
    with pytest.raises(ValueError, match='both train and validation'):
        grouped_train_val_split(
            entries, cluster_fields=('cluster',), fixed_val_ids=('p1',),
            fixed_train_ids=('same',))
