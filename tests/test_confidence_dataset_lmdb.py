"""置信度回归数据集（LMDB）测试：验证长度推断、只读共享环境与候选界面契约。

守护 ConfidenceRegressionDataset 数据管线：
无元数据时能从记录本身推断数据集长度、多实例共享只读 LMDB 环境且关闭互不干扰、
candidate_interface_v1 模式保留完整复合物张量与稳定性元数据并拒绝旧 schema、
CompleteGroupBatchSampler 保证整组成批（组不跨批、记录不丢、容量不足显式报错）。
"""
import pickle

import lmdb
import pytest
import torch

from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset
from disorderflow.utils.data import CompleteGroupBatchSampler
from disorderflow.utils.protein.constants import Fragment


# 数据集长度必须在缺少 metadata 时由记录键推断——防止空数据集或长度错报。
def test_confidence_dataset_infers_length_without_metadata(tmp_path):
    path = tmp_path / "confidence.lmdb"
    env = lmdb.open(str(path), map_size=1_000_000)
    with env.begin(write=True) as transaction:
        transaction.put(b"00000000", pickle.dumps({"sequence": "AA"}))
        transaction.put(b"00000001", pickle.dumps({"sequence": "AAA"}))
    env.close()
    dataset = ConfidenceRegressionDataset({"db_path": str(path)})
    assert len(dataset) == 2
    dataset.close()


# 同一 LMDB 的多个数据集实例必须共享只读环境，且其一关闭后其余仍可用——
# 防止重复打开写句柄或过早释放环境导致后续读取崩溃。
def test_confidence_datasets_share_readonly_environment(tmp_path):
    path = tmp_path / "shared.lmdb"
    env = lmdb.open(str(path), map_size=1_000_000)
    with env.begin(write=True) as transaction:
        transaction.put(b"00000000", pickle.dumps({"sequence": "AA"}))
    env.close()
    first = ConfidenceRegressionDataset({"db_path": str(path)})
    second = ConfidenceRegressionDataset({"db_path": str(path)})
    assert first._env is second._env
    first.close()
    assert len(second) == 1
    second.close()


# 辅助函数：构造符合 candidate_interface_confidence_v1 schema 的完整记录。
def _candidate_interface_entry(schema="candidate_interface_confidence_v1"):
    return {
        "schema_version": schema,
        "sequence": "AAA",
        "batch": {
            "aa": torch.tensor([0, 1, 2]),
            "generate_flag": torch.tensor([True, False, False]),
            "fragment_type": torch.tensor([
                int(Fragment.Heavy), int(Fragment.Light), int(Fragment.Antigen)]),
        },
        "af2_plddt": torch.tensor([0.8, 0.7, 0.6]),
        "af2_iptm": torch.tensor(0.5),
        "af2_pae_matrix": torch.tensor([
            [0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
        "af2_pae_normalized": False,
    }


# 候选界面数据集必须保留完整复合物契约：生成掩码、3x3 PAE 矩阵、
# 未归一化标志、整组约束与分组索引——防止截断为纯序列而丢失结构监督信号。
def test_candidate_interface_dataset_preserves_full_complex_contract(tmp_path):
    path = tmp_path / "successor"
    path.mkdir()
    with (path / "00000000.pkl").open("wb") as handle:
        pickle.dump(_candidate_interface_entry(), handle)
    dataset = ConfidenceRegressionDataset({
        "db_path": str(path), "candidate_interface_v1": True})
    batch = dataset[0]
    assert torch.equal(
        batch["generate_flag"], torch.tensor([True, False, False]))
    assert batch["af2_pae_matrix"].shape == (3, 3)
    assert batch["af2_pae_normalized"].item() is False
    assert dataset.requires_complete_groups is True
    assert dataset.group_indices == [[0]]


# 复测稳定性元数据（样本权重、iptm/plddt/PAE 的 std 与 SEM）必须原样透传到 batch——
# 防止不确定性加权训练因字段丢失而静默退化为等权。
def test_candidate_interface_dataset_attaches_stability_metadata(tmp_path):
    path = tmp_path / "weighted_successor"
    path.mkdir()
    entry = _candidate_interface_entry()
    entry.update({
        "confidence_sample_weight": 0.4,
        "af2_iptm_std": 0.06,
        "af2_candidate_plddt_std": 0.03,
        "af2_interface_pae_normalized_std": 0.08,
        "af2_iptm_sem": 0.024,
        "af2_candidate_plddt_sem": 0.012,
        "af2_interface_pae_normalized_sem": 0.032,
    })
    with (path / "00000000.pkl").open("wb") as handle:
        pickle.dump(entry, handle)
    dataset = ConfidenceRegressionDataset({
        "db_path": str(path), "candidate_interface_v1": True})
    batch = dataset[0]
    assert batch["confidence_sample_weight"].item() == pytest.approx(0.4)
    assert batch["af2_iptm_std"].item() == pytest.approx(0.06)
    assert batch["af2_candidate_plddt_std"].item() == pytest.approx(0.03)
    assert batch["af2_interface_pae_normalized_std"].item() == pytest.approx(0.08)
    assert batch["af2_iptm_sem"].item() == pytest.approx(0.024)
    assert batch["af2_candidate_plddt_sem"].item() == pytest.approx(0.012)
    assert batch["af2_interface_pae_normalized_sem"].item() == pytest.approx(0.032)


# 旧 schema（整型版本号）必须显式拒绝——防止新旧数据格式混用导致字段语义错乱。
def test_candidate_interface_dataset_rejects_legacy_schema(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    with (path / "00000000.pkl").open("wb") as handle:
        pickle.dump(_candidate_interface_entry(schema=1), handle)
    dataset = ConfidenceRegressionDataset({
        "db_path": str(path), "candidate_interface_v1": True})
    with pytest.raises(ValueError, match="successor dataset schema"):
        dataset[0]


# 整组成批采样器：每个 batch 恰为一个完整组且覆盖全部记录——
# 防止组内样本跨批破坏复合物级监督（如成对对比损失）。
def test_complete_group_sampler_keeps_every_group_and_record():
    dataset = type("GroupedDataset", (), {
        "group_indices": [list(range(15)), list(range(15, 32)), list(range(32, 48))]
    })()
    sampler = CompleteGroupBatchSampler(
        dataset, max_batch_records=20, shuffle=False)
    batches = list(sampler)
    assert batches == dataset.group_indices
    assert sorted(index for batch in batches for index in batch) == list(range(48))


# 组大小超过批容量时必须显式报错而非静默截断——防止不完整组进入训练。
def test_complete_group_sampler_rejects_insufficient_capacity():
    dataset = type("GroupedDataset", (), {
        "group_indices": [list(range(17))]
    })()
    with pytest.raises(ValueError, match="exceeds batch capacity"):
        CompleteGroupBatchSampler(dataset, max_batch_records=4)


# shuffle 模式下不同 epoch 的批顺序必须不同但均覆盖全部记录——
# 防止种子固定导致每轮顺序完全重复（失去打乱意义）。
def test_complete_group_sampler_advances_shuffle_epoch():
    dataset = type("GroupedDataset", (), {
        "group_indices": [[index] for index in range(8)]
    })()
    sampler = CompleteGroupBatchSampler(
        dataset, max_batch_records=2, shuffle=True, seed=11)
    first = list(sampler)
    second = list(sampler)
    assert first != second
    assert sorted(index for batch in first for index in batch) == list(range(8))
    assert sorted(index for batch in second for index in batch) == list(range(8))
