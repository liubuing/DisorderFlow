"""无序度监督管线的契约测试：守护证据归一化的掩码语义、低置信度教师不得覆盖实验证据、
聚类跨拆分审计的拒绝、冻结产物对 mask/confidence 的保真、DisProt 未注释区不当有序处理、
CAID 清单对快照缺失条目的 unresolved 标记，以及链合并时的无监督字段默认值。"""
import json
import pickle

import numpy as np
import pytest
import torch

from disorderflow.datasets.disorder_augmented import load_disorder_lookup
from disorderflow.disorder_supervision import (
    audit_cluster_splits,
    merge_evidence,
    normalize_evidence,
)
from disorderflow.utils.transforms.merge import MergeChains
from scripts.build.build_caid_cluster_manifest import build as build_caid_cluster_manifest
from scripts.build.build_disorder_supervision import build_artifact
from scripts.build.prepare_disprot_supervision import experimental_profile


# 辅助函数：构造最小监督证据记录（来源/逐残基值/可选置信度/聚类 ID）。
def evidence(source, values, confidence=None, cluster="c1"):
    record = {
        "id": "sample",
        "sequence": "AAA",
        "values": values,
        "source": source,
        "cluster_id": cluster,
    }
    if confidence is not None:
        record["confidence"] = confidence
    return record


# 归一化语义：mask 外的残基视为未知，其置信度必须压为零——防止未注释位置冒充确定的无序/有序标签。
def test_normalize_evidence_masks_unknown_residues():
    record = evidence("disprot_experimental", [1.0, 0.0, 0.0])
    record["mask"] = [True, True, False]
    result = normalize_evidence(record)
    assert result.mask.tolist() == [True, True, False]
    assert result.confidence.tolist() == [1.0, 1.0, 0.0]


# 证据层级：合并时实验证据（disprot_experimental）必须压过外部预测器教师，
# 防止弱监督信号反向污染实验金标准。
def test_low_confidence_teacher_cannot_override_experiment():
    experimental = evidence("disprot_experimental", [1.0, 0.0, 1.0])
    teacher = evidence("external_predictor", [0.0, 1.0, 0.0])
    merged = merge_evidence([experimental, teacher])
    assert np.all(merged.values[[0, 2]] > 0.5)
    assert merged.values[1] < 0.5


# 聚类隔离：同一 cluster_id 出现在 train 与 external_test 中审计必须判 invalid——
# 防止同源序列跨拆分泄漏造成外部评估虚高。
def test_cluster_audit_rejects_cross_split_cluster():
    row = {"cluster_id": "cluster-a", "sequence": "AAAA"}
    audit = audit_cluster_splits({"train": [row], "external_test": [row]})
    assert not audit["valid"]
    assert any("cluster" in error for error in audit["errors"])


# 留出守护：构建产物时若输入包含被留出的聚类（held-out）必须报错，保证冻结产物不含评测聚类。
def test_builder_rejects_held_out_cluster(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(evidence("disprot_experimental", [1, 0, 1])) + "\n")
    with pytest.raises(ValueError, match="Held-out clusters"):
        build_artifact([path], "train", tmp_path / "artifact.pkl", ["c1"])


# 产物保真：v3 产物加载后 mask 全真、confidence 等于 (0.5+0.6)/2 的按值插值——
# 冻结产物的监督掩码与置信度必须逐位可复现。
def test_v3_artifact_preserves_mask_and_confidence(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(evidence("rmsf_proxy", [0.2, 0.8, 0.4])) + "\n")
    output = tmp_path / "artifact.pkl"
    build_artifact([path], "train", output)
    loaded = load_disorder_lookup(output, expected_split="train", require_envelope=True)
    assert loaded["sample"]["mask"].all()
    assert np.allclose(loaded["sample"]["confidence"], 0.55)


# 加载过滤：expected_ids 中无标签的 ID 必须被过滤掉，防止大结构库中未标注条目被当作监督样本。
def test_v3_artifact_can_filter_a_larger_structure_store(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(evidence("disprot_experimental", [1, 0, 1])) + "\n")
    output = tmp_path / "artifact.pkl"
    build_artifact([path], "train", output)
    loaded = load_disorder_lookup(
        output, expected_split="train", expected_ids=["sample", "unlabelled"],
        require_envelope=True)
    assert set(loaded) == {"sample"}


# 未注释≠有序：DisProt 中只有实验支持（EXP/IDA）的 order 区才记为有序，
# 未注释位置 mask 为 False 且值为 0——防止"缺证据"被误读为"确定有序"。
def test_disprot_profile_does_not_treat_unannotated_as_ordered():
    entry = {
        "sequence": "AAAAAA",
        "regions": [
            {"term_namespace": "Structural state", "term_name": "disorder",
             "ec_go": "EXP", "ec_id": "ECO:1", "start": 2, "end": 3},
            {"term_namespace": "Structural state", "term_name": "order",
             "ec_go": "IDA", "ec_id": "ECO:2", "start": 5, "end": 5},
        ],
    }
    values, mask, _ = experimental_profile(entry)
    assert mask == [False, True, True, False, True, False]
    assert values == [0, 1, 1, 0, 0, 0]


# CAID 清单：快照中缺失的条目必须显式标记为 unresolved 且进审计，防止静默丢弃破坏账目守恒。
def test_caid_manifest_marks_missing_snapshot_entries_unresolved(tmp_path):
    entries = tmp_path / "entries.json"
    entries.write_text(json.dumps({"data": [
        {"disprot_id": "DP1", "uniref50": "UniRef50_A", "acc": "P1"},
    ]}))
    sequences = tmp_path / "sequences.fasta"
    sequences.write_text(">DP1\nAAA\n>DP2\nAAA\n")
    output = tmp_path / "clusters.json"
    audit_path = tmp_path / "clusters.audit.json"

    audit = build_caid_cluster_manifest(entries, sequences, output, audit_path)

    manifest = json.loads(output.read_text())
    assert manifest == {"DP1": "UniRef50_A", "DP2": "unresolved:DP2"}
    assert audit["resolved_count"] == 1
    assert audit["unresolved_ids"] == ["DP2"]


# 默认字段：MergeChains 对缺失的无监督字段必须填充零值（监督掩码全 False、置信度全 0），
# 防止共享 batch 时字段缺失引发隐式广播错误。
def test_merge_chains_fills_unsupervised_disorder_fields():
    transform = MergeChains()
    data = {"aa": torch.zeros(3, dtype=torch.long)}
    assert not transform._data_attr(data, "disorder_supervision_mask").any()
    assert transform._data_attr(data, "disorder_confidence").sum() == 0
