"""数据工厂（data_factory）的契约测试：守护反事实样本"配对但不伪造结合标签"、构建的确定性与
谱系可验证性、源记录去重先于反事实扩张，以及外部评估源禁入训练，还有六条 IMGT CDR 的链一致反事实。"""
import gzip
import json
import pickle

import lmdb
import pytest

from disorderflow.data_factory import build_dataset, generate_counterfactuals, verify_dataset


# 辅助函数：构造最小抗体观测记录（VH/VL/H3/抗原序列）。
def sample_raw(pdb_id="1abc"):
    return {
        "pdb_id": pdb_id,
        "vh_sequence": "QVQLVESCARDRGFDYWGGG",
        "vl_sequence": "DIQMTQSPSS",
        "cdr_h3_sequence": "CARDRGFDYW",
        "antigen_sequence": "DAEFRHDSGYE",
    }


# 辅助函数：把记录写入 LMDB 并落盘 ID 索引，模拟 phase3 数据源。
def write_phase3(path, records):
    env = lmdb.open(str(path), map_size=1024 * 1024, subdir=False)
    identifiers = []
    with env.begin(write=True) as txn:
        for record in records:
            identifier = record["pdb_id"]
            identifiers.append(identifier)
            txn.put(identifier.encode(), pickle.dumps(record))
    env.close()
    with open(str(path) + "-ids", "wb") as handle:
        pickle.dump(identifiers, handle)


# 辅助函数：构造单一数据源的 source 元数据（类型/路径/拆分/证据层级）。
def source(path, split="train", name="phase3_test"):
    return {
        "name": name,
        "type": "phase3_lmdb",
        "path": path.name,
        "split": split,
        "release": "test-v1",
        "license": "test-only",
        "evidence_tier": "experimental_structure",
        "usage": split,
    }


# 辅助函数：组装数据工厂配置（种子/分片大小/反事实操作/数据源列表）。
def config(sources):
    return {
        "dataset_name": "test_factory",
        "seed": 17,
        "shard_size": 5,
        "counterfactuals": {"operations": [
            "cdr_single_mutation", "cdr_composition_shuffle",
            "antigen_composition_shuffle",
        ]},
        "sources": sources,
    }


# 反事实契约：变体与观测样本共享 group_id（配对比较用），但结合标签必须为 None、
# 证据层级标记为 synthetic_weak——防止合成样本被误当真实结合/非结合数据训练。
def test_counterfactuals_are_matched_but_not_false_nonbinder_labels():
    from disorderflow.data_factory import make_observed_record

    observed = make_observed_record(sample_raw(), source(type("P", (), {"name": "x"})()), "train")
    variants = generate_counterfactuals(observed, [
        "cdr_single_mutation", "cdr_composition_shuffle",
        "antigen_composition_shuffle",
    ], seed=3)

    assert len(variants) == 3
    assert all(item["group_id"] == observed["group_id"] for item in variants)
    assert all(item["targets"]["binding_label"] is None for item in variants)
    assert all(item["evidence"]["tier"] == "synthetic_weak" for item in variants)
    assert variants[0]["antibody"]["cdr_h3"] != observed["antibody"]["cdr_h3"]


# 确定性与可验证性：同配置两次构建得到相同 canonical_fingerprint 与逐分片 sha256，
# 审计通过且谱系零错误，首条记录的 lineage.operation 为 observed_bound——保证产物可复现可审计。
def test_factory_build_is_deterministic_and_verifiable(tmp_path):
    database = tmp_path / "train.lmdb"
    write_phase3(database, [sample_raw("1abc"), sample_raw("2def")])
    settings = config([source(database)])
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest_a = build_dataset(settings, tmp_path, first)
    manifest_b = build_dataset(settings, tmp_path, second)
    audit = verify_dataset(first)

    assert manifest_a["records"] == 8
    assert manifest_a["groups"] == 2
    assert manifest_a["canonical_fingerprint"] == manifest_b["canonical_fingerprint"]
    assert [item["sha256"] for item in manifest_a["shards"]] == [
        item["sha256"] for item in manifest_b["shards"]]
    assert audit["status"] == "pass"
    assert audit["lineage_errors"] == 0
    with gzip.open(first / manifest_a["shards"][0]["path"], "rt", encoding="ascii") as handle:
        record = json.loads(next(handle))
    assert record["lineage"]["operation"] == "observed_bound"


# 去重契约：完全相同的源记录只保留一份再扩张反事实，防止重复样本膨胀数据集并泄漏进验证。
def test_factory_deduplicates_source_records_before_counterfactual_expansion(tmp_path):
    database = tmp_path / "train.lmdb"
    records = [sample_raw("same"), sample_raw("same")]
    write_phase3(database, records)
    manifest = build_dataset(config([source(database)]), tmp_path, tmp_path / "output")

    assert manifest["groups"] == 1
    assert manifest["records"] == 4
    assert manifest["duplicate_source_records_skipped"] == 1


# 边界守护：外部评估数据源（如 caid2_binding）禁止进入训练构建，必须显式报错而非静默混入。
def test_factory_rejects_external_evaluation_sources(tmp_path):
    database = tmp_path / "train.lmdb"
    write_phase3(database, [sample_raw()])
    forbidden = source(database, name="caid2_binding")
    with pytest.raises(ValueError, match="External evaluation source"):
        build_dataset(config([forbidden]), tmp_path, tmp_path / "output")


# 六 CDR 覆盖：IMGT 注释齐全时，六条 CDR 各自的单点突变反事实都要生成，
# 且突变后 CDR 序列仍真实出现在对应链（VH/VL）中、链序列确有改变——防止链-CDR 不一致的伪样本。
def test_all_six_imgt_cdrs_generate_matched_chain_consistent_counterfactuals():
    from disorderflow.data_factory import make_observed_record

    cdrs = {
        "H1": "GFTFSSYT", "H2": "ISSGGAYT", "H3": "CARDRGFDYW",
        "L1": "QSVSSSY", "L2": "GAS", "L3": "LQIYNMPIT",
    }
    raw = {
        "pdb_id": "six-cdr",
        "vh_sequence": "QVQLGFTFSSYTWVRQISSGGAYTKGRFCARDRGFDYWGQGT",
        "vl_sequence": "DIQMQSVSSSYWYQQGASRFSGLQIYNMPITFGQG",
        "cdr_h3_sequence": cdrs["H3"],
        "cdrs": cdrs,
        "antigen_sequence": "DAEFRHDSGYE",
        "cdr_annotation": {"scheme": "IMGT", "source": "official_precomputed"},
    }
    metadata = source(type("P", (), {"name": "x"})())
    metadata["require_all_six_cdrs"] = True
    observed = make_observed_record(raw, metadata, "train", "statecontrast_factory_v2")
    variants = generate_counterfactuals(observed, [
        "cdr_single_mutation_each", "cdr_composition_shuffle_each",
        "antigen_composition_shuffle",
    ], seed=9)

    mutation_variants = [
        item for item in variants if item["lineage"]["operation"].startswith("cdr_single")]
    assert observed["antibody"]["cdr_annotation"]["scheme"] == "IMGT"
    assert set(observed["antibody"]["cdrs"]) == set(cdrs)
    assert len(mutation_variants) == 6
    for variant in mutation_variants:
        cdr_name = variant["lineage"]["operation_detail"]["cdr"]
        chain = "vh" if cdr_name.startswith("H") else "vl"
        assert variant["antibody"]["cdrs"][cdr_name] in variant["antibody"][chain]
        assert variant["antibody"][chain] != observed["antibody"][chain]
