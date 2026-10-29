"""RCSB 候选界面扩展发现脚本测试：验证实体角色分类、发现状态机与不可覆盖性。

守护 RCSB 发现管线（candidate_interface_extension_rcsb_snapshot_v1）：
实体按描述分类为 heavy/light/unclassified、
抗体-肽复合物标记为 candidate、病毒抗原延后（deferred）而非排除、
非病毒大蛋白排除并给出理由、已有输出文件必须拒绝覆盖。
"""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

# 模块级加载被测脚本（scripts/build/discover_rcsb_candidate_interface_extension.py）。
SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "discover_rcsb_candidate_interface_extension.py"
SPEC = importlib.util.spec_from_file_location("discover_rcsb_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 辅助函数：构造最小 RCSB entry 载荷（含实验方法与分辨率）。
def make_entry(pdb_id, entities, title="", method="X-RAY DIFFRACTION", resolution=2.0):
    return {
        "rcsb_id": pdb_id,
        "struct": {"title": title},
        "rcsb_accession_info": {"initial_release_date": "2026-01-01T00:00:00Z"},
        "exptl": [{"method": method}],
        "rcsb_entry_info": {"resolution_combined": [resolution]},
        "polymer_entities": entities,
    }


# 辅助函数：构造最小 RCSB polymer entity 载荷（描述、长度、链 ID）。
def make_entity(entity_id, description, length, polymer_type="Protein"):
    return {
        "rcsb_id": entity_id,
        "rcsb_polymer_entity": {"pdbx_description": description},
        "entity_poly": {
            "rcsb_entity_polymer_type": polymer_type,
            "rcsb_sample_sequence_length": length,
            "pdbx_seq_one_letter_code_can": "A" * length,
        },
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": "1",
            "asym_ids": ["A"],
            "auth_asym_ids": ["A"],
        },
        "rcsb_cluster_membership": [],
    }


# 辅助函数：写入带 sha256 清单的快照目录，模拟已采集的 RCSB 快照。
def write_snapshot(tmp_path, entries):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entries_payload = json.dumps({"entries": entries}, indent=2).encode("ascii")
    (snapshot / "entries.json").write_bytes(entries_payload)
    import hashlib
    manifest = {
        "schema_version": "candidate_interface_extension_rcsb_snapshot_v1",
        "classification": "metadata_only",
        "files": {"entries.json": {
            "byte_count": len(entries_payload),
            "sha256": hashlib.sha256(entries_payload).hexdigest(),
        }},
    }
    (snapshot / "acquisition.json").write_text(json.dumps(manifest), encoding="ascii")
    return snapshot


NOW = lambda: datetime(2026, 9, 1, tzinfo=timezone.utc)


# 角色分类：heavy/light 描述归类正确、VHH 纳米抗体视为 heavy、无关蛋白 unclassified——
# 防止分类规则回归导致抗体链识别失败。
def test_classify_entity_roles():
    assert MODULE.classify_entity(make_entity("X_1", "Fab heavy chain", 120), "") == "heavy"
    assert MODULE.classify_entity(make_entity("X_2", "Fab light chain", 110), "") == "light"
    assert MODULE.classify_entity(make_entity("X_3", "VHH nanobody", 118), "") == "heavy"
    assert MODULE.classify_entity(make_entity("X_4", "some kinase", 300), "") == "unclassified"


# 抗体+设计肽（18 残基）复合物必须被发现为 candidate 并正确计数——
# 守护核心发现逻辑（5-50 残基肽抗原）。
def test_peptide_candidate_discovered(tmp_path):
    entities = [
        make_entity("X_1", "antibody heavy chain", 120),
        make_entity("X_2", "antibody light chain", 110),
        make_entity("X_3", "designed peptide", 18),
    ]
    snapshot = write_snapshot(tmp_path, [make_entry("1ABC", entities, "Fab in complex with peptide")])
    artifact = MODULE.discover(snapshot, tmp_path / "discovery.json", now=NOW)
    entry = artifact["entries"][0]
    assert entry["status"] == "candidate"
    assert entry["peptide_antigens"][0]["length"] == 18
    assert artifact["counts"]["antibody_peptide_candidates"] == 1


# 病毒肽抗原必须延后（deferred_viral_antigen）而非直接排除——
# 保证病毒相关候选可追溯、后续可按策略重新启用（隔离而非丢失）。
def test_viral_antigen_deferred_not_excluded(tmp_path):
    entities = [
        make_entity("X_1", "antibody heavy chain", 120),
        make_entity("X_2", "antibody light chain", 110),
        make_entity("X_3", "influenza hemagglutinin peptide", 15),
    ]
    snapshot = write_snapshot(tmp_path, [make_entry("2DEF", entities, "Fab bound to viral peptide")])
    artifact = MODULE.discover(snapshot, tmp_path / "discovery.json", now=NOW)
    entry = artifact["entries"][0]
    assert entry["status"] == "deferred_viral_antigen"
    assert artifact["counts"]["deferred_viral_antigen"] == 1
    assert artifact["counts"]["antibody_peptide_candidates"] == 0


# 非病毒大蛋白（白蛋白）必须排除且理由字段为 no_peptide_antigen_5_50——
# 防止无肽抗原的复合物污染候选池，且排除原因可审计。
def test_nonviral_protein_antigen_excluded(tmp_path):
    entities = [
        make_entity("X_1", "antibody heavy chain", 120),
        make_entity("X_2", "antibody light chain", 110),
        make_entity("X_3", "human serum albumin", 585),
    ]
    snapshot = write_snapshot(tmp_path, [make_entry("3GHI", entities, "Fab bound to albumin")])
    artifact = MODULE.discover(snapshot, tmp_path / "discovery.json", now=NOW)
    entry = artifact["entries"][0]
    assert entry["status"] == "excluded"
    assert "no_peptide_antigen_5_50" in entry["reasons"]


# 冻结不可重写：输出文件已存在时 discover 必须拒绝且原文件内容不变——
# 防止发现结果被静默覆盖破坏溯源。
def test_discovery_refuses_overwrite(tmp_path):
    snapshot = write_snapshot(tmp_path, [])
    output = tmp_path / "discovery.json"
    output.write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.discover(snapshot, output, now=NOW)
    assert output.read_text() == "sentinel"
