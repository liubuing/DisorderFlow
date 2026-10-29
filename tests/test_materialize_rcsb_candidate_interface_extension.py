"""RCSB 候选界面扩展物化的契约测试：守护角色链按标题描述划分（非蛋白肽排除）、
最小距离计算的数值与分块正确性、组合排名的字典序与确定性平局处理，
以及物化输出"已存在拒绝、可断点续跑"的写入安全。"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# 模块级加载：物化脚本不在包内，通过 importlib 按路径导入一次供所有测试复用。
SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "materialize_rcsb_candidate_interface_extension.py"
SPEC = importlib.util.spec_from_file_location("materialize_rcsb_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 辅助函数：构造最小 RCSB 聚合实体记录（描述/长度/链 ID/聚合类型）。
def make_entity(entity_id, description, length, chains, polymer_type="Protein"):
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
            "asym_ids": chains,
            "auth_asym_ids": chains,
        },
        "rcsb_cluster_membership": [],
    }


# 角色划分：按描述关键词把聚合物实体分为重链/轻链/肽三组，防止 MHC 等辅助链被误判为肽。
def test_role_chains_splits_auth_chain_ids():
    entry = {
        "struct": {"title": "Fab in complex with peptide"},
        "polymer_entities": [
            make_entity("X_1", "antibody heavy chain", 120, ["E"]),
            make_entity("X_2", "antibody light chain", 110, ["D"]),
            make_entity("X_3", "MAGE-A4 peptide", 10, ["C"]),
            make_entity("X_4", "MHC class I alpha", 280, ["A"]),
            make_entity("X_5", "beta-2 microglobulin", 100, ["B"]),
        ],
    }
    heavy, light, peptide = MODULE.role_chains(entry, "Fab in complex with peptide")
    assert heavy == ["E"]
    assert light == ["D"]
    assert peptide == ["C"]


# 聚合类型过滤：RNA 适配体等非蛋白"肽"必须被排除（peptide 为空），防止核酸配体混入界面数据集。
def test_role_chains_ignores_nonprotein_peptides():
    entry = {
        "struct": {"title": ""},
        "polymer_entities": [
            make_entity("X_1", "antibody heavy chain", 120, ["H"]),
            make_entity("X_2", "antibody light chain", 110, ["L"]),
            make_entity("X_3", "RNA aptamer", 24, ["P"], polymer_type="RNA"),
        ],
    }
    heavy, light, peptide = MODULE.role_chains(entry, "")
    assert heavy == ["H"]
    assert light == ["L"]
    assert peptide == []


# 距离几何：最近距离按逐对计算（3.0），空数组返回 inf 而非崩溃——防止空链导致物化中断。
def test_min_distance_basic_and_empty():
    left = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    right = np.array([[3.0, 0.0, 0.0]])
    assert MODULE.min_distance(left, right) == pytest.approx(3.0)
    assert MODULE.min_distance(np.zeros((0, 3)), right) == float("inf")


# 分块计算：大点集按 block 分块后结果不变，防止大结构触发内存爆炸。
def test_min_distance_blocks_large_arrays():
    left = np.tile([[0.0, 0.0, 0.0]], (1000, 1))
    right = np.tile([[1.0, 0.0, 0.0]], (1000, 1))
    assert MODULE.min_distance(left, right, block=64) == pytest.approx(1.0)


# 排名方向：接触 H3 位置更多的组合排名值更小（更优），防止排序方向写反选中弱界面。
def test_combo_rank_prefers_more_h3_contacts():
    weaker = {
        "n_contacting_h3_positions": 1,
        "n_h3_antigen_residue_contacts": 2,
        "minimum_h3_antigen_distance": 3.0,
        "minimum_antibody_peptide_distance": 3.0,
        "heavy_chain_id": "A", "light_chain_id": "B", "peptide_chain_id": "C",
    }
    stronger = dict(weaker, n_contacting_h3_positions=3, heavy_chain_id="Z")
    assert MODULE.combo_rank(stronger) < MODULE.combo_rank(weaker)


# 确定性平局：指标完全相同时按链 ID 字典序破平局；缺距离（None）必须排在任何有值组合之后——
# 防止平局抖动或缺失值被当作最优。
def test_combo_rank_breaks_ties_deterministically():
    base = {
        "n_contacting_h3_positions": 2,
        "n_h3_antigen_residue_contacts": 4,
        "minimum_h3_antigen_distance": 3.5,
        "minimum_antibody_peptide_distance": 3.0,
        "heavy_chain_id": "A", "light_chain_id": "B", "peptide_chain_id": "C",
    }
    other = dict(base, heavy_chain_id="B")
    assert MODULE.combo_rank(base) < MODULE.combo_rank(other)
    none_distance = dict(base, minimum_h3_antigen_distance=None)
    assert MODULE.combo_rank(none_distance) > MODULE.combo_rank(base)


# 不可覆盖：结构清单已存在时物化必须拒绝，防止既有冻结产物被重建覆盖。
def test_materialize_refuses_existing_manifest(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({"classification": "metadata_only"}))
    output = tmp_path / "structures"
    (output / "pdb").mkdir(parents=True)
    (output / "structural_manifest.json").write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.materialize(discovery, output)


# 断点续跑：允许在无清单时重新物化，但快照目录缺失必须报 FileNotFoundError——
# 保证恢复路径存在且依赖可见。
def test_materialize_allows_resume_without_manifest(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({
        "classification": "metadata_only",
        "entries": [],
        "snapshot_dir": "data/does-not-exist",
    }))
    output = tmp_path / "structures"
    (output / "pdb").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        MODULE.materialize(discovery, output)
