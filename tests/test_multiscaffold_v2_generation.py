"""多支架确认集 v2 生成的契约测试：守护规范化结构的三链连续编号（H/L/P 从 1 起）、
失败槽位也必须完整保留（8 个 attempt 逐一记录），以及重设计只改冻结的 H3 索引位置。"""
import json
from pathlib import Path

from Bio.PDB import PDBParser

from scripts.generate_multiscaffold_confirmatory_v2 import (
    attempt_rows,
    canonicalize_structure,
    full_heavy,
)

ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：读取留出清单中的第一个组件作为代表性测试样本。
def first_component():
    holdout = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/holdout_manifest.json"
    ).read_text())
    return holdout["components"][0]


# 规范化结构：三链重命名为 H/L/P，残基数与序列长度一致且编号从 1 连续——
# 防止链名或编号错位破坏下游打分输入。
def test_canonical_structure_has_contiguous_role_chains(tmp_path):
    component = first_component()
    representative = component["representative"]
    output = tmp_path / "canonical.pdb"
    canonicalize_structure(representative, output)
    model = PDBParser(QUIET=True).get_structure("canonical", output)[0]
    expected_lengths = {
        "H": len(representative["heavy_sequence"]),
        "L": len(representative["light_sequence"]),
        "P": len(representative["antigen_sequence"]),
    }
    assert {chain.id for chain in model} == {"H", "L", "P"}
    for chain_id, expected_length in expected_lengths.items():
        residues = list(model[chain_id].get_residues())
        assert len(residues) == expected_length
        assert [residue.id[1] for residue in residues] == list(
            range(1, expected_length + 1))


# 失败留痕：生成失败时 8 个槽位必须全部以独立 attempt_id 记录 status=failed 与错误信息——
# 防止失败槽位被静默吞掉导致有效样本数虚增。
def test_attempt_rows_preserve_every_failed_slot():
    component = first_component()
    rows = attempt_rows(
        component, "proteinmpnn", 2609, 8, [], "generation failed",
        {"input_cif_sha256": "a", "canonical_pdb_sha256": "b",
         "checkpoint_or_model_sha256": "c"}, 1.25)
    assert len(rows) == 8
    assert len({row["attempt_id"] for row in rows}) == 8
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["error"] == "generation failed" for row in rows)


# H3 限定：full_heavy 在 H3 索引处写入替换序列，其余位置逐残基不变——防止重设计越出冻结区间。
def test_full_heavy_changes_only_frozen_h3_positions():
    representative = first_component()["representative"]
    indices = representative["h3_heavy_indices_zero_based"]
    replacement = "A" * len(indices)
    designed = full_heavy(representative["heavy_sequence"], indices, replacement)
    assert "".join(designed[index] for index in indices) == replacement
    assert all(
        designed[index] == amino_acid
        for index, amino_acid in enumerate(representative["heavy_sequence"])
        if index not in set(indices))
