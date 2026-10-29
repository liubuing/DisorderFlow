"""StateContrast 真实数据接入测试：验证 PDB 集成解析、几何就绪与实验数据 schema 校验。

守护 statecontrast 真实数据管线：
多模型 PDB 必须全模型解析并审计所选链的 SEQRES 覆盖、
无抗体 pose 时接触打分必须阻断（geometry_ready 但不冒充可打接触分）、
结合状态矩阵保留状态/重复/单位/删失信息、
HDX 摄取数据拒绝伪造单位——防止臆造或残缺实验数据进入监督。
"""
import csv

import pytest
from statecontrast_experimental_data import (
    SCHEMA_VERSION,
    ExperimentalDataValidationError,
    load_binding_state_matrix_csv,
    load_hdx_peptide_uptake_csv,
)
from statecontrast_real_ensembles import (
    StructureValidationError,
    parse_pdb_ensemble,
    real_ensemble_mode_status,
)


# 辅助函数：生成一行固定格式的 PDB ATOM 记录（可加模型偏移构造多模型）。
def _atom(serial, atom, residue, chain, residue_number, x, model_offset=0):
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain:1s}{residue_number:4d}    "
        f"{x + model_offset:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {atom[0]:>2s}  \n"
    )


# 辅助函数：构造含两个 MODEL 的最小 PDB 集成文本（A 链 AG + H 链 TYR10）。
def _multimodel_pdb():
    lines = ["SEQRES   1 A    2  ALA GLY\n", "MODEL        1\n"]
    lines.extend([
        _atom(1, "N", "ALA", "A", 1, 1.0),
        _atom(2, "CA", "ALA", "A", 1, 1.5),
        _atom(3, "CA", "GLY", "A", 2, 2.5),
        _atom(4, "CA", "TYR", "H", 10, 4.0),
        "ENDMDL\n",
        "MODEL        2\n",
        _atom(5, "N", "ALA", "A", 1, 1.0, 10),
        _atom(6, "CA", "ALA", "A", 1, 1.5, 10),
        _atom(7, "CA", "GLY", "A", 2, 2.5, 10),
        _atom(8, "CA", "TYR", "H", 10, 4.0, 10),
        "ENDMDL\n",
    ])
    return "".join(lines)


# 全模型解析与序列审计：两个模型都被选中、链序列与期望一致、SEQRES 覆盖 100%——
# 防止只取第一个模型或跳过序列核对。
def test_pdb_adapter_parses_all_models_and_audits_selected_chain(tmp_path):
    path = tmp_path / "ensemble.pdb"
    path.write_text(_multimodel_pdb(), encoding="utf-8")

    ensemble = parse_pdb_ensemble(path, chain="A", expected_sequence="AG")

    assert ensemble["selected_models"] == [1, 2]
    assert ensemble["selected_chains"] == ["A"]
    assert ensemble["conformer_count"] == 2
    assert ensemble["contacts"] is None
    audit = ensemble["conformers"][0]["chains"][0]["sequence_audit"]
    assert audit["observed_sequence"] == "AG"
    assert audit["seqres_coverage"] == 1.0
    assert audit["expected_sequence_exact"] is True


# 模型选择与校验：可按编号选模型；无坐标文件必须显式报错——防止空结构静默通过。
def test_pdb_adapter_selects_model_and_fails_for_invalid_structure(tmp_path):
    path = tmp_path / "ensemble.pdb"
    path.write_text(_multimodel_pdb(), encoding="utf-8")
    assert parse_pdb_ensemble(path, model=2, chain="A")["selected_models"] == [2]

    invalid = tmp_path / "invalid.pdb"
    invalid.write_text("HEADER    NO COORDINATES\n", encoding="utf-8")
    with pytest.raises(StructureValidationError, match="no recognized protein coordinates"):
        parse_pdb_ensemble(invalid)


# 几何就绪分级：结构集成可就绪但缺抗体 pose 时接触打分必须被显式阻断——
# 防止在无对接姿态时伪造接触打分。
def test_geometry_readiness_blocks_contacts_without_explicit_antibody_pose(tmp_path):
    path = tmp_path / "ensemble.pdb"
    path.write_text(_multimodel_pdb(), encoding="utf-8")
    config = {"negative_states": {"mode": "structure_ensemble", "ensembles": [{
        "name": "state", "pdb": path.name, "chain": "A", "state_type": "fibril"
    }]}}

    status = real_ensemble_mode_status(config, tmp_path)

    assert status["geometry_status"] == "geometry_ready"
    assert status["contact_scoring_status"] == "blocked_without_pose"
    assert status["status"] == "geometry_ready_but_contact_scoring_blocked_without_pose"
    assert status["entries"][0]["geometry"]["contacts"] is None


# 辅助函数：按给定字段写 CSV fixture。
def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# 结合状态矩阵：状态、重复、单位与删失（right-censoring）必须逐列保留——
# 防止删失测量被当作普通点估计。
def test_binding_state_matrix_preserves_states_replicates_units_and_censoring(tmp_path):
    path = tmp_path / "binding.csv"
    fields = [
        "schema_version", "experiment_id", "assay_type", "binder_id", "state_id",
        "replicate_id", "metric", "value", "unit", "censoring", "quality_flag",
        "evidence_tier",
    ]
    base = {
        "schema_version": SCHEMA_VERSION, "experiment_id": "exp-1", "assay_type": "SPR",
        "binder_id": "ab-1", "replicate_id": "r1", "metric": "kd", "unit": "nM",
        "quality_flag": "pass", "evidence_tier": "tier_1_primary",
    }
    _write_csv(path, fields, [
        {**base, "state_id": "monomer", "value": "2.5", "censoring": "none"},
        {**base, "state_id": "fibril", "value": "100", "censoring": "right"},
    ])

    rows = load_binding_state_matrix_csv(path)

    assert [row["state_id"] for row in rows] == ["monomer", "fibril"]
    assert rows[0]["value"] == 2.5
    assert rows[1]["censoring"] == "right"
    assert rows[1]["unit"] == "nM"


# HDX 校验：肽段起止解析为整数；伪造的 uptake 单位必须拒绝——
# 防止单位混乱的数据污染监督信号。
def test_hdx_loader_validates_peptide_uptake_and_rejects_made_up_units(tmp_path):
    path = tmp_path / "hdx.csv"
    fields = [
        "schema_version", "experiment_id", "protein_id", "state_id", "peptide_id",
        "peptide_start", "peptide_end", "timepoint", "timepoint_unit", "uptake_value",
        "uptake_unit", "replicate_id", "censoring", "quality_flag", "evidence_tier",
    ]
    row = {
        "schema_version": SCHEMA_VERSION, "experiment_id": "hdx-1", "protein_id": "p1",
        "state_id": "bound", "peptide_id": "pep-1", "peptide_start": "3",
        "peptide_end": "8", "timepoint": "30", "timepoint_unit": "s",
        "uptake_value": "1.4", "uptake_unit": "Da", "replicate_id": "r2",
        "censoring": "none", "quality_flag": "warning",
        "evidence_tier": "tier_2_processed",
    }
    _write_csv(path, fields, [row])
    assert load_hdx_peptide_uptake_csv(path)[0]["peptide_start"] == 3

    row["uptake_unit"] = "invented"
    _write_csv(path, fields, [row])
    with pytest.raises(ExperimentalDataValidationError, match="invalid uptake_unit"):
        load_hdx_peptide_uptake_csv(path)
