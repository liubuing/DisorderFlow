"""候选界面扩展参考并集构建脚本的契约测试：守护 PDB ID 规范化、快照条目 ID 的去重拒绝、
参考记录字段的正确派生，以及"只接受冻结并集基底、拒绝覆盖输出"的写入安全。"""
import importlib.util
import json
from pathlib import Path

import pytest

# 模块级加载：脚本不在包内，通过 importlib 按绝对路径导入一次供所有测试复用。
SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "build_candidate_interface_extension_reference_union.py"
SPEC = importlib.util.spec_from_file_location("ext_union", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# PDB ID 规范化：大写转小写、剥离 pdb_ 前缀与 _0 后缀、去前导零，保证跨来源 ID 可对齐。
def test_normalize_pdb_variants():
    assert MODULE.normalize_pdb("1ABC") == "1abc"
    assert MODULE.normalize_pdb("pdb_1abc_0") == "1abc"
    assert MODULE.normalize_pdb("0001abc") == "1abc"


# 快照解析：字符串与 {"identifier": ...} 两种条目形式都要规范化，且保持顺序。
def test_snapshot_entry_ids_normalizes_unique_ids(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "search_response.json").write_text(json.dumps({
        "result_set": ["1ABC", "2DEF", {"identifier": "3ghi"}],
    }), encoding="ascii")
    ids = MODULE.snapshot_entry_ids(snapshot)
    assert ids == ["1abc", "2def", "3ghi"]


# 重复拒绝：规范化后仍重复的条目必须报错，防止同一结构被重复并入参考并集。
def test_snapshot_entry_ids_rejects_duplicates(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "search_response.json").write_text(json.dumps({
        "result_set": ["1ABC", {"identifier": "1abc"}],
    }), encoding="ascii")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.snapshot_entry_ids(snapshot)


# 记录构建：reference_id 必须按 EXTR 前缀顺序编号、pdb_id 规范化、序列字段原样透传，
# 保证扩展参考记录与既有冻结参考格式严格一致。
def test_reference_records_builds_valid_records(tmp_path):
    structural = tmp_path / "structural_manifest.json"
    structural.write_text(json.dumps({
        "records": [{
            "instance": "33CI_E_D_C",
            "pdb_id": "33CI",
            "vh_sequence": "QVQLVQ",
            "vl_sequence": "DIQMTQ",
            "paired_cdr_sequence": "SSAAAAGGGGLLL",
            "cdr_h3_sequence": "AAAA",
            "antigen_sequence": "KVAELVHFL",
        }],
    }), encoding="utf-8")
    records = MODULE.reference_records(structural)
    assert records[0]["reference_id"] == "EXTR00001"
    assert records[0]["pdb_id"] == "33ci"
    assert records[0]["vh_sequence"] == "QVQLVQ"


# 冻结守护：基底文件状态不是冻结的 successor-v3 参考并集时必须拒绝构建，防止在错误基底上扩展。
def test_build_rejects_non_union_base(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"status": "something_else", "records": []}))
    with pytest.raises(ValueError, match="not a frozen successor-v3 reference union"):
        MODULE.build(base, tmp_path / "s", tmp_path / "m", tmp_path / "out.json")


# 不可覆盖：输出文件已存在时必须拒绝写入，防止误覆盖既有冻结产物。
def test_build_refuses_overwrite(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("sentinel")
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"status": MODULE.UNION_STATUS, "records": [],
                                "exact_exposed_pdb_ids": []}))
    with pytest.raises(FileExistsError):
        MODULE.build(base, tmp_path / "s", tmp_path / "m", out)
