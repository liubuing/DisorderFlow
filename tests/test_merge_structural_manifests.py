"""结构清单合并脚本测试：验证去重合并、重复实例拒绝与不可覆盖。

守护 scripts/build/merge_structural_manifests.py 的清单合并溯源：
多个清单合并为唯一实例集合、重复 instance 必须显式拒绝（防双计）、
输出已存在必须拒绝覆盖——防止合并产物被静默重写或实例重复膨胀统计。
"""
import importlib.util
import json
from pathlib import Path

import pytest

# 模块级加载被测脚本（scripts/build/merge_structural_manifests.py）。
SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "merge_structural_manifests.py"
SPEC = importlib.util.spec_from_file_location("merge_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 辅助函数：写入含给定 instance 记录（序列、接触、分辨率字段）的清单文件。
def make_manifest(path, instances):
    records = []
    for instance in instances:
        records.append({
            "instance": instance,
            "pdb_id": instance.split("_")[0],
            "vh_sequence": "QVQLVQ",
            "vl_sequence": "DIQMTQ",
            "paired_cdr_sequence": "SSAAAAGGGGLLL",
            "cdr_h3_sequence": "AAAA",
            "antigen_sequence": "KVAELVHFL",
            "n_contacting_h3_positions": 1,
            "n_h3_antigen_residue_contacts": 2,
            "resolution": 2.0,
        })
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


# 合并语义：不同清单的唯一实例全部并入且计数正确——防止合并丢记录或重复计入。
def test_merge_combines_unique_records(tmp_path):
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    make_manifest(m1, ["1ABC_H_L_P"])
    make_manifest(m2, ["2DEF_H_L_P"])
    merged = MODULE.merge([m1, m2], tmp_path / "out.json")
    assert merged["counts"]["total_records"] == 2
    assert sorted(r["instance"] for r in merged["records"]) == ["1ABC_H_L_P", "2DEF_H_L_P"]


# 重复防护：相同 instance 出现在两个清单必须抛错——防止同一结构被双计进统计。
def test_merge_rejects_duplicate_instances(tmp_path):
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    make_manifest(m1, ["1ABC_H_L_P"])
    make_manifest(m2, ["1ABC_H_L_P"])
    with pytest.raises(ValueError, match="Duplicate instance"):
        MODULE.merge([m1, m2], tmp_path / "out.json")


# 冻结不可重写：输出已存在必须拒绝且原文件不动——防止合并结果被静默覆盖。
def test_merge_refuses_overwrite(tmp_path):
    m1 = tmp_path / "m1.json"
    make_manifest(m1, ["1ABC_H_L_P"])
    out = tmp_path / "out.json"
    out.write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.merge([m1], out)
