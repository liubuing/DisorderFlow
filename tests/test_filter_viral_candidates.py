"""病毒候选过滤脚本测试：验证只保留病毒抗原候选与输出不可覆盖。

守护 scripts/build/filter_viral_candidates.py 的病毒隔离语义：
只有 status=candidate 且 viral_antigen=True 的条目进入输出（被排除/延后的不算）、
输出文件已存在必须拒绝覆盖——防止病毒候选混入通用池或结果被静默重写。
"""
import importlib.util
import json
from pathlib import Path

import pytest

# 模块级加载被测脚本（scripts/build/filter_viral_candidates.py）。
SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "filter_viral_candidates.py"
SPEC = importlib.util.spec_from_file_location("filter_viral", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 过滤语义：非病毒 candidate、已排除条目、deferred 条目都不得进入病毒候选输出——
# 防止病毒相关结构被错误归入或漏出隔离池。
def test_filter_viral_keeps_only_viral_candidates(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({
        "classification": "metadata_only",
        "schema_version": "v1",
        "snapshot_dir": "data/snap",
        "snapshot_manifest_sha256": "abc",
        "entries": [
            {"pdb_id": "1ABC", "status": "candidate", "viral_antigen": True},
            {"pdb_id": "2DEF", "status": "candidate", "viral_antigen": False},
            {"pdb_id": "3GHI", "status": "excluded", "viral_antigen": True},
            {"pdb_id": "4JKL", "status": "deferred_viral_antigen"},
        ],
    }), encoding="utf-8")
    artifact = MODULE.filter_viral(discovery, tmp_path / "out.json")
    assert artifact["counts"]["viral_antigen_candidates"] == 1
    assert artifact["entries"][0]["pdb_id"] == "1ABC"


# 冻结不可重写：输出已存在必须拒绝且原文件不动——防止过滤结果被静默覆盖。
def test_filter_viral_refuses_overwrite(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({"classification": "metadata_only", "entries": []}))
    out = tmp_path / "out.json"
    out.write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.filter_viral(discovery, out)
