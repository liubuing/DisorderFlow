"""RCSB 候选界面扩展获取脚本测试：验证结果 ID 解析与冻结查询哈希校验。

守护 RCSB 快照获取管线的可复现性不变量：
result_ids 兼容紧凑/详细两种响应且统一大写、截断响应必须拒绝、
acquire 严格校验查询哈希与协议章节——防止静默使用被篡改或缺失契约的查询。
"""
import importlib.util
import json
from pathlib import Path

import pytest


# 模块级加载被测脚本（scripts/build/acquire_rcsb_candidate_interface_extension.py）。
SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "acquire_rcsb_candidate_interface_extension.py"
SPEC = importlib.util.spec_from_file_location("acquire_rcsb_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 紧凑格式（字符串列表）的 RCSB 响应应能直接提取结果 ID。
def test_result_ids_accepts_compact_response():
    response = {"total_count": 2, "result_set": ["1ABC", "2DEF"]}
    assert MODULE.result_ids(response) == ["1ABC", "2DEF"]


# 详细格式（对象列表）同样支持，且 ID 统一归一化为大写——
# 防止大小写不一致导致下游文件名/注册表对不上。
def test_result_ids_accepts_verbose_response():
    response = {
        "total_count": 2,
        "result_set": [{"identifier": "1abc"}, {"identifier": "2def"}],
    }
    assert MODULE.result_ids(response) == ["1ABC", "2DEF"]


# total_count 与实际返回条数不一致（响应被截断）必须显式报错，
# 防止不完整快照被当作完整结果进入候选池。
def test_result_ids_rejects_truncated_response():
    with pytest.raises(ValueError, match="truncated"):
        MODULE.result_ids({"total_count": 3, "result_set": ["1ABC"]})


# 冻结校验：协议中记录的 query sha256 与实际查询文件不符时 acquire 必须拒绝——
# 防止"冻结查询不可重写"的溯源承诺被破坏。
def test_acquire_rejects_changed_frozen_query(tmp_path):
    query = tmp_path / "query.json"
    protocol = tmp_path / "protocol.json"
    query.write_text("{}\n", encoding="ascii")
    protocol.write_text(json.dumps({
        "discovery_contract": {"query_config_sha256": "not-the-query-hash"},
    }), encoding="ascii")

    with pytest.raises(ValueError, match="Frozen RCSB query hash"):
        MODULE.acquire(query, protocol, tmp_path / "snapshot")


# 协议缺少必需章节（discovery_round_v2）时必须显式报错，
# 防止契约不完整的协议被静默接受。
def test_acquire_rejects_missing_protocol_section(tmp_path):
    import hashlib
    query = tmp_path / "query.json"
    protocol = tmp_path / "protocol.json"
    query.write_text("{}\n", encoding="ascii")
    protocol.write_text(json.dumps({
        "discovery_round_v2": {
            "query_config_sha256": hashlib.sha256(b"{}\n").hexdigest(),
        },
    }), encoding="ascii")

    with pytest.raises(ValueError, match="Protocol lacks section"):
        MODULE.acquire(query, protocol, tmp_path / "snapshot")
