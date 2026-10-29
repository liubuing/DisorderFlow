"""StateContrast v2 折外（OOF）聚合测试：验证冻结折完整性与组件唯一归属。

守护 scripts.aggregate_statecontrast_v2_oof.aggregate：
聚合必须覆盖契约中的每一个冻结折（缺折即报错）、
同一组件的效果出现在多个 held-out 折时必须拒绝——
防止折外统计缺折抽样或组件跨折重复计入（伪独立）。
"""
import json

import pytest

from scripts.aggregate_statecontrast_v2_oof import aggregate


# 冻结折完整性：契约有 2 折但只提供 0 个评估文件必须报错——防止缺折聚合掩盖部分失败。
def test_oof_aggregation_requires_every_frozen_fold(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"folds": [{"fold": 0}, {"fold": 1}]}))
    gate = tmp_path / "gate.yml"
    gate.write_text("classification: test\n")
    with pytest.raises(ValueError, match="Every frozen fold"):
        aggregate(contract, gate, {}, tmp_path / "out.json")


# 组件唯一归属：同一组件效果不得同时出现在折 0 与折 1 的 held-out 结果中——
# 防止组件跨折重复而夸大折外样本量。
def test_oof_aggregation_rejects_component_overlap(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"folds": [{"fold": 0}, {"fold": 1}]}))
    gate = tmp_path / "gate.yml"
    gate.write_text("classification: test\n")
    evaluations = {}
    for fold in (0, 1):
        path = tmp_path / f"fold{fold}.json"
        path.write_text(json.dumps({
            "split": "val", "component_effects": {"same": 1.0},
            "matched_pairs": [], "checkpoint_sha256": str(fold),
        }))
        evaluations[fold] = path
    with pytest.raises(ValueError, match="multiple held-out folds"):
        aggregate(contract, gate, evaluations, tmp_path / "out.json")
