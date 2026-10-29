"""StateContrast v2 记录契约测试：验证目标/负状态必备与教师分覆盖率门槛。

守护 disorderflow.statecontrast_v2_contract.validate_statecontrast_v2_records：
每个组必须同时含 target 与负状态记录（缺一即无效）、
独立教师分覆盖率低于门槛时契约必须失败——
防止无对照或无监督信号的组进入状态对比训练。
"""
from disorderflow.statecontrast_v2_contract import validate_statecontrast_v2_records


# 辅助函数：构造一条最小 v2 记录（状态来源、权重与教师分可调）。
def _record(group, state, source="experimental", teacher=0.5):
    return {
        "group_id": group,
        "state": {"type": state, "source": source, "prior_weight": 1.0},
        "targets": {
            "state_training_weight": 1.0,
            "independent_teacher_score": teacher,
        },
    }


# 对照完备性：只有 target 没有负状态的组必须判无效——防止无对比的状态监督。
def test_contract_requires_target_and_negative_state():
    invalid = validate_statecontrast_v2_records([_record("g", "target")])
    assert invalid["valid"] is False
    valid = validate_statecontrast_v2_records([
        _record("g", "target"), _record("g", "apo")])
    assert valid["valid"] is True


# 教师分覆盖：缺教师分的记录拉低覆盖率，低于门槛必须判无效——
# 防止大部分记录无监督信号仍通过校验。
def test_contract_checks_teacher_coverage():
    rows = [_record("g", "target", teacher=1.0), _record("g", "off_target", teacher=None)]
    result = validate_statecontrast_v2_records(rows, minimum_teacher_coverage=0.8)
    assert result["teacher_coverage"] == 0.5
    assert result["valid"] is False
