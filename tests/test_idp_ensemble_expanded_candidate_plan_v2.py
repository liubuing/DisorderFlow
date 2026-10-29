"""IDP 集成扩展候选计划 v2 测试：验证面板与 pose 就绪组件的一致性约束。

守护 scripts.prepare_idp_ensemble_expanded_candidate_plan_v2：
计划组件必须与 pose 面板就绪组件逐一对应，
future_confirmation_eligible_count 按实际可确认数量统计——
防止计划引用不存在 pose 来源的组件（幽灵组件）。
"""
import json

from scripts.prepare_idp_ensemble_expanded_candidate_plan_v2 import prepare


# 组件对应约束：计划组件须在 pose 面板中就绪；此处单组件计划合法但
# 未来确认资格计数必须如实为 0——防止扩大化的资格声明。
def test_expanded_candidate_plan_requires_matching_pose_components(tmp_path):
    panel = tmp_path / "panel.json"
    poses = tmp_path / "poses.json"
    output = tmp_path / "plan.json"
    panel.write_text(json.dumps({
        "components": [{
            "component_id": "C1", "target": "tau", "lineage_proxy": "L1"
        }]
    }), encoding="ascii")
    poses.write_text(json.dumps({
        "status": "pose_panel_ready", "components": {"C1": {"ready": True}}
    }), encoding="ascii")
    result = prepare(panel, poses, output)
    assert len(result["components"]) == 1
    assert result["future_confirmation_eligible_count"] == 0
