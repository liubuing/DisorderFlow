"""IdP 系综 v3 姿态协议检查的契约测试：守护姿态清单缺失时整体必须 blocked 且无就绪组件，
防止协议要求（每组件最少姿态数）在证据缺失时被静默跳过。"""
import json

from scripts.check_idp_ensemble_v3_pose_protocol import check


# fail-closed：协议存在、面板存在但姿态清单缺失时，状态必须为 pose_panel_blocked 且就绪组件为 0——
# 防止无姿态证据的组件冒充满足协议。
def test_missing_pose_manifest_is_blocked(tmp_path):
    protocol = tmp_path / "protocol.yml"
    protocol.write_text(
        "requirements:\n  minimum_poses_per_component: 2\nclaim_boundary: test\n",
        encoding="ascii",
    )
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps({"components": [{"component_id": "C1"}]}), encoding="ascii"
    )
    result = check(protocol, panel, tmp_path / "missing.json", tmp_path / "status.json")
    assert result["status"] == "pose_panel_blocked"
    assert result["ready_components"] == 0
