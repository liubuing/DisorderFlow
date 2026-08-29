import json

from scripts.check_idp_ensemble_v3_pose_protocol import check


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
