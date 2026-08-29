import json

from scripts.prepare_idp_ensemble_expanded_candidate_plan_v2 import prepare


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
