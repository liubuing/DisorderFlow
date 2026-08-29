import json

from scripts.prepare_idp_ensemble_expanded_resampling_v2 import prepare


def test_resampling_requests_experimental_ready_components(tmp_path):
    readiness = tmp_path / "readiness.json"
    donor = tmp_path / "donor.json"
    readiness.write_text(json.dumps({
        "components": [
            {"component_id": "C1", "multi_pose_ready": True},
            {"component_id": "C2", "multi_pose_ready": False},
        ]
    }), encoding="ascii")
    donor.write_text(json.dumps({
        "components": [
            {"component_id": "C1", "experimental_pose_ready": True},
            {"component_id": "C2", "experimental_pose_ready": False},
        ]
    }), encoding="ascii")
    result = prepare(readiness, donor, tmp_path / "output.json")
    assert result["sampling_requested_count"] == 1
    assert result["components"][0]["multi_pose_ready"] is False
    assert result["components"][1]["multi_pose_ready"] is True
