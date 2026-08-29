import hashlib
import json

from scripts.build_statecontrast_v2_manifest import assign_fold, assign_split, build


def test_split_assignment_is_source_independent_and_deterministic():
    assert assign_split("cluster", 5, 0.8, 0.1) == assign_split(
        "cluster", 5, 0.8, 0.1)
    assert assign_fold("cluster", 5, 5) == assign_fold("cluster", 5, 5)


def test_manifest_builds_complete_explicit_state_group(tmp_path, monkeypatch):
    coordinate = tmp_path / "pose.pdb"
    coordinate.write_text("MODEL\nEND\n", encoding="ascii")
    digest = hashlib.sha256(coordinate.read_bytes()).hexdigest()
    states = []
    for state_type in ("target", "apo", "off_target"):
        states.append({
            "state_id": state_type,
            "pose_id": f"{state_type}_0",
            "state_type": state_type,
            "source": "experimental" if state_type == "target" else "matched_control",
            "source_panel": "experimental",
            "prior_weight": 1.0,
            "coordinate_path": coordinate.name,
            "coordinate_sha256": digest,
            "independent_teacher_score": 1.0 if state_type == "target" else 0.0,
            "independent_teacher_required": True,
            "antibody_chains": ["H"],
            "antigen_chains": [] if state_type == "apo" else ["A"],
        })
    source = tmp_path / "states.json"
    source.write_text(json.dumps({
        "status": "explicit_states_frozen_before_training",
        "components": [{
            "component_id": "component",
            "target": "target",
            "antibody_lineage_cluster": "lineage",
            "global_sequence_cluster": "sequence_cluster",
            "candidate_sequence": "AAAA",
            "native_h3": "AAAA",
            "substitution_bucket": 2,
            "dataset_source": "development",
            "states": states,
        }],
    }), encoding="utf-8")
    monkeypatch.setattr("scripts.build_statecontrast_v2_manifest.ROOT", tmp_path)
    result = build(source, tmp_path / "manifest.json")
    assert result["validation"]["valid"] is True
    assert result["records"][0]["fold_id"] in range(5)
    assert {row["state"]["type"] for row in result["records"]} == {
        "target", "apo", "off_target"}
