import json

from scripts.audit_statecontrast_v2_cv_isolation import audit


def test_cv_audit_detects_candidate_group_leak(tmp_path):
    manifest = tmp_path / "manifest.json"
    base = {
        "antibody_lineage_cluster": "l", "global_sequence_cluster": "s",
        "teacher_group_id": "t", "group_id": "g", "component_id": "c",
    }
    manifest.write_text(json.dumps({
        "cross_validation_folds": 2,
        "records": [{**base, "fold_id": 0}, {**base, "fold_id": 1}],
    }))
    result = audit(manifest, tmp_path / "out.json")
    assert result["status"] == "statecontrast_v2_cv_isolation_failed"
    assert result["leaks"]["candidate_groups"] == ["g"]
