from scripts.audit_statecontrast_v2_fold_loaders import audit


def test_current_fold_loaders_are_component_disjoint(tmp_path):
    result = audit(
        __import__("pathlib").Path(
            "reviewer_outputs/statecontrast_v2_cross_validation_v2/contract.json"),
        tmp_path / "audit.json",
    )
    assert all(result["checks"].values())
    assert result["heldout_component_count"] == 23
