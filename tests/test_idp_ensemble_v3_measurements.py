from scripts.check_idp_ensemble_v3_measurements import check


def test_missing_measurements_are_blocked(tmp_path):
    panel = tmp_path / "panel.json"
    panel.write_text(
        '{"components": [{"component_id": "C1"}]}', encoding="ascii"
    )
    result = check(tmp_path / "missing.csv", panel, tmp_path / "status.json")
    assert result["status"] == "measurements_blocked"
    assert result["measured_components"] == 0
