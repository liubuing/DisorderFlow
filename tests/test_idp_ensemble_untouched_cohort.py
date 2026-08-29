import json

from scripts.admit_idp_ensemble_untouched_cohort import admit


def test_historical_registry_cannot_admit_confirmation_cohort(tmp_path):
    discovery = tmp_path / "discovery.json"
    registry = tmp_path / "components.csv"
    output = tmp_path / "admission.json"
    discovery.write_text(json.dumps({"candidate_entry_ids": []}), encoding="ascii")
    registry.write_text("component_id,target\nOLD,tau\n", encoding="ascii")
    result = admit(discovery, registry, output)
    assert result["status"] == "untouched_cohort_blocked"
    assert result["observed"]["exact_new_structures"] == 0
    assert result["decision"].startswith("do_not")
