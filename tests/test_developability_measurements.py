import json

from scripts.audit_developability_measurements import audit


def test_missing_measurements_fail_closed(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        "classification: test\n"
        "required_measurements:\n"
        "  minimum_replicates: 2\n"
        "  minimum_expression_mg_per_l: 10\n"
        "  minimum_sec_monomer_fraction: 0.9\n"
        "  maximum_aggregate_fraction: 0.1\n"
        "  maximum_polyspecificity_normalized: 0.35\n"
        "claim_boundary: test only\n")
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"records": [{
        "candidate_id": "x", "replicates": None,
        "expression_mg_per_l": None, "sec_monomer_fraction": None,
        "aggregate_fraction": None, "polyspecificity_normalized": None,
    }]}))
    result = audit(config, measurements, tmp_path / "out.json")
    assert result["status"] == "developability_measurements_blocked"
    assert result["pass_count"] == 0


def test_complete_measurement_record_can_pass(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        "classification: test\n"
        "required_measurements:\n"
        "  minimum_replicates: 2\n"
        "  minimum_expression_mg_per_l: 10\n"
        "  minimum_sec_monomer_fraction: 0.9\n"
        "  maximum_aggregate_fraction: 0.1\n"
        "  maximum_polyspecificity_normalized: 0.35\n"
        "claim_boundary: test only\n")
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"records": [{
        "candidate_id": "x", "replicates": 2,
        "expression_mg_per_l": 20, "sec_monomer_fraction": 0.95,
        "aggregate_fraction": 0.05, "polyspecificity_normalized": 0.2,
    }]}))
    result = audit(config, measurements, tmp_path / "out.json")
    assert result["status"] == "developability_measurements_complete"
    assert result["pass_count"] == 1
