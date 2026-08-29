from pathlib import Path

from scripts.check_idp_level2_computational_validation import check, evaluate_gate

ROOT = Path(__file__).resolve().parents[1]


def test_fraction_gate_is_fail_closed_at_threshold():
    evidence = {"result": {"good": 3, "total": 4}}
    rule = {
        "source": "result", "numerator": "good", "denominator": "total",
        "minimum_fraction": 0.8, "category": "existing_evidence",
    }
    result = evaluate_gate(rule, evidence)
    assert result["observed"] == 0.75
    assert result["passed"] is False


def test_current_level2_evidence_fails_without_claim_upgrade(tmp_path):
    result = check(
        ROOT / "configs/benchmarks/idp_level2_computational_validation_v1.yml",
        tmp_path / "status.json",
    )
    assert result["status"] == "level2_failed_existing_evidence"
    assert result["passed"] is False
    assert "idp_extension_replication" in result["failed_existing_evidence"]
    assert "independent_bfn_scoring" in result["failed_existing_evidence"]
    assert "expanded_idp_independent_support" in result["failed_existing_evidence"]
    assert "expanded_idp_source_stability" in result["failed_existing_evidence"]
    assert "untouched_idp_cohort" in result["missing_evidence"]
