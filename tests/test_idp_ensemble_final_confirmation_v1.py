from pathlib import Path

from scripts.check_idp_ensemble_final_confirmation_v1 import check

ROOT = Path(__file__).resolve().parents[1]


def test_final_confirmation_is_fail_closed_without_new_idp_cohort(tmp_path):
    result = check(
        ROOT / "configs/benchmarks/idp_ensemble_final_confirmation_v1.yml",
        tmp_path / "readiness.json",
    )
    assert result["status"] == "final_confirmation_blocked"
    assert result["checks"]["pipeline_contract_frozen"] is True
    assert result["checks"]["checkpoint_frozen"] is True
    assert result["checks"]["untouched_idp_cohort_ready"] is False
