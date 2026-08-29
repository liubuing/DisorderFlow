from pathlib import Path

from scripts.analyze_idp_ensemble_bfn_state_compatibility import analyze

ROOT = Path(__file__).resolve().parents[1]


def test_state_compatibility_diagnostic_uses_candidate_sensitive_scores(tmp_path):
    result = analyze(
        ROOT / "configs/benchmarks/idp_ensemble_bfn_state_compatibility_dev_v1.yml",
        tmp_path / "analysis.json",
    )
    assert result["endpoint"]["name"] == "bfn_fixed_candidate_state_compatibility"
    assert result["median_candidate_score_range"] > 0.01
    assert result["decision"] in {
        "eligible_to_freeze_for_new_untouched_panel",
        "do_not_promote_state_compatibility_endpoint",
    }
