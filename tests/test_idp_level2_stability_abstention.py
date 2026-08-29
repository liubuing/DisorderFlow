from pathlib import Path

from scripts.analyze_idp_level2_stability_abstention import analyze

ROOT = Path(__file__).resolve().parents[1]


def test_two_axis_abstention_requires_both_stability_checks(tmp_path):
    result = analyze(
        ROOT / "configs/benchmarks/idp_level2_stability_abstention_diagnostic_v1.yml",
        tmp_path / "analysis.json",
    )
    assert result["attempted_pairs"] == 48
    assert all(
        row["eligible"] == (
            row["source_panel_stable"] and row["leave_one_pose_out_stable"]
        )
        for row in result["records"]
    )
    assert result["decision"] == "diagnostic_only_do_not_change_level2_status"
