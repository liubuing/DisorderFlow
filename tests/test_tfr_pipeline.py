"""Tests for the TfR design pipeline negative-design ranking logic.

Covers apply_negative_design_ranking and format_tfr_report with mocked design
dicts + off-target scores. No BFN/AF2/GPU required — the ranking module is
tested in isolation by passing pre-computed off_target_scores.
"""
from tfr_design_pipeline import (
    apply_negative_design_ranking,
    format_tfr_report,
)


def _mk(seq, composite, full_ab=None):
    return {
        "sequence": seq,
        "composite_score": composite,
        "full_ab_seq": full_ab or seq,
    }


def test_negative_design_rejects_off_target_binders():
    """Designs above off_target_iptm_max are flagged and demoted to the bottom."""
    designs = [
        _mk("AAAA", 0.80, full_ab="AB1"),
        _mk("BBBB", 0.70, full_ab="AB2"),  # high off-target
        _mk("CCCC", 0.60, full_ab="AB3"),
    ]
    scores = {"AB1": 0.05, "AB2": 0.60, "AB3": 0.10}  # AB2 exceeds 0.3 threshold
    out = apply_negative_design_ranking(designs, scores, off_target_iptm_max=0.3)

    # AB2 should be last and flagged as rejected.
    rejected = [d for d in out if d["off_target_rejected"]]
    assert len(rejected) == 1 and rejected[0]["full_ab_seq"] == "AB2"
    assert out[-1]["full_ab_seq"] == "AB2"

    # Non-rejected designs retain their relative (specificity) order.
    non_rej = [d for d in out if not d["off_target_rejected"]]
    assert non_rej[0]["full_ab_seq"] == "AB1"  # highest specificity_score


def test_specificity_score_penalises_off_target():
    """A design with higher off-target ipTM gets a lower specificity_score,
    even when on-target composite is equal."""
    designs = [_mk("AAAA", 0.80, full_ab="AB1"),
               _mk("BBBB", 0.80, full_ab="AB2")]
    scores = {"AB1": 0.05, "AB2": 0.25}
    out = apply_negative_design_ranking(designs, scores, off_target_iptm_max=0.3)
    by_key = {d["full_ab_seq"]: d for d in out}
    assert by_key["AB1"]["specificity_score"] > by_key["AB2"]["specificity_score"]
    # AB1 ranks first.
    assert out[0]["full_ab_seq"] == "AB1"


def test_missing_off_target_score_defaults_zero():
    """Designs without an off-target score get 0.0 (treated as non-binding)."""
    designs = [_mk("AAAA", 0.7, full_ab="AB1")]
    out = apply_negative_design_ranking(designs, {}, off_target_iptm_max=0.3)
    assert out[0]["off_target_iptm"] == 0.0
    assert out[0]["off_target_rejected"] is False


def test_re_ranked_list_gets_consecutive_positions():
    """After re-ranking, callers re-assign ranks 1..N (pipeline does this)."""
    designs = [_mk("A", 0.8, "AB1"), _mk("B", 0.6, "AB2"), _mk("C", 0.7, "AB3")]
    scores = {"AB1": 0.0, "AB2": 0.8, "AB3": 0.1}
    out = apply_negative_design_ranking(designs, scores, off_target_iptm_max=0.3)
    # caller assigns ranks:
    for i, d in enumerate(out):
        d["rank"] = i + 1
    assert out[0]["rank"] == 1
    assert out[-1]["full_ab_seq"] == "AB2"  # rejected → last


def test_format_tfr_report_handles_empty():
    assert "No TfR designs" in format_tfr_report([])


def test_format_tfr_report_includes_rejected_flag():
    designs = [
        {"rank": 1, "sequence": "ACDEFGHIKABCDEFGHIJKLMNOPQRSTUVWX",
         "composite_score": 0.8, "specificity_score": 0.75,
         "off_target_iptm": 0.05, "off_target_rejected": False},
        {"rank": 2, "sequence": "KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK",
         "composite_score": 0.6, "specificity_score": 0.30,
         "off_target_iptm": 0.50, "off_target_rejected": True},
    ]
    report = format_tfr_report(designs)
    assert "REJECT" in report
    assert "TfR Nanobody" in report
