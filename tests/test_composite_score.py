"""Tests for composite scoring in modules/idp_antibody_design.py.

Covers: disorder_to_af2_reliability (calibration sigmoid) and composite_score /
rank_designs with mocked design dicts. No model/AF2/GPU required.
"""
import math

from idp_antibody_design import (
    composite_score,
    disorder_to_af2_reliability,
    graft_cdrs,
    rank_designs,
)


# ── disorder_to_af2_reliability ──

def test_reliability_monotone_decreasing():
    """Higher disorder → lower AF2 reliability (AF2 less trustworthy)."""
    r_low = disorder_to_af2_reliability(0.1)
    r_mid = disorder_to_af2_reliability(0.4)
    r_high = disorder_to_af2_reliability(0.8)
    assert r_low > r_mid > r_high


def test_reliability_bounds():
    """Reliability is always in [0, 1] for disorder in [0, 1]."""
    for d in [0.0, 0.2, 0.35, 0.5, 0.7, 1.0]:
        r = disorder_to_af2_reliability(d)
        assert 0.0 < r <= 1.0
    # At the midpoint disorder=0.35, reliability ≈ 0.5 (sigmoid midpoint).
    assert abs(disorder_to_af2_reliability(0.35) - 0.5) < 0.02


# ── composite_score (no AF2, FixBB fallback path) ──

def _mk(seq="ACDEFGHIK", ppl=50.0, entropy=1.5, plddt=0.9, iptm=0.8, pae=5.0):
    return {"sequence": seq, "ppl": ppl, "entropy": entropy,
            "plddt": plddt, "iptm": iptm, "pae": pae}


def test_composite_score_no_af2_orders_by_quality():
    """Without AF2, lower PPL/entropy and higher pLDDT/ipTM → higher score."""
    good = _mk(ppl=20.0, entropy=0.5, plddt=0.95, iptm=0.85)
    bad = _mk(ppl=150.0, entropy=2.5, plddt=0.5, iptm=0.3)
    assert composite_score(good, use_af2=False) > composite_score(bad, use_af2=False)


def test_composite_score_bounded():
    s = composite_score(_mk(), use_af2=False)
    assert 0.0 <= s <= 1.0


# ── rank_designs ──

def test_rank_designs_sorts_descending():
    seg_results = [{
        "designs": [
            _mk(ppl=150.0, entropy=2.5, plddt=0.5, iptm=0.3),
            _mk(ppl=20.0, entropy=0.5, plddt=0.95, iptm=0.85),
            _mk(ppl=80.0, entropy=1.5, plddt=0.8, iptm=0.6),
        ],
        "segment_start": 1, "segment_end": 10,
        "segment_mean_disorder": 0.2, "epitope_mode": "extracted",
    }]
    ranked = rank_designs(seg_results, use_af2=False)
    scores = [r["composite_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["rank"] == 1
    # Best design is the good one (low ppl).
    assert ranked[0]["ppl"] == 20.0


def test_rank_designs_quality_labels():
    seg_results = [{"designs": [_mk(ppl=20.0, entropy=0.5, plddt=0.95, iptm=0.9)],
                    "segment_start": 1, "segment_end": 10,
                    "segment_mean_disorder": 0.2, "epitope_mode": "extracted"}]
    ranked = rank_designs(seg_results, use_af2=False)
    assert ranked[0]["quality_label"] in ("HIGH", "MEDIUM", "LOW")


# ── graft_cdrs ──

def test_graft_cdrs_replaces_regions():
    """Grafted CDR sequences replace the corresponding scaffold positions."""
    scaffold = "AAAAAAAAAAAAAAAAAAA"  # 19 A's
    # CDR ranges (1-indexed): 2-4 (len 3), 8-9 (len 2)
    cdr_ranges = [(2, 4, 3), (8, 9, 2)]
    designed = "DEFG"  # 3 + 2... but designed must match total len 5
    designed = "DEFGH"  # len 5
    out, muts = graft_cdrs(scaffold, designed, cdr_ranges)
    assert out[1:4] == "DEF"          # positions 2-4 (0-idx 1-3)
    assert out[7:9] == "GH"           # positions 8-9 (0-idx 7-8)
    assert len(muts) == 5             # all 5 positions changed from A
