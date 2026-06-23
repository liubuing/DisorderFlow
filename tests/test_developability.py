"""Tests for negative-design + MD + immunogenicity modules + scorer integration.

All proxy backends (no GPU / no GROMACS / no NetMHCIIpan). Validates that:
  - off_target_docking degrades gracefully when PDB is missing.
  - immunogenicity proxy is in [0,1] and ranks hydrophobic CDRs higher.
  - md_stability 'skip' backend returns None.
  - closed_loop_scorer applies the new off_target/md_rmsd/immunogenicity thresholds.
"""
import os

import pytest

# ── immunogenicity proxy ──
from immunogenicity import predict_immunogenicity


def test_immunogenicity_proxy_bounded():
    for seq in ["", "A", "ACDEFGHIKLMNPQRSTVWY", "W" * 50, "D" * 50]:
        s = predict_immunogenicity(seq, backend="proxy")
        assert 0.0 <= s <= 1.0


def test_immunogenicity_hydrophobic_higher_than_soluble():
    """A hydrophobic/aromatic stretch scores higher (worse) than a charged one."""
    hydrophobic = predict_immunogenicity("WFWFWFWFWFWFWFWFWFWF", backend="proxy")
    charged = predict_immunogenicity("DEDEDEDEDEDEDEDEDEDE", backend="proxy")
    assert hydrophobic > charged


def test_immunogenicity_netmhciipan_falls_back_to_proxy():
    """When netMHCIIpan binary is absent, falls back to proxy (never None)."""
    s = predict_immunogenicity("ACDEFGHIKL", backend="netmhciipan",
                               netmhciipan_exe="/nonexistent/netMHCIIpan")
    assert 0.0 <= s <= 1.0  # proxy fallback, not None


# ── off_target_docking graceful degradation ──
def test_off_target_scores_missing_pdb_returns_empty():
    from off_target_docking import compute_off_target_scores
    out = compute_off_target_scores(
        [{"full_ab_seq": "ACDE"}], "/nonexistent/off.pdb", "A", verbose=False)
    assert out == {}


def test_off_target_batch_filter_keeps_passing():
    from off_target_docking import batch_off_target_filter
    # With a missing PDB, scores are empty → all designs pass (iptm defaults 0).
    designs = [{"full_ab_seq": "ACDE", "sequence": "ACDE"}]
    kept = batch_off_target_filter(designs, off_target_iptm_max=0.3,
                                   off_target_pdb="/nonexistent/off.pdb",
                                   off_target_chain="A")
    assert len(kept) == 1
    assert kept[0]["off_target_iptm"] == 0.0


# ── md_stability ──
def test_md_skip_returns_none():
    from md_stability import run_md_stability
    assert run_md_stability("any.pdb", backend="skip") is None


def test_md_gromacs_no_binary_returns_none_rmsd():
    """gromacs backend with no gmx installed → rmsd None (graceful)."""
    from md_stability import run_md_stability
    res = run_md_stability("any.pdb", backend="gromacs",
                           gromacs_exe="/nonexistent/gmx", verbose=False)
    assert res["rmsd"] is None


# ── closed_loop_scorer integration (new thresholds) ──
def test_scorer_rejects_off_target_binder():
    from closed_loop_scorer import MultiObjectiveRanker, MULTI_OBJECTIVE_THRESHOLDS
    th = dict(MULTI_OBJECTIVE_THRESHOLDS)
    th["off_target_iptm_max"] = 0.3
    ranker = MultiObjectiveRanker(thresholds=th)
    designs = [
        {"plddt": 80, "iptm": 0.7, "off_target_iptm": 0.1},   # passes
        {"plldt": 80, "iptm": 0.7, "off_target_iptm": 0.6},   # rejected (off-target)
    ]
    # Fix typo'd key on second design:
    designs[1]["plddt"] = 80
    passed, counts = ranker.apply_hard_thresholds(designs)
    assert len(passed) == 1
    assert counts["off_target"] == 1


def test_scorer_rejects_immunogenic_design():
    from closed_loop_scorer import MultiObjectiveRanker, MULTI_OBJECTIVE_THRESHOLDS
    th = dict(MULTI_OBJECTIVE_THRESHOLDS)
    th["immunogenicity_max"] = 0.6
    ranker = MultiObjectiveRanker(thresholds=th)
    designs = [
        {"plddt": 80, "iptm": 0.7, "immunogenicity": 0.2},    # passes
        {"plddt": 80, "iptm": 0.7, "immunogenicity": 0.9},    # rejected
    ]
    passed, counts = ranker.apply_hard_thresholds(designs)
    assert len(passed) == 1
    assert counts["immuno"] == 1


# ── cascade_filter integration ──
def test_cascade_rejects_off_target_with_enabled_threshold():
    from cascade_filter import apply_cascade
    designs = [
        {"sequence": "GOOD", "plddt": 0.9, "iptm": 0.8, "ppl": 20,
         "off_target_iptm": 0.1, "immunogenicity": 0.2},
        {"sequence": "BBBBOFFARGET", "plddt": 0.9, "iptm": 0.8, "ppl": 20,
         "off_target_iptm": 0.6, "immunogenicity": 0.2},
    ]
    filtered, _ = apply_cascade(designs, thresholds={
        "off_target_iptm_max": 0.3, "immunogenicity_max": 0.9,
        "ppl_max": 100, "entropy_max": 3.0,
    })
    seqs = [r["sequence"] for r in filtered]
    assert "GOOD" in seqs and "BBBBOFFARGET" not in seqs
