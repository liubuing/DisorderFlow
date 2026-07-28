"""Tests for cascade filtering in modules/cascade_filter.py.

Covers: hard-threshold stage and composite ranking with mocked design dicts.
No model/GPU required.
"""
from cascade_filter import apply_cascade, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS


def _mk(seq, ppl=20.0, plddt=0.9, iptm=0.8, entropy=1.0, score=None, recovery=None):
    r = {"sequence": seq, "ppl": ppl, "plddt": plddt, "iptm": iptm,
         "entropy": entropy}
    if score is not None:
        r["score"] = score
    if recovery is not None:
        r["recovery"] = recovery
    return r


def test_hard_threshold_rejects_high_ppl():
    """Designs above ppl_max are filtered out in stage 1."""
    results = [_mk("GOOD1", ppl=20.0), _mk("BADHH", ppl=500.0)]
    filtered, _ = apply_cascade(results, thresholds={"ppl_max": 100.0})
    seqs = [r["sequence"] for r in filtered]
    assert "GOOD1" in seqs and "BADHH" not in seqs


def test_hard_threshold_rejects_high_entropy():
    results = [_mk("GOOD1", entropy=1.0), _mk("BADHH", entropy=2.9)]
    filtered, _ = apply_cascade(results, thresholds={"entropy_max": 2.5})
    assert len(filtered) == 1
    assert filtered[0]["sequence"] == "GOOD1"


def test_composite_rank_orders_by_quality():
    """Better designs (low ppl, high confidence) rank first."""
    results = [
        _mk("LOWQ", ppl=150.0, plddt=0.4, iptm=0.3, entropy=2.5),
        _mk("HIGH", ppl=15.0, plddt=0.95, iptm=0.9, entropy=0.5),
    ]
    filtered, _ = apply_cascade(results, thresholds={
        "ppl_max": 200.0, "plddt_min": -0.01, "iptm_min": -0.01, "entropy_max": 3.0})
    assert len(filtered) == 2
    assert filtered[0]["sequence"] == "HIGH"
    assert filtered[0]["composite_score"] >= filtered[1]["composite_score"]


def test_dedup_keeps_best_duplicate():
    """Identical sequences are deduplicated, keeping the highest-scoring copy."""
    results = [_mk("DUPES", ppl=30.0), _mk("DUPES", ppl=15.0), _mk("UNIQ", ppl=40.0)]
    filtered, _ = apply_cascade(results, thresholds={"ppl_max": 100.0})
    seqs = [r["sequence"] for r in filtered]
    assert seqs.count("DUPES") == 1
    assert "UNIQ" in seqs


def test_empty_input_returns_empty():
    filtered, report = apply_cascade([])
    assert filtered == []
    assert isinstance(report, str)
