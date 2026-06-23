"""Tests for pure-function disorder analysis logic in modules/idp_disorder_analysis.py.

Covers: find_ordered_segments — segment detection, gap tolerance, min-window,
sorting, and residue_id mapping. No model/checkpoint/GPU required.
"""
import numpy as np

from idp_disorder_analysis import find_ordered_segments


def test_segment_detection_simple():
    """A single contiguous ordered stretch yields one segment."""
    # 20 residues: first 12 ordered (<0.3), rest disordered.
    scores = np.array([0.1] * 12 + [0.8] * 8)
    segs = find_ordered_segments(scores, threshold=0.3, min_window=8)
    assert len(segs) == 1
    s = segs[0]
    assert s["start"] == 1 and s["end"] == 12
    assert s["length"] == 12
    assert s["mean_disorder"] < 0.3


def test_min_window_filters_short_segments():
    """Ordered stretches shorter than min_window are discarded."""
    scores = np.array([0.1] * 5 + [0.9] * 10 + [0.1] * 12)  # 5-ordered, 12-ordered
    segs = find_ordered_segments(scores, threshold=0.3, min_window=8)
    assert len(segs) == 1
    assert segs[0]["length"] == 12


def test_gap_tolerance_bridges_small_gaps():
    """Up to gap_tolerance disordered residues are bridged into one segment."""
    # 8 ordered, 1 gap (disordered), 8 ordered → one segment with gap bridged.
    scores = np.array([0.1] * 8 + [0.9] + [0.1] * 8)
    segs = find_ordered_segments(scores, threshold=0.3, min_window=8, gap_tolerance=2)
    assert len(segs) == 1
    assert segs[0]["start"] == 1 and segs[0]["end"] == 17
    assert segs[0]["total_span"] == 17  # includes the bridged gap
    assert segs[0]["length"] == 16      # ordered count excludes the gap


def test_gap_too_large_splits_segments():
    """A gap larger than gap_tolerance splits into two segments."""
    scores = np.array([0.1] * 8 + [0.9, 0.9, 0.9] + [0.1] * 8)  # 3-gap
    segs = find_ordered_segments(scores, threshold=0.3, min_window=8, gap_tolerance=2)
    assert len(segs) == 2


def test_residue_id_mapping():
    """Custom residue_ids are mapped through to segment boundaries."""
    scores = np.array([0.1] * 10 + [0.8] * 5)
    rids = list(range(100, 115))
    segs = find_ordered_segments(scores, residue_ids=rids, threshold=0.3, min_window=8)
    assert len(segs) == 1
    assert segs[0]["start"] == 100 and segs[0]["end"] == 109


def test_segments_sorted_by_mean_disorder():
    """Segments are sorted by ascending mean_disorder, then descending length."""
    # Two ordered stretches: second has lower mean disorder.
    scores = np.array([0.25] * 10 + [0.9] * 5 + [0.05] * 10)
    segs = find_ordered_segments(scores, threshold=0.3, min_window=8)
    assert len(segs) == 2
    assert segs[0]["mean_disorder"] <= segs[1]["mean_disorder"]
    assert segs[0]["start"] == 16  # lower-disorder segment (0.05) ranks first


def test_no_ordered_residues_returns_empty():
    scores = np.array([0.8] * 20)
    segs = find_ordered_segments(scores, threshold=0.3, min_window=8)
    assert segs == []
