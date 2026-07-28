#!/usr/bin/env python
"""Positive/negative state-contrast scorer for IDP antibody paratopes."""
from __future__ import annotations

from typing import Dict, Iterable, List

from state_contact_scorer import score_sequence_on_contact_map


def score_state_contrast(sequence: str, positive_states: Iterable, negative_states: Iterable) -> Dict:
    """Score a paratope by positive retention minus negative-state binding."""
    pos_rows = [_score_state(sequence, s) for s in positive_states]
    neg_rows = [_score_state(sequence, s) for s in negative_states]
    pos_score = sum(r["state_contact_score"] for r in pos_rows) / max(1, len(pos_rows))
    neg_max = max((r["state_contact_score"] for r in neg_rows), default=0.0)
    neg_mean = sum(r["state_contact_score"] for r in neg_rows) / max(1, len(neg_rows))
    gap = pos_score - neg_max
    return {
        "sequence": sequence,
        "positive_score": round(pos_score, 4),
        "negative_max_score": round(neg_max, 4),
        "negative_mean_score": round(neg_mean, 4),
        "specificity_gap": round(gap, 4),
        "positive_states": pos_rows,
        "negative_states": neg_rows,
    }


def rank_state_contrast_candidates(variants: Iterable[Dict], positive_states: List, negative_states: List) -> List[Dict]:
    rows = []
    native_seq = positive_states[0].contact_map["paratope_sequence"] if positive_states else ""
    native = score_state_contrast(native_seq, positive_states, negative_states) if native_seq else {"specificity_gap": 0.0}
    native_gap = native["specificity_gap"]
    for v in variants:
        s = score_state_contrast(v["sequence"], positive_states, negative_states)
        gap_delta = s["specificity_gap"] - native_gap
        rows.append({
            **v,
            "positive_score": s["positive_score"],
            "negative_max_score": s["negative_max_score"],
            "negative_mean_score": s["negative_mean_score"],
            "specificity_gap": s["specificity_gap"],
            "native_specificity_gap": native_gap,
            "specificity_gap_delta": round(gap_delta, 4),
            "statecontrast_score": round(0.70 * gap_delta + 0.30 * s["positive_score"], 4),
        })
    rows.sort(key=lambda r: (r["statecontrast_score"], r["specificity_gap_delta"], r["positive_score"]), reverse=True)
    for i, r in enumerate(rows, 1):
        r["statecontrast_rank"] = i
    return rows


def _score_state(sequence: str, state) -> Dict:
    score = score_sequence_on_contact_map(sequence, state.contact_map)
    return {
        "state": state.name,
        "state_type": state.state_type,
        "state_contact_score": score["state_contact_score"],
        "hotspot_score": score["hotspot_score"],
    }
