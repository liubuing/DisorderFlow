"""Deployment-grade, honest confidence policy for candidate ranking.

Only interface PAE is deployment-grade (see
``publication/candidate_interface_pae_deployment_v1.json``). pLDDT and ipTM are
reported but explicitly abstained and never used for ranking. PPL and entropy
are BFN-native sequence-quality signals, not confidence signals, and serve only
as tie-breakers after the confidence signal.
"""

from __future__ import annotations

from typing import Iterable

_PAE_DEFAULT = 31.0
_CONFIDENCE_ABSTENTIONS = ("plddt", "iptm")
_CONFIDENCE_POLICY = "pae_only"


def deployment_pae(design: dict) -> float | None:
    """Return the deployment-grade interface PAE for a design (lower is better).

    Prefers the AF2-derived ``af2_interface_pae`` (design-specific); falls back
    to the BFN-native ``pae`` (constant for fixed-backbone designs). Returns
    ``None`` when neither is available.
    """
    value = design.get("af2_interface_pae")
    if value is None:
        value = design.get("pae")
    if value is None:
        return None
    return float(value)


def honest_rank_key(entry: dict):
    """Sort key over ranked entries: deployment PAE first, then PPL/entropy."""
    pae = entry.get("interface_pae")
    ppl = entry.get("ppl")
    entropy = entry.get("entropy")
    return (
        pae is None,                       # missing PAE sorts last
        _PAE_DEFAULT if pae is None else pae,
        ppl is None, (ppl if ppl is not None else 200.0),
        entropy is None, (entropy if entropy is not None else 3.0),
    )


def _reported_not_deployed(design: dict, *keys: str) -> dict:
    return {key: design.get(key) for key in keys}


def rank_designs_honest(all_designs: Iterable[dict]) -> list[dict]:
    """Rank designs by deployment-grade interface PAE with explicit abstentions.

    Args:
        all_designs: iterable of segment results, each a dict with a ``designs``
            list of per-sample dicts.

    Returns:
        A single sorted list of design entries (ascending deployment PAE), each
        carrying ``confidence_policy`` and ``confidence_abstentions`` so the
        abstained signals are surfaced rather than silently used.
    """
    ranked = []
    for seg_result in all_designs:
        for index, design in enumerate(seg_result.get("designs", [])):
            ranked.append({
                "sequence": design.get("sequence", ""),
                "interface_pae": deployment_pae(design),
                "plddt_reported_not_deployed": design.get("plddt"),
                "iptm_reported_not_deployed": design.get("iptm"),
                "ppl": design.get("ppl"),
                "entropy": design.get("entropy"),
                "confidence_policy": _CONFIDENCE_POLICY,
                "confidence_abstentions": list(_CONFIDENCE_ABSTENTIONS),
                "sample_index": index,
                "segment_start": seg_result.get("segment_start"),
                "segment_end": seg_result.get("segment_end"),
                "segment_mean_disorder": seg_result.get("segment_mean_disorder"),
            })
    ranked.sort(key=honest_rank_key)
    for position, entry in enumerate(ranked, 1):
        entry["rank"] = position
    return ranked
