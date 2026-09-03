"""Per-conformation confidence reducer (generic; see note below).

NOTE: In this project's AF2 setup (single-sequence, templates disabled), AF2
returns a single interface PAE per (design, antigen) that does NOT vary across
target conformations. Per-conformation PAE is therefore not a valid ensemble
signal; the conformational ensemble enters only through structure-based contact
robustness (see ``modules.ensemble_contact``). This module is retained as a
generic reducer for the case where per-conformation confidence is available
(e.g. template-enabled AF2), and its ranking uses PAE only, with pLDDT/ipTM
abstained.
"""

from __future__ import annotations

from typing import Iterable

_PAE_MISSING = 31.0


def robust_interface_pae(per_conf_pae: Iterable[float | None],
                         robust_stat: str = "p25") -> dict:
    """Reduce per-conformation interface PAE to a robust summary.

    Args:
        per_conf_pae: interface PAE per conformation (lower is better).
        robust_stat: "p25" (default), "min", or "median".

    Returns:
        dict with min/median/max/robust/n_missing/per_conf.
    """
    values = [float(v) for v in per_conf_pae if v is not None]
    missing = sum(1 for v in per_conf_pae if v is None)
    if not values:
        return {
            "min": None, "median": None, "max": None, "robust": None,
            "n_missing": missing, "n_conformations": len(list(per_conf_pae)),
            "per_conf": list(per_conf_pae),
        }
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if robust_stat == "min":
        robust = float(arr.min())
    elif robust_stat == "median":
        robust = float(np.median(arr))
    else:
        robust = float(np.percentile(arr, 25))
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "robust": robust,
        "n_missing": missing,
        "n_conformations": len(values) + missing,
        "per_conf": list(per_conf_pae),
    }


def ensemble_rank_key(design: dict):
    """Sort key: robust PAE first, then median PAE, then missing count."""
    robust = design.get("robust_pae")
    median = design.get("median_pae")
    return (
        robust is None,
        _PAE_MISSING if robust is None else robust,
        median is None,
        _PAE_MISSING if median is None else median,
        design.get("n_missing", 0),
    )


def rank_designs_ensemble(designs: Iterable[dict], robust_stat: str = "p25") -> list[dict]:
    """Rank designs by ensemble-robust interface PAE across conformations.

    Args:
        designs: iterable of design dicts, each with a ``per_conf_pae`` list
            (one interface PAE per conformation).
        robust_stat: percentile strategy ("p25" default).

    Returns:
        sorted list of enriched design entries (ascending robust PAE).
    """
    ranked = []
    for design in designs:
        summary = robust_interface_pae(design.get("per_conf_pae", []), robust_stat)
        ranked.append({
            "sequence": design.get("sequence", ""),
            "robust_pae": summary["robust"],
            "median_pae": summary["median"],
            "min_pae": summary["min"],
            "max_pae": summary["max"],
            "n_missing": summary["n_missing"],
            "n_conformations": summary["n_conformations"],
            "per_conf_pae": summary["per_conf"],
            "confidence_policy": "pae_only",
            "confidence_abstentions": ["plddt", "iptm"],
        })
    ranked.sort(key=ensemble_rank_key)
    for position, entry in enumerate(ranked, 1):
        entry["rank"] = position
    return ranked
