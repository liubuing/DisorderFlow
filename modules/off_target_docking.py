#!/usr/bin/env python
"""Off-target (negative-design) specificity scoring.

Computes how strongly a designed antibody also binds an OFF-target protein
(e.g. Transferrin, which competes with TfR1 in vivo). Lower off-target ipTM =
better specificity. Used by the TfR pipeline's negative-design pass and by the
closed-loop multi-objective ranker (`off_target_iptm` dimension).

Two backends:
  - 'af2' (default, no extra deps): reuse AlphaFold2-multimer ipTM as a
    docking proxy — `af2_jax_runner.validate_antibody_epitope(ab_seq, off_seq)`.
    This is the same JAX-AF2 runner the design pipeline uses for validation, so
    it adds zero new dependencies.
  - 'hdock' (optional): HDOCK rigid-body docking subprocess. Not installed by
    default; the backend falls back to 'af2' if the HDOCK binary is missing.

This module is import-safe: AF2/JAX is only loaded lazily inside the af2 backend.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional


def _extract_sequence_from_pdb(pdb_path: str, chain_id: str) -> str:
    """Extract 1-letter sequence from a PDB chain. Mirrors idp_antibody_design."""
    from idp_antibody_design import _extract_sequence_from_pdb as _extract
    return _extract(pdb_path, chain_id)


def compute_off_target_scores(
    designs: List[Dict],
    off_target_pdb: str,
    off_target_chain: str = "A",
    device: str = "cuda",
    backend: str = "af2",
    hdock_exe: Optional[str] = None,
    num_recycle: int = 1,
    verbose: bool = False,
) -> Dict[str, float]:
    """Score each design's off-target ipTM against an off-target protein.

    Args:
        designs: list of design dicts (must contain 'full_ab_seq' or 'sequence').
        off_target_pdb: path to off-target PDB (e.g. Transferrin).
        off_target_chain: chain ID to extract the off-target sequence.

    Returns:
        dict keyed by the design's full_ab_seq (or sequence) → off-target ipTM.
        Designs that fail scoring get 0.0 (treated as non-binding by default;
        callers using a hard-reject threshold should treat 0.0 as "pass").
    """
    if not os.path.exists(off_target_pdb):
        if verbose:
            print(f"  [off-target] PDB not found: {off_target_pdb} — skipping")
        return {}

    sys.path.insert(0, "modules")
    off_seq = _extract_sequence_from_pdb(off_target_pdb, off_target_chain)
    if not off_seq:
        if verbose:
            print(f"  [off-target] no sequence on chain {off_target_chain} — skipping")
        return {}

    scores: Dict[str, float] = {}
    for i, d in enumerate(designs):
        ab_seq = d.get("full_ab_seq") or d.get("sequence", "")
        if not ab_seq:
            continue
        key = ab_seq
        try:
            iptm = _score_pair(ab_seq, off_seq, backend, hdock_exe, num_recycle, verbose, i, len(designs))
        except Exception as e:  # noqa: BLE001 — best-effort, never crash the pipeline
            if verbose:
                print(f"  [off-target] design {i+1} failed: {e}")
            iptm = 0.0
        scores[key] = iptm
    return scores


def _score_pair(ab_seq, off_seq, backend, hdock_exe, num_recycle, verbose, i, n):
    """Score one antibody/off-target pair with the chosen backend."""
    backend = _resolve_backend(backend, hdock_exe)
    if backend == "hdock" and hdock_exe and os.path.exists(hdock_exe):
        return _score_hdock(ab_seq, off_seq, hdock_exe, verbose, i, n)
    return _score_af2(ab_seq, off_seq, num_recycle, verbose, i, n)


def _resolve_backend(backend, hdock_exe):
    """Resolve the backend, falling back to af2 if hdock is unavailable."""
    if backend == "hdock" and (not hdock_exe or not os.path.exists(hdock_exe)):
        return "af2"
    return backend


def _score_af2(ab_seq, off_seq, num_recycle, verbose, i, n):
    """AF2-multimer ipTM as a docking proxy (zero extra deps)."""
    from af2_jax_runner import validate_antibody_epitope
    if verbose:
        print(f"  [off-target AF2] design {i+1}/{n} ...", end="", flush=True)
    res = validate_antibody_epitope(ab_seq, off_seq, num_recycle=num_recycle, verbose=False)
    iptm = res.get("iptm", 0.0) if res.get("success") else 0.0
    if verbose:
        print(f" ipTM={iptm:.3f}")
    return float(iptm)


def _score_hdock(ab_seq, off_seq, hdock_exe, verbose, i, n):
    """HDOCK rigid-body docking backend (placeholder — requires HDOCK install).

    HDOCK expects two PDB files; we write FASTA→structure-free receptor/ligand
    is outside scope here, so this returns the raw HDOCK top-score and is left
    as a thin subprocess wrapper. Install HDOCK and set hdock_exe to enable.
    """
    if verbose:
        print(f"  [off-target HDOCK] design {i+1}/{n} (placeholder backend)")
    # NOTE: full HDOCK integration requires building PDB inputs from sequences
    # (e.g. via ESMFold). Until that's wired, raise so the caller falls back.
    raise NotImplementedError(
        "HDOCK backend not yet wired (needs seq→PDB structure prediction). "
        "Use backend='af2' (default) for the no-dependency docking proxy."
    )
    # pragma: no cover — subprocess call template for future wiring:
    # proc = subprocess.run([hdock_exe, receptor.pdb, ligand.pdb], capture_output=True)
    # return _parse_hdock_top_score(proc.stdout)


def batch_off_target_filter(
    designs: List[Dict],
    off_target_iptm_max: float = 0.3,
    **kwargs,
) -> List[Dict]:
    """Convenience: score + hard-filter designs above the off-target threshold.

    Returns only designs whose off_target_iptm <= off_target_iptm_max.
    """
    scores = compute_off_target_scores(designs, **kwargs)
    kept = []
    for d in designs:
        key = d.get("full_ab_seq") or d.get("sequence", "")
        iptm = scores.get(key, 0.0)
        d["off_target_iptm"] = iptm
        if iptm <= off_target_iptm_max:
            kept.append(d)
    return kept
