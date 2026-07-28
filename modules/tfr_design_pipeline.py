#!/usr/bin/env python
"""TfR (Transferrin Receptor) Nanobody Design Pipeline — one-click BFN design.

Wraps the IDP-aware antibody design pipeline
(`idp_antibody_design.run_idp_antibody_design`) with TfR-specific defaults and
adds a negative-design hook (competitive specificity against Transferrin).

Application context (see docs/海参肽-TfR纳米抗体BBB递送系统_研究方案.md):
  - Design a VHH/nanobody that binds TfR (BBB shuttle) on an ordered epitope.
  - Negative design: avoid the Transferrin (Tf) competitive binding site so the
    shuttle does not block Tf binding in vivo.

This module is target-aware but engine-agnostic: all design is done by the BFN
(`run_bfn_design`). Heavy lifting (disorder prediction, epitope building, CDR
design, AF2 validation, composite ranking) is delegated to the existing
pipeline. We only add:
  1. Sensible TfR defaults (chain, epitope windows, disorder threshold).
  2. A negative-design scoring pass that re-ranks designs by off-target ipTM
     against Transferrin (lower = better specificity).

No GPU/checkpoint is imported at module load — everything is lazy so the module
can be unit-tested on the ranking logic alone.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Optional

# Default TfR / Tf structures (populated by scripts/download_tfr_targets.py).
# These are placeholders; the actual files must be downloaded.
TFR_TARGETS_DIR = os.path.join("data", "tfr_targets")
DEFAULT_TFR_PDB = os.path.join(TFR_TARGETS_DIR, "6GZV.pdb")       # TfR-Tf complex
DEFAULT_TFR_CHAIN = "A"
DEFAULT_TF_PDB = os.path.join(TFR_TARGETS_DIR, "1A8E.pdb")        # apo Transferrin
DEFAULT_TF_CHAIN = "A"
DEFAULT_SCAFFOLD_PDB = os.path.join("data", "misfolding_targets", "5IMK.pdb")
DEFAULT_SCAFFOLD_CHAIN = "B"
DEFAULT_CDR_SPEC = "B:26-33,51-58,97-113"  # nanobody CDR1/2/3 (IMGT-ish)


def run_tfr_design(
    target_pdb: str = DEFAULT_TFR_PDB,
    target_chain: str = DEFAULT_TFR_CHAIN,
    scaffold_pdb: str = DEFAULT_SCAFFOLD_PDB,
    scaffold_chain: str = DEFAULT_SCAFFOLD_CHAIN,
    cdr_spec: str = DEFAULT_CDR_SPEC,
    disorder_threshold: float = 0.3,
    num_segments: int = 3,
    num_samples: int = 20,
    stochastic: bool = True,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    use_af2: bool = False,
    af2_num_recycle: int = 3,
    use_af2_jax: bool = True,
    colabfold_exe: str = "colabfold_batch",
    fixbb: bool = False,
    negative_design: bool = True,
    off_target_pdb: Optional[str] = DEFAULT_TF_PDB,
    off_target_chain: str = DEFAULT_TF_CHAIN,
    off_target_iptm_max: float = 0.3,
    top_n: int = 20,
    verbose: bool = True,
) -> Dict:
    """One-click TfR nanobody design via BFN.

    Stages (all BFN-driven):
      1. Disorder analysis on TfR → ordered epitope segments
      2. BFN CDR design on each ordered segment (Complex mode by default)
      3. (optional) AF2 multimer validation for design-specific confidence
      4. (optional) Negative design: re-rank by off-target (Transferrin) ipTM
      5. Return Top-N candidates

    Args:
        negative_design: if True, compute off-target ipTM against Transferrin
            and demote/promote designs accordingly (see modules/off_target_docking.py).
        off_target_iptm_max: hard-reject designs whose off-target ipTM exceeds this.

    Returns:
        dict with keys: ranked_designs (top-N), report, output_dir, off_target_scores.
    """
    sys.path.insert(0, ".")
    sys.path.insert(0, "modules")
    from idp_antibody_design import run_idp_antibody_design

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="tfr_design_")
    os.makedirs(output_dir, exist_ok=True)

    context_chains = [] if fixbb else None  # None = Complex (epitope visible)

    result = run_idp_antibody_design(
        target_pdb=target_pdb,
        target_chain=target_chain,
        scaffold_pdb=scaffold_pdb,
        scaffold_chain=scaffold_chain,
        disorder_threshold=disorder_threshold,
        num_segments=num_segments,
        num_samples=num_samples,
        stochastic=stochastic,
        output_dir=output_dir,
        device=device,
        verbose=verbose,
        use_af2=use_af2,
        af2_num_recycle=af2_num_recycle,
        use_af2_jax=use_af2_jax,
        colabfold_exe=colabfold_exe,
        context_chains=context_chains,
    )

    ranked = result.get("ranked_designs", [])
    off_target_scores: Dict[str, float] = {}

    # ── Negative design: penalise designs that also bind Transferrin ──
    if negative_design and ranked and off_target_pdb and os.path.exists(off_target_pdb):
        if verbose:
            print("\n[Negative Design] Computing off-target (Transferrin) ipTM...")
        try:
            from off_target_docking import compute_off_target_scores
            off_target_scores = compute_off_target_scores(
                ranked, off_target_pdb, off_target_chain,
                device=device, verbose=verbose,
            )
            ranked = apply_negative_design_ranking(
                ranked, off_target_scores, off_target_iptm_max)
        except Exception as e:  # noqa: BLE001 — docking best-effort, non-fatal
            if verbose:
                print(f"  [Negative Design] skipped (off-target scoring failed: {e})")

    # Re-assign ranks after negative-design re-ranking
    for i, d in enumerate(ranked):
        d["rank"] = i + 1

    result["ranked_designs"] = ranked[:top_n]
    result["off_target_scores"] = off_target_scores
    result["report"] = format_tfr_report(ranked[:top_n])

    if verbose:
        print(result["report"])
    return result


def apply_negative_design_ranking(
    ranked: List[Dict],
    off_target_scores: Dict[str, float],
    off_target_iptm_max: float = 0.3,
) -> List[Dict]:
    """Re-rank designs: hard-reject off-target binders, then sort by specificity.

    A design's "specificity" = on-target composite − off-target ipTM (clamped).
    Designs are demoted when their off-target ipTM exceeds the threshold.

    Args:
        off_target_scores: keyed by design sequence (or a stable id) → off-target ipTM.

    Returns:
        re-ranked list (off-target binders moved to the bottom with a flag).
    """
    enriched = []
    for d in ranked:
        key = d.get("full_ab_seq") or d.get("sequence", "")
        off_iptm = off_target_scores.get(key, 0.0)
        d["off_target_iptm"] = off_iptm
        d["off_target_rejected"] = off_iptm > off_target_iptm_max
        # Specificity composite: keep on-target score but subtract off-target signal.
        d["specificity_score"] = float(d.get("composite_score", 0.0)) - off_iptm * 0.5
        enriched.append(d)

    # Sort: non-rejected first (by specificity), then rejected last (by off-target asc).
    enriched.sort(
        key=lambda d: (d["off_target_rejected"], -d["specificity_score"], d["off_target_iptm"])
    )
    return enriched


def format_tfr_report(ranked: List[Dict]) -> str:
    """Human-readable TfR design report (Top-N candidates)."""
    if not ranked:
        return "No TfR designs generated."
    lines = [
        "=" * 80,
        f"  TfR Nanobody Design — Top {len(ranked)} Candidates (BFN + negative design)",
        "=" * 80,
        f"  {'Rank':<5} {'Spec':<7} {'OnTgt':<8} {'OffTgt':<8} {'Flag':<8} {'CDR sequence'}",
        "  " + "─" * 70,
    ]
    for d in ranked:
        seq = d.get("sequence", "")
        if len(seq) > 40:
            seq = seq[:18] + "…" + seq[-18:]
        flag = "REJECT" if d.get("off_target_rejected") else "ok"
        lines.append(
            f"  {d['rank']:<5} "
            f"{d.get('specificity_score', 0):<7.3f} "
            f"{d.get('composite_score', 0):<8.3f} "
            f"{d.get('off_target_iptm', 0):<8.3f} "
            f"{flag:<8} {seq}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)
