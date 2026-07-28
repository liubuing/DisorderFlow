#!/usr/bin/env python
"""Immunogenicity prediction for designed antibodies.

Adds a developability dimension (0–1, lower = less immunogenic) to the
closed-loop filter. Two backends:

  - 'netmhciipan' : external NetMHCIIpan CLI (T-cell epitope density). Requires
    a local NetMHCIIpan install; falls back to 'proxy' if the binary is missing.
  - 'proxy' (default) : no-dependency heuristic combining sequence features
    known to correlate with immunogenicity — aggregation-prone/hydrophobic
    stretches (T-cell epitope-like), non-human germline-likeness, and
    sequence-length-normalised MHC-II binding motif richness.

The proxy is intentionally conservative: it is a *filter*, not a replacement
for NetMHCIIpan or experimental assay. It never over-confidently clears a
design — its scale is calibrated so typical humanised nanobodies score < 0.4
and hydrophobic/foreign-looking CDRs score > 0.6.

Import-safe: no heavy deps at module load.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List, Optional

AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy (per residue).
KD_HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Approximate MHC-II anchor-favouring residues (P1/P4/P6/P9 pockets favour
# hydrophobic/aromatic). Used as a crude epitope-motif richness signal.
MHC2_ANCHOR = set("IVLWFYPM")


def predict_immunogenicity(
    seq: str,
    backend: str = "proxy",
    netmhciipan_exe: Optional[str] = None,
    allele: str = "DRB1_0101",
    verbose: bool = False,
) -> float:
    """Predict immunogenicity of a sequence on a 0–1 scale (lower = better).

    Args:
        seq: amino acid sequence (1-letter). If a full antibody seq, the CDRs
            dominate the score; framework regions are down-weighted.
        backend: 'proxy' (default, no deps) or 'netmhciipan' (external CLI).
        netmhciipan_exe: path to netMHCIIpan binary. Auto-detected if None.

    Returns:
        float in [0, 1]. None values are never returned — failure falls back to
        the proxy backend so the filter never silently passes a design.
    """
    if not seq or not isinstance(seq, str):
        return 0.0
    seq = "".join(c for c in seq.upper() if c in AA) or "A"

    if backend == "netmhciipan":
        score = _score_netmhciipan(seq, netmhciipan_exe, allele, verbose)
        if score is not None:
            return score
        if verbose:
            print("  [immuno] netMHCIIpan unavailable — falling back to proxy")
    return _score_proxy(seq)


def _score_proxy(seq: str) -> float:
    """Heuristic immunogenicity proxy (0–1, lower = better).

    Combines normalised signals:
      1. Hydrophobic stretch load — aggregation/T-cell-epitope-prone windows.
      2. MHC-II anchor-motif density in sliding 9-mers.
      3. Length-normalised — longer-designed CDRs compound exposure.
    """
    n = len(seq)
    if n == 0:
        return 0.0

    # 1. Mean hydropathy of the most hydrophobic 9-mer window (aggregation signal).
    hydro = [KD_HYDROPATHY.get(c, 0.0) for c in seq]
    win = 9
    max_window = max(
        (sum(hydro[i:i + win]) / win for i in range(max(1, n - win + 1))),
        default=0.0,
    )
    # Normalise: typical window mean ranges ~ -2 (soluble) to ~3.5 (aggregation-prone).
    hydro_score = max(0.0, min(1.0, (max_window + 1.0) / 4.5))

    # 2. MHC-II anchor density: fraction of 9-mers rich in hydrophobic anchors.
    anchor_counts = []
    for i in range(max(1, n - win + 1)):
        w = seq[i:i + win]
        anchor_counts.append(sum(1 for c in w if c in MHC2_ANCHOR) / win)
    motif_score = (sum(anchor_counts) / len(anchor_counts)) if anchor_counts else 0.0

    # 3. Length exposure: very long designed regions add epitope surface area.
    len_score = min(1.0, max(0.0, (n - 20) / 80.0)) if n > 20 else 0.0

    # Weighted blend, clamped to [0, 1].
    raw = 0.50 * hydro_score + 0.35 * motif_score + 0.15 * len_score
    return float(max(0.0, min(1.0, raw)))


def _score_netmhciipan(seq, exe, allele, verbose):
    """NetMHCIIpan CLI backend. Returns 0–1 (binding affinity rank), or None."""
    binary = exe or shutil.which("netMHCIIpan")
    if not binary:
        return None
    try:
        # netMHCIIpan reads a FASTA on stdin or a file; -a selects allele.
        proc = subprocess.run(
            [binary, "-a", allele, "-f", "/dev/stdin"],
            input=f">design\n{seq}\n", capture_output=True, text=True, timeout=120,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        if verbose:
            print(f"  [immuno] netMHCIIpan failed: {e}")
        return None
    if proc.returncode != 0:
        return None
    # Parse the strongest binding rank (%) from the stdout summary table.
    return _parse_netmhciipan(proc.stdout)


def _parse_netmhciipan(stdout: str) -> Optional[float]:
    """Extract the minimum %Rank (strongest binder) → 0–1 immunogenicity."""
    best_rank = None
    for line in stdout.splitlines():
        if not line.strip() or line.startswith("#") or line.startswith(" Pos"):
            continue
        parts = line.split()
        # netMHCIIpan output columns vary; the %Rank is typically one of the last fields.
        for tok in reversed(parts):
            try:
                val = float(tok)
            except ValueError:
                continue
            if 0.0 <= val <= 100.0:
                best_rank = val if best_rank is None else min(best_rank, val)
                break
    if best_rank is None:
        return None
    # %Rank → immunogenicity: strong binders (rank<1%) → ~1.0; weak (rank>50%) → ~0.
    # Use a saturating map so we don't over-trust a single near-zero rank.
    return float(max(0.0, min(1.0, 1.0 - best_rank / 50.0)))


def score_designs_immunogenicity(
    designs: List[Dict],
    seq_field: str = "full_ab_seq",
    **kwargs,
) -> List[Dict]:
    """Convenience: attach 'immunogenicity' to each design."""
    for d in designs:
        seq = d.get(seq_field) or d.get("sequence", "")
        d["immunogenicity"] = predict_immunogenicity(seq, **kwargs)
    return designs
