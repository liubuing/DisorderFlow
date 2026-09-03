"""Generalized ensemble contact-robustness scoring for IDP targets.

Replaces the A-beta42-only ``idp_multiconf_scorer.score_robust`` with a
conformation-list-driven version: any target with multiple PDB conformations
(A-beta42, alpha-synuclein, tau) can be scored. Contact robustness is the 25th
percentile of per-conformation heavy-atom (Cb) contacts, so a design that
contacts only one conformation is not considered robust.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _cb_positions(pdb_path: str, chain_id: str) -> np.ndarray | None:
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    try:
        model = next(parser.get_structure("s", pdb_path).get_models())
    except Exception:  # noqa: BLE001
        return None
    if chain_id not in model:
        return None
    coords = []
    for residue in model[chain_id].get_residues():
        if residue.id[0] != " ":
            continue
        atom = "CB" if "CB" in residue else "CA"
        if atom in residue:
            coords.append(residue[atom].get_coord())
    if not coords:
        return None
    return np.asarray(coords, dtype=float)


def _kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    h = mobile_centered.T @ reference_centered
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return (mobile_centered @ rotation) + reference_center


def per_conformation_contacts(
        antibody_pdb: str,
        conformation_pdbs: Sequence[str],
        epitope_residues: Sequence[int],
        ab_chain: str = "H",
        antigen_chain: str = "A",
        cdr_regions: Sequence[tuple[int, int]] | None = None,
        contact_cutoff: float = 8.0,
        reference_index: int = 0) -> list[int | None]:
    """Heavy-atom contacts between an antibody and each antigen conformation.

    Each conformation is Kabsch-aligned onto the reference conformation over
    the epitope residues before counting contacts, so only epitope-relative
    geometry drives the score. ``cdr_regions`` restricts the antibody side to
    the design (CDR) residues.
    """
    ab = _cb_positions(antibody_pdb, ab_chain)
    if ab is None:
        return [None] * len(conformation_pdbs)
    if cdr_regions:
        cdr_mask = np.zeros(len(ab), dtype=bool)
        for start, end in cdr_regions:
            cdr_mask[start - 1:end] = True
        ab = ab[cdr_mask]
    antigen = [_cb_positions(p, antigen_chain) for p in conformation_pdbs]
    if any(a is None for a in antigen) or not antigen:
        return [None] * len(conformation_pdbs)
    if reference_index >= len(antigen):
        reference_index = 0
    reference_epi = antigen[reference_index][np.asarray(epitope_residues) - 1]

    results = []
    for conf in antigen:
        conf_epi = conf[np.asarray(epitope_residues) - 1]
        aligned = _kabsch_align(conf_epi, reference_epi)
        distances = np.linalg.norm(
            ab[:, None, :] - aligned[None, :, :], axis=-1)
        results.append(int((distances < contact_cutoff).sum()))
    return results


def robust_contacts(per_conf: Iterable[int | None],
                    robust_stat: str = "p25") -> dict:
    """Reduce per-conformation contact counts to a robust summary."""
    values = [int(v) for v in per_conf if v is not None]
    missing = sum(1 for v in per_conf if v is None)
    total = len(list(per_conf))
    if not values:
        return {"min": None, "median": None, "max": None, "robust": None,
                "n_missing": missing, "n_conformations": total,
                "per_conf": list(per_conf)}
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
        "n_conformations": total,
        "per_conf": list(per_conf),
    }
