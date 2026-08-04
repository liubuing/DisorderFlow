"""Contact splitting and recovery metrics for T2 peptide perturbation tests."""

from __future__ import annotations

import hashlib


def split_contact_pairs(contact_pairs, record_id, restraint_fraction=0.5):
    """Deterministically split residue contacts into supplied and held-out sets."""
    pairs = sorted(set(contact_pairs), key=repr)
    if len(pairs) < 2:
        raise ValueError("T2 requires at least two native residue contacts")
    ranked = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{record_id}|{pair!r}".encode("utf-8")).hexdigest(),
    )
    count = max(1, min(len(ranked) - 1, round(len(ranked) * restraint_fraction)))
    supplied = set(ranked[:count])
    held_out = set(ranked[count:])
    return supplied, held_out


def recovery_metrics(initial_rmsd, final_rmsd, initial_held_out, final_held_out):
    return {
        "rmsd_recovery_angstrom": float(initial_rmsd - final_rmsd),
        "held_out_contact_recovery": float(final_held_out - initial_held_out),
        "rmsd_improved": bool(final_rmsd < initial_rmsd),
        "held_out_contacts_improved": bool(final_held_out > initial_held_out),
    }
