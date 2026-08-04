"""Deterministic length-adaptive peptide backbone torsion perturbations."""

from __future__ import annotations

import copy
import math
import random

import numpy as np

from modules.peptide_conformer_ensemble import antibody_aligned_rmsd


BACKBONE = ("N", "CA", "C", "O")


def smooth_torsion_direction(sequence, seed):
    """Create a deterministic smooth unit-RMS phi/psi perturbation direction."""
    rng = random.Random(int(seed))
    length = len(sequence)
    if length < 4:
        raise ValueError("Torsion perturbation requires at least four residues")
    raw_phi = np.asarray([rng.gauss(0, 1) for _ in sequence], dtype=float)
    raw_psi = np.asarray([rng.gauss(0, 1) for _ in sequence], dtype=float)
    kernel = np.asarray([0.25, 0.5, 0.25])
    phi = np.convolve(np.pad(raw_phi, 1, mode="edge"), kernel, mode="valid")
    psi = np.convolve(np.pad(raw_psi, 1, mode="edge"), kernel, mode="valid")
    phi[0] = 0.0
    psi[-1] = 0.0
    for index, amino_acid in enumerate(sequence):
        if amino_acid == "P":
            phi[index] *= 0.2
    magnitude = math.sqrt(float(np.mean(np.concatenate([phi, psi]) ** 2)))
    if magnitude < 1e-8:
        raise ValueError("Degenerate torsion perturbation direction")
    return phi / magnitude, psi / magnitude


def chain_backbone_coordinates(model, antibody_chains, peptide_chain):
    antibody = []
    peptide = []
    for chain in model:
        target = antibody if chain.id in antibody_chains else (
            peptide if chain.id == peptide_chain else None)
        if target is None:
            continue
        for residue in chain:
            for atom_name in BACKBONE:
                if atom_name in residue:
                    target.append(np.asarray(residue[atom_name].coord, dtype=float))
    if len(antibody) < 3 or len(peptide) < 3:
        raise ValueError("Incomplete antibody or peptide backbone")
    return np.asarray(antibody), np.asarray(peptide)


def standard_residues(chain):
    return [
        residue for residue in chain
        if residue.id[0] == " " and all(name in residue for name in ("N", "CA", "C"))
    ]


def perturb_model(model, peptide_chain, phi_direction, psi_direction, scale_degrees):
    """Return a copied model with peptide phi/psi values rotated by a fixed scale."""
    perturbed = copy.deepcopy(model)
    chain = perturbed[peptide_chain]
    chain.atom_to_internal_coordinates()
    residues = standard_residues(chain)
    if len(residues) != len(phi_direction) or len(residues) != len(psi_direction):
        raise ValueError("Torsion direction length does not match peptide residues")
    applied = []
    for index, residue in enumerate(residues):
        internal = residue.internal_coord
        phi_delta = float(phi_direction[index] * scale_degrees)
        psi_delta = float(psi_direction[index] * scale_degrees)
        if internal.pick_angle("phi") is not None and phi_delta:
            internal.bond_rotate("phi", phi_delta)
        if internal.pick_angle("psi") is not None and psi_delta:
            internal.bond_rotate("psi", psi_delta)
        applied.append({"residue": residue.id[1], "phi_delta": phi_delta,
                        "psi_delta": psi_delta})
    chain.internal_to_atom_coordinates()
    return perturbed, applied


def search_target_rmsd(model, antibody_chains, peptide_chain, sequence, seed,
                       target_min=1.5, target_max=3.0, target_center=2.25,
                       maximum_scale_degrees=60.0, grid_step_degrees=2.0):
    """Find the lowest-scale deterministic torsion perturbation in a target RMSD tier."""
    reference_antibody, reference_peptide = chain_backbone_coordinates(
        model, antibody_chains, peptide_chain)
    phi, psi = smooth_torsion_direction(sequence, seed)
    candidates = []
    scale = float(grid_step_degrees)
    while scale <= float(maximum_scale_degrees) + 1e-8:
        candidate, applied = perturb_model(model, peptide_chain, phi, psi, scale)
        candidate_antibody, candidate_peptide = chain_backbone_coordinates(
            candidate, antibody_chains, peptide_chain)
        antibody_rmsd, peptide_rmsd = antibody_aligned_rmsd(
            reference_antibody, candidate_antibody,
            reference_peptide, candidate_peptide)
        candidates.append({
            "scale_degrees": scale,
            "antibody_rmsd": antibody_rmsd,
            "peptide_rmsd": peptide_rmsd,
            "model": candidate,
            "applied": applied,
        })
        scale += float(grid_step_degrees)
    in_tier = [
        candidate for candidate in candidates
        if target_min <= candidate["peptide_rmsd"] <= target_max]
    if not in_tier:
        return None, [{key: value for key, value in candidate.items() if key != "model"}
                      for candidate in candidates]
    selected = min(
        in_tier,
        key=lambda candidate: (
            abs(candidate["peptide_rmsd"] - target_center),
            candidate["scale_degrees"]),
    )
    trace = [{key: value for key, value in candidate.items() if key != "model"}
             for candidate in candidates]
    return selected, trace
