"""Geometry and quality-control primitives for local peptide ensembles."""

from __future__ import annotations

import math

import numpy as np


BACKBONE = frozenset({"N", "CA", "C", "O"})


def kabsch_rmsd(reference, mobile):
    """Return RMSD after fitting mobile coordinates onto reference coordinates."""
    reference = np.asarray(reference, dtype=float)
    mobile = np.asarray(mobile, dtype=float)
    if reference.shape != mobile.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching (N, 3) shapes")
    if len(reference) < 3:
        raise ValueError("At least three coordinates are required")
    reference_center = reference.mean(axis=0)
    mobile_center = mobile.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    fitted = (mobile - mobile_center) @ rotation + reference_center
    return float(np.sqrt(np.mean(np.sum((fitted - reference) ** 2, axis=1))))


def antibody_aligned_rmsd(reference_antibody, mobile_antibody,
                          reference_peptide, mobile_peptide):
    """Return antibody fit RMSD and peptide RMSD in the antibody-aligned frame."""
    reference_antibody = np.asarray(reference_antibody, dtype=float)
    mobile_antibody = np.asarray(mobile_antibody, dtype=float)
    reference_peptide = np.asarray(reference_peptide, dtype=float)
    mobile_peptide = np.asarray(mobile_peptide, dtype=float)
    if reference_antibody.shape != mobile_antibody.shape:
        raise ValueError("Antibody coordinate sets do not match")
    if reference_peptide.shape != mobile_peptide.shape:
        raise ValueError("Peptide coordinate sets do not match")
    reference_center = reference_antibody.mean(axis=0)
    mobile_center = mobile_antibody.mean(axis=0)
    covariance = (mobile_antibody - mobile_center).T @ (
        reference_antibody - reference_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    fitted_antibody = (mobile_antibody - mobile_center) @ rotation + reference_center
    fitted_peptide = (mobile_peptide - mobile_center) @ rotation + reference_center
    antibody_rmsd = float(np.sqrt(np.mean(np.sum(
        (fitted_antibody - reference_antibody) ** 2, axis=1))))
    peptide_rmsd = float(np.sqrt(np.mean(np.sum(
        (fitted_peptide - reference_peptide) ** 2, axis=1))))
    return antibody_rmsd, peptide_rmsd


def residue_contact_pairs(records, left_selector, right_selector, cutoff=4.5):
    """Return residue-pair identities with at least one heavy-atom contact."""
    left = [record for record in records if left_selector(record)]
    right = [record for record in records if right_selector(record)]
    pairs = set()
    if not left or not right:
        return pairs
    left_xyz = np.asarray([record["coord"] for record in left], dtype=float)
    right_xyz = np.asarray([record["coord"] for record in right], dtype=float)
    left_indices, right_indices = np.where(
        np.linalg.norm(left_xyz[:, None, :] - right_xyz[None, :, :], axis=-1)
        <= cutoff)
    for left_index, right_index in zip(left_indices, right_indices):
        pairs.add((left[left_index]["residue_key"], right[right_index]["residue_key"]))
    return pairs


def contact_retention(native_pairs, observed_pairs):
    native_pairs = set(native_pairs)
    observed_pairs = set(observed_pairs)
    if not native_pairs:
        raise ValueError("Native contact set is empty")
    return len(native_pairs & observed_pairs) / len(native_pairs)


def nonlocal_clash_count(records, cutoff=1.5):
    """Count heavy-atom clashes excluding same and adjacent residues."""
    from scipy.spatial import cKDTree

    if not records:
        return 0
    coordinates = np.asarray([record["coord"] for record in records], dtype=float)
    clashes = 0
    for left_index, right_index in cKDTree(coordinates).query_pairs(cutoff):
        left = records[left_index]
        right = records[right_index]
        if left["residue_key"] == right["residue_key"]:
            continue
        if left["chain"] == right["chain"] and abs(
                left["residue_index"] - right["residue_index"]) <= 1:
            continue
        clashes += 1
    return clashes


def ensemble_dispersion(coordinate_sets):
    """Summarize internal peptide-backbone diversity after peptide fitting."""
    coordinate_sets = [np.asarray(value, dtype=float) for value in coordinate_sets]
    if len(coordinate_sets) < 2:
        return {"mean_pairwise_rmsd": 0.0, "max_pairwise_rmsd": 0.0}
    values = []
    for left_index in range(len(coordinate_sets)):
        for right_index in range(left_index + 1, len(coordinate_sets)):
            values.append(kabsch_rmsd(
                coordinate_sets[left_index], coordinate_sets[right_index]))
    return {
        "mean_pairwise_rmsd": float(np.mean(values)),
        "max_pairwise_rmsd": float(np.max(values)),
    }


def conformer_passes(metrics, thresholds):
    checks = {
        "finite_energy": math.isfinite(metrics["potential_energy_kj_mol"]),
        "antibody_preserved": (
            metrics["antibody_backbone_rmsd"] <= thresholds["max_antibody_rmsd"]),
        "minimum_peptide_displacement": (
            metrics["peptide_backbone_rmsd"] >= thresholds["min_peptide_rmsd"]),
        "maximum_peptide_displacement": (
            metrics["peptide_backbone_rmsd"] <= thresholds["max_peptide_rmsd"]),
        "native_contacts_retained": (
            metrics["native_contact_retention"] >= thresholds["min_contact_retention"]),
        "zero_nonlocal_clashes": metrics["nonlocal_clashes_lt_1_5A"] == 0,
        "ca_geometry_not_worsened": (
            metrics["ca_geometry_outliers"] <= metrics["reference_ca_geometry_outliers"]),
        "cn_geometry_not_worsened": (
            metrics["cn_geometry_outliers"] <= metrics["reference_cn_geometry_outliers"]),
    }
    return checks, all(checks.values())
