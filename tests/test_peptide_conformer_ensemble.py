import numpy as np

from modules.peptide_conformer_ensemble import (
    antibody_aligned_rmsd,
    conformer_passes,
    contact_retention,
    ensemble_dispersion,
    nonlocal_clash_count,
    residue_contact_pairs,
)


def test_antibody_alignment_separates_rigid_motion_from_peptide_change():
    antibody = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    peptide = np.array([[0, 0, 1], [1, 0, 1]], dtype=float)
    moved_antibody = antibody + 4
    moved_peptide = peptide + 4
    moved_peptide[1, 2] += 1
    antibody_rmsd, peptide_rmsd = antibody_aligned_rmsd(
        antibody, moved_antibody, peptide, moved_peptide)
    assert antibody_rmsd < 1e-12
    assert peptide_rmsd > 0.7


def test_contact_pairs_and_retention_use_residue_identity():
    records = [
        {"chain": "H", "residue_index": 0, "residue_key": ("H", "1"),
         "coord": np.array([0, 0, 0])},
        {"chain": "P", "residue_index": 0, "residue_key": ("P", "1"),
         "coord": np.array([0, 0, 4])},
    ]
    pairs = residue_contact_pairs(
        records, lambda row: row["chain"] == "H", lambda row: row["chain"] == "P")
    assert pairs == {(('H', '1'), ('P', '1'))}
    assert contact_retention(pairs, pairs) == 1.0


def test_nonlocal_clashes_ignore_adjacent_residues():
    records = [
        {"chain": "P", "residue_index": 0, "residue_key": ("P", "1"),
         "coord": np.array([0, 0, 0])},
        {"chain": "P", "residue_index": 1, "residue_key": ("P", "2"),
         "coord": np.array([2, 0, 0])},
        {"chain": "P", "residue_index": 3, "residue_key": ("P", "4"),
         "coord": np.array([2, 0, 0])},
    ]
    assert nonlocal_clash_count(records) == 1


def test_ensemble_dispersion_detects_shape_change():
    first = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    second = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=float)
    assert ensemble_dispersion([first, second])["mean_pairwise_rmsd"] > 0


def test_native_geometry_outlier_is_allowed_when_not_worsened():
    metrics = {
        "potential_energy_kj_mol": -1.0,
        "antibody_backbone_rmsd": 0.1,
        "peptide_backbone_rmsd": 1.0,
        "native_contact_retention": 0.8,
        "nonlocal_clashes_lt_1_5A": 0,
        "ca_geometry_outliers": 1,
        "cn_geometry_outliers": 0,
        "reference_ca_geometry_outliers": 1,
        "reference_cn_geometry_outliers": 0,
    }
    thresholds = {
        "max_antibody_rmsd": 0.5,
        "min_peptide_rmsd": 0.05,
        "max_peptide_rmsd": 3.0,
        "min_contact_retention": 0.5,
    }
    _, passed = conformer_passes(metrics, thresholds)
    assert passed is True
