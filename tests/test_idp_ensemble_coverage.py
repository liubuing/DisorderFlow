import numpy as np

from scripts.audit_idp_ensemble_coverage import (
    antibody_aligned_antigen_rmsd,
    effective_pose_count,
    kabsch_rmsd,
)


def test_kabsch_rmsd_is_rigid_transform_invariant():
    reference = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    mobile = reference + np.asarray([4, -2, 3])
    assert kabsch_rmsd(reference, mobile) < 1e-8


def test_antigen_rmsd_uses_antibody_alignment_frame():
    antibody = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    antigen = np.asarray([[2, 0, 0], [2, 1, 0]], dtype=float)
    moved_antibody = antibody + 3
    moved_antigen = antigen + 3
    assert antibody_aligned_antigen_rmsd(
        antibody, moved_antibody, antigen, moved_antigen) < 1e-8


def test_effective_pose_count_uses_duplicate_components():
    assert effective_pose_count(
        ["a", "b", "c"], ["1", "2", "3"],
        [["a", "b"], ["b", "c"], ["a", "c"]]) == 1
