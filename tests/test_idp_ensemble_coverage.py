"""IdP 系综覆盖度审计的契约测试：守护 Kabsch RMSD 的刚体不变性、抗原 RMSD 以抗体对齐坐标系
（而非全局对齐）计算，以及有效姿态数按重复分量折算——防止覆盖度指标被几何处理方式虚增。"""
import numpy as np

from scripts.audit_idp_ensemble_coverage import (
    antibody_aligned_antigen_rmsd,
    effective_pose_count,
    kabsch_rmsd,
)


# 刚体不变性：纯平移副本的 Kabsch RMSD 必须趋零，防止实现引入镜像或尺度误差。
def test_kabsch_rmsd_is_rigid_transform_invariant():
    reference = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    mobile = reference + np.asarray([4, -2, 3])
    assert kabsch_rmsd(reference, mobile) < 1e-8


# 对齐框架：整体平移后，以抗体对齐再量抗原 RMSD 必须不变——防止用全局对齐掩盖抗原的相对位移。
def test_antigen_rmsd_uses_antibody_alignment_frame():
    antibody = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    antigen = np.asarray([[2, 0, 0], [2, 1, 0]], dtype=float)
    moved_antibody = antibody + 3
    moved_antigen = antigen + 3
    assert antibody_aligned_antigen_rmsd(
        antibody, moved_antibody, antigen, moved_antigen) < 1e-8


# 重复折算：三个姿态两两互为重复分量时有效姿态数只能算 1，防止冗余构象冒充系综多样性。
def test_effective_pose_count_uses_duplicate_components():
    assert effective_pose_count(
        ["a", "b", "c"], ["1", "2", "3"],
        [["a", "b"], ["b", "c"], ["a", "c"]]) == 1
