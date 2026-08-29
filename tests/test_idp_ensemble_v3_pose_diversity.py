import numpy as np

from scripts.audit_idp_ensemble_v3_pose_diversity import kabsch_metrics


def test_kabsch_metrics_remove_rigid_translation():
    framework = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    h3 = np.array([[0, 0, 1]], dtype=float)
    translation = np.array([3, -2, 4], dtype=float)
    framework_rmsd, h3_rmsd = kabsch_metrics(
        framework, framework + translation, h3, h3 + translation
    )
    assert framework_rmsd < 1e-12
    assert h3_rmsd < 1e-12
