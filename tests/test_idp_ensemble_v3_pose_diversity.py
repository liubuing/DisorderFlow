"""IDP 集成 v3 pose 多样性审计测试：验证 Kabsch 指标的刚体平移不变性。

守护 scripts.audit_idp_ensemble_v3_pose_diversity.kabsch_metrics：
整体平移后骨架与 H3 的 Kabsch RMSD 必须为 0——
防止 pose 多样性把刚体位移误计为构象差异（多样性虚高）。
"""
import numpy as np

from scripts.audit_idp_ensemble_v3_pose_diversity import kabsch_metrics


# 刚体不变量：纯平移不改变 Kabsch 对齐后的 RMSD（framework 与 h3 均 < 1e-12）——
# 防止未做最优叠加的坐标差被当成真实构象多样性。
def test_kabsch_metrics_remove_rigid_translation():
    framework = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    h3 = np.array([[0, 0, 1]], dtype=float)
    translation = np.array([3, -2, 4], dtype=float)
    framework_rmsd, h3_rmsd = kabsch_metrics(
        framework, framework + translation, h3, h3 + translation
    )
    assert framework_rmsd < 1e-12
    assert h3_rmsd < 1e-12
