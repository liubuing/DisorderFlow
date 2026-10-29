"""构象校准指标测试：验证 per-residue RMSF 的刚体不变量与校准评分的方向性。

守护构象集合校准管线：
RMSF 对刚体旋转平移必须为零（只度量内部涨落）、对内部运动必须敏感、
calibration_metrics 在预测与观测柔韧性完全一致时取满分、
样本不足时 verdict 必须 fail-closed（不给"校准成功"的结论）。
"""
import numpy as np


# 刚体不变量：整体旋转+平移后的构象逐残基 RMSF 必须为 0——
# 防止 RMSF 混入绝对坐标差异而错报柔性。
def test_rmsf_is_invariant_to_rigid_transform():
    from disorderflow.utils.conformation_calibration import per_residue_rmsf

    reference = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = reference @ rotation.T + np.array([7.0, -3.0, 2.0])
    assert np.allclose(per_residue_rmsf(np.stack([reference, moved])), 0.0, atol=1e-7)


# 灵敏度：单个残基的真实内部位移必须在 RMSF 中反映为该残基取最大且显著大于 0——
# 防止平均化抹平局部运动信号。
def test_rmsf_detects_internal_motion():
    from disorderflow.utils.conformation_calibration import per_residue_rmsf

    reference = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ])
    moved = reference.copy()
    moved[-1] += [2.0, 1.0, 0.0]
    rmsf = per_residue_rmsf(np.stack([reference, moved]))
    assert rmsf[-1] == rmsf.max()
    assert rmsf[-1] > 0.1


# 方向性校准：预测柔性与观测柔性单调一致时 spearman/top20_recall/flexible_auc 均为满分——
# 防止指标方向或排名实现写反（如把刚性当柔性奖励）。
def test_calibration_metrics_reward_matching_flexibility():
    from disorderflow.utils.conformation_calibration import calibration_metrics

    predicted = np.arange(20, dtype=float)
    observed = predicted**2
    uncertainty = predicted[::-1]
    metrics = calibration_metrics(predicted, observed, uncertainty)
    assert metrics['spearman'] == 1.0
    assert metrics['top20_recall'] == 1.0
    assert metrics['flexible_auc'] == 1.0


# fail-closed 语义：样本量不足（9 < 10）时即使各项指标都很高，
# verdict 也必须是 insufficient_evidence——防止小样本偶然通过校准门槛。
def test_calibration_verdict_fails_closed_on_small_sample():
    from disorderflow.utils.conformation_calibration import calibration_verdict

    results = [
        {
            'spearman': 0.9,
            'top20_recall': 0.9,
            'flexible_auc': 0.9,
            'partial_spearman': 0.9,
            'uncertainty_spearman': 0.1,
        }
        for _ in range(9)
    ]
    assert calibration_verdict(results)['verdict'] == 'insufficient_evidence'
