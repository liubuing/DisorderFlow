import numpy as np


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


def test_calibration_metrics_reward_matching_flexibility():
    from disorderflow.utils.conformation_calibration import calibration_metrics

    predicted = np.arange(20, dtype=float)
    observed = predicted**2
    uncertainty = predicted[::-1]
    metrics = calibration_metrics(predicted, observed, uncertainty)
    assert metrics['spearman'] == 1.0
    assert metrics['top20_recall'] == 1.0
    assert metrics['flexible_auc'] == 1.0


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
