import pytest

from disorderflow.disorder_metrics import pooled_disorder_metrics


def test_pooled_disorder_metrics_perfect_separation():
    metrics = pooled_disorder_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert metrics["disorder_auc_roc"] == pytest.approx(1.0)
    assert metrics["disorder_auc_pr"] == pytest.approx(1.0)
    assert metrics["disorder_mcc"] == pytest.approx(1.0)
    assert metrics["disorder_positive_rate"] == pytest.approx(0.5)


def test_pooled_disorder_metrics_exposes_all_positive_collapse():
    metrics = pooled_disorder_metrics([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0])
    assert metrics["disorder_mcc"] == 0.0
    assert metrics["disorder_positive_rate"] == 1.0
