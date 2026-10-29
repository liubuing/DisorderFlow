"""无序度池化指标测试：验证 AUC/MCC/阳性率在完美分离与全阳性塌缩下的行为。

守护 disorderflow.disorder_metrics 评估管线：
完美分离时 AUC-ROC/AUC-PR/MCC 均为 1、
预测全为阳性（塌缩）时 MCC 必须为 0 且 positive_rate 暴露塌缩——
防止塌缩预测器靠命中率虚高蒙混过关。
"""
import pytest

from disorderflow.disorder_metrics import pooled_disorder_metrics


# 完美分离基线：三个核心指标必须同时为 1 且阳性率如实统计——防止指标实现方向错误。
def test_pooled_disorder_metrics_perfect_separation():
    metrics = pooled_disorder_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert metrics["disorder_auc_roc"] == pytest.approx(1.0)
    assert metrics["disorder_auc_pr"] == pytest.approx(1.0)
    assert metrics["disorder_mcc"] == pytest.approx(1.0)
    assert metrics["disorder_positive_rate"] == pytest.approx(0.5)


# 全阳性塌缩：MCC 必须退化为 0（而非 NaN 或虚高值），
# positive_rate=1 暴露塌缩——防止"全预测为无序"的退化模型得到好分数。
def test_pooled_disorder_metrics_exposes_all_positive_collapse():
    metrics = pooled_disorder_metrics([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0])
    assert metrics["disorder_mcc"] == 0.0
    assert metrics["disorder_positive_rate"] == 1.0
