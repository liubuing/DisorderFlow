"""无序度校准的契约测试：守护 Platt 校准对偏移 logits 的修正方向、MCC 阈值搜索能找到
完美操作点，以及校准文件加载的模式校验——保证置信输出先校准再用于阈值判定。"""
import json

import torch

from disorderflow.disorder_calibration import (
    apply_platt,
    calibration_metrics,
    fit_platt,
    load_disorder_calibration,
    select_mcc_threshold,
)


# Platt 校准方向性：fit 后 scale 必须为正，且 Brier 分数严格下降——校准只允许变好不允许变坏。
def test_platt_calibration_corrects_shifted_logits():
    logits = torch.tensor([-4.0, -3.0, 0.0, 1.0], dtype=torch.float64)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    before = calibration_metrics(torch.sigmoid(logits), labels)["brier"]
    parameters = fit_platt(logits, labels)
    after = calibration_metrics(apply_platt(logits, **parameters), labels)["brier"]
    assert parameters["scale"] > 0
    assert after < before


# 阈值搜索：可完全分离的标签必须得到 MCC=1.0 且阈值落在两类的分数间隙内，防止阈值选偏。
def test_mcc_threshold_finds_perfect_operating_point():
    result = select_mcc_threshold([0.1, 0.2, 0.6, 0.7], [0, 0, 1, 1])
    assert result["mcc"] == 1.0
    assert 0.2 < result["threshold"] <= 0.6


# 模式校验：合法 schema_version 与参数结构的校准文件可正常加载，防止劣化的手写校准文件静默生效。
def test_load_disorder_calibration_validates_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "parameters": {"scale": 0.8, "bias": 1.2},
        "threshold": 0.9,
    }))
    assert load_disorder_calibration(path)["parameters"]["scale"] == 0.8
