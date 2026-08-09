import json

import torch

from disorderflow.disorder_calibration import (
    apply_platt,
    calibration_metrics,
    fit_platt,
    load_disorder_calibration,
    select_mcc_threshold,
)


def test_platt_calibration_corrects_shifted_logits():
    logits = torch.tensor([-4.0, -3.0, 0.0, 1.0], dtype=torch.float64)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    before = calibration_metrics(torch.sigmoid(logits), labels)["brier"]
    parameters = fit_platt(logits, labels)
    after = calibration_metrics(apply_platt(logits, **parameters), labels)["brier"]
    assert parameters["scale"] > 0
    assert after < before


def test_mcc_threshold_finds_perfect_operating_point():
    result = select_mcc_threshold([0.1, 0.2, 0.6, 0.7], [0, 0, 1, 1])
    assert result["mcc"] == 1.0
    assert 0.2 < result["threshold"] <= 0.6


def test_load_disorder_calibration_validates_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "parameters": {"scale": 0.8, "bias": 1.2},
        "threshold": 0.9,
    }))
    assert load_disorder_calibration(path)["parameters"]["scale"] == 0.8
