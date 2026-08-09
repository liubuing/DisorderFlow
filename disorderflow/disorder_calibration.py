"""Post-hoc calibration and operating-point selection for disorder logits."""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F


def apply_platt(logits, scale, bias):
    values = torch.as_tensor(logits, dtype=torch.float64)
    return torch.sigmoid(values * float(scale) + float(bias))


def load_disorder_calibration(path):
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported disorder calibration schema")
    parameters = artifact.get("parameters", {})
    if "scale" not in parameters or "bias" not in parameters or "threshold" not in artifact:
        raise ValueError("Incomplete disorder calibration artifact")
    if float(parameters["scale"]) <= 0:
        raise ValueError("Disorder calibration scale must be positive")
    return artifact


def fit_platt(logits, labels, weights=None, max_iter=100):
    values = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(labels, dtype=torch.float64)
    evidence = (torch.ones_like(targets) if weights is None
                else torch.as_tensor(weights, dtype=torch.float64))
    if values.numel() == 0 or values.shape != targets.shape or evidence.shape != targets.shape:
        raise ValueError("Logits, labels, and weights must be non-empty and have equal shapes")
    if torch.any(evidence < 0) or evidence.sum() <= 0:
        raise ValueError("Calibration weights must be non-negative with a positive sum")
    if not torch.any(targets == 0) or not torch.any(targets == 1):
        raise ValueError("Platt calibration requires both classes")

    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_scale, bias], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        scale = torch.exp(log_scale.clamp(-5.0, 5.0))
        losses = F.binary_cross_entropy_with_logits(
            values * scale + bias, targets, reduction="none")
        loss = (losses * evidence).sum() / evidence.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "scale": float(torch.exp(log_scale.clamp(-5.0, 5.0)).detach()),
        "bias": float(bias.detach()),
    }


def calibration_metrics(probabilities, labels, weights=None, bins=10):
    scores = [float(value) for value in probabilities]
    targets = [int(float(value) >= 0.5) for value in labels]
    evidence = ([1.0] * len(scores) if weights is None
                else [float(value) for value in weights])
    if not scores or not (len(scores) == len(targets) == len(evidence)):
        raise ValueError("Probabilities, labels, and weights must be non-empty and aligned")
    total_weight = sum(evidence)
    brier = sum(w * (p - y) ** 2 for p, y, w in zip(
        scores, targets, evidence, strict=True)) / total_weight
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(scores)
                   if low <= score < high or (index == bins - 1 and score == 1.0)]
        member_weight = sum(evidence[i] for i in members)
        if member_weight:
            confidence = sum(evidence[i] * scores[i] for i in members) / member_weight
            frequency = sum(evidence[i] * targets[i] for i in members) / member_weight
            ece += member_weight / total_weight * abs(confidence - frequency)
    return {"brier": brier, "ece": ece}


def mcc_at_threshold(probabilities, labels, threshold):
    predictions = [float(value) >= threshold for value in probabilities]
    targets = [float(value) >= 0.5 for value in labels]
    tp = sum(p and y for p, y in zip(predictions, targets, strict=True))
    tn = sum(not p and not y for p, y in zip(predictions, targets, strict=True))
    fp = sum(p and not y for p, y in zip(predictions, targets, strict=True))
    fn = sum(not p and y for p, y in zip(predictions, targets, strict=True))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "threshold": float(threshold),
        "positive_rate": (tp + fp) / len(targets),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_mcc_threshold(probabilities, labels):
    scores = [float(value) for value in probabilities]
    if not scores or len(scores) != len(labels):
        raise ValueError("Probabilities and labels must be non-empty and aligned")
    candidates = {0.0, 1.0}
    unique_scores = sorted(set(scores))
    candidates.update(unique_scores)
    candidates.update((left + right) / 2 for left, right in zip(
        unique_scores, unique_scores[1:], strict=False))
    results = [mcc_at_threshold(scores, labels, threshold) for threshold in candidates]
    return max(results, key=lambda result: (result["mcc"], -abs(result["threshold"] - 0.5)))
