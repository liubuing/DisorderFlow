"""Dependency-free pooled metrics for residue-level disorder prediction."""

from __future__ import annotations

import math


def auc_roc(scores, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc_pr(scores, labels):
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    tp = 0
    ap = 0.0
    previous_recall = 0.0
    for index, (_, label) in enumerate(
            sorted(zip(scores, labels, strict=True), key=lambda item: -item[0]), 1):
        if label == 1:
            tp += 1
        recall = tp / n_pos
        ap += (tp / index) * (recall - previous_recall)
        previous_recall = recall
    return ap


def pooled_disorder_metrics(scores, labels, threshold=0.5):
    if len(scores) != len(labels):
        raise ValueError(f"Score/label length mismatch: {len(scores)} != {len(labels)}")
    if not scores:
        raise ValueError("At least one disorder score is required")
    if any(not math.isfinite(float(score)) or not 0 <= float(score) <= 1 for score in scores):
        raise ValueError("Disorder scores must be finite probabilities in [0, 1]")
    binary_labels = [1 if float(label) >= 0.5 else 0 for label in labels]
    predictions = [1 if float(score) >= threshold else 0 for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, binary_labels, strict=True))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, binary_labels, strict=True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, binary_labels, strict=True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, binary_labels, strict=True))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "disorder_auc_roc": auc_roc(scores, binary_labels),
        "disorder_auc_pr": auc_pr(scores, binary_labels),
        "disorder_mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "disorder_positive_rate": sum(predictions) / len(predictions),
        "disorder_n_residues": len(scores),
    }
