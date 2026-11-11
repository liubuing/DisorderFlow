#!/usr/bin/env python3
"""Task-specific pairwise ranker on AAYL: ECLS vs composition vs both.

For each family, build all within-family pairs, train a logistic pairwise
ranker on 5-fold CV, and compare held-out Spearman with the endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "ecls_cross_arch_v1" / "aayl_m3" / "scores.json"
OUT = ROOT / "results" / "ecls_factorial_v1" / "task_ranker.json"
AA = "ACDEFGHIKLMNPQRSTVWY"


def composition(sequence: str) -> np.ndarray:
    return np.array([sequence.count(aa) / len(sequence) for aa in AA], dtype=float)


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_vals = np.asarray(values, dtype=float)[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def fit_score(features, endpoint, train_idx, test_idx):
    train_features = features[train_idx]
    train_endpoint = endpoint[train_idx]
    pairs, labels = [], []
    for i in range(len(train_idx)):
        for j in range(i + 1, len(train_idx)):
            if train_endpoint[i] == train_endpoint[j]:
                continue
            diff = train_features[i] - train_features[j]
            pairs.append(diff)
            labels.append(1.0 if train_endpoint[i] > train_endpoint[j] else 0.0)
            pairs.append(-diff)
            labels.append(0.0 if train_endpoint[i] > train_endpoint[j] else 1.0)
    if not pairs:
        return np.zeros(len(test_idx))
    model = LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0)
    model.fit(np.asarray(pairs), np.asarray(labels))
    return features[test_idx] @ model.coef_[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    records = json.loads(args.scores.read_text(encoding="utf-8"))["records"]
    families = sorted({record["family"] for record in records})
    rng = np.random.default_rng(20261111)
    report = {}
    for family in families:
        subset = [record for record in records if record["family"] == family]
        endpoint = np.array([record["endpoint"] for record in subset])
        ecls = np.array([record["ecls"] for record in subset])
        comp = np.array([composition(record["h3_sequence"]) for record in subset])
        indices = np.arange(len(subset))
        rng.shuffle(indices)
        folds = np.array_split(indices, 5)
        results = {"ecls": [], "composition": [], "ecls_plus_composition": []}
        for fold in folds:
            test_idx = fold
            train_idx = np.setdiff1d(indices, fold)
            score_ecls = fit_score(ecls[:, None], endpoint, train_idx, test_idx)
            score_comp = fit_score(comp, endpoint, train_idx, test_idx)
            score_both = fit_score(
                np.column_stack([ecls, comp]), endpoint, train_idx, test_idx)
            results["ecls"].append(spearman(score_ecls, endpoint[test_idx]))
            results["composition"].append(spearman(score_comp, endpoint[test_idx]))
            results["ecls_plus_composition"].append(
                spearman(score_both, endpoint[test_idx]))
        report[family] = {
            "n": len(subset),
            "folds": {key: [round(value, 4) for value in values]
                      for key, values in results.items()},
            "mean": {key: round(float(np.mean(values)), 4)
                     for key, values in results.items()},
            "sd": {key: round(float(np.std(values)), 4)
                   for key, values in results.items()},
        }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
