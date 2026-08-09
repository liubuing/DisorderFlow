#!/usr/bin/env python
"""Post hoc cluster-level sign-flip sensitivity analysis for saved ECLS results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sign_flip_p(values, seed=2621, trials=1_000_000):
    """Two-sided sign-flip P value, exact up to 20 units and Monte Carlo above."""
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    if len(values) <= 20:
        indices = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        means = np.abs((signs * values).mean(axis=1))
        return {
            "method": "exact_two_sided_sign_flip",
            "p": float(np.mean(means >= observed - 1e-12)),
            "permutations": int(len(means)),
            "seed": None,
        }
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    chunk_size = 10_000
    while completed < trials:
        count = min(chunk_size, trials - completed)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        means = np.abs((signs * values).mean(axis=1))
        exceedances += int(np.sum(means >= observed - 1e-12))
        completed += count
    return {
        "method": "monte_carlo_two_sided_sign_flip",
        "p": float((exceedances + 1) / (trials + 1)),
        "permutations": int(trials),
        "seed": int(seed),
    }


def leave_one_out_range(values):
    values = list(map(float, values))
    means = [
        float(np.mean(values[:index] + values[index + 1:]))
        for index in range(len(values))
    ]
    return [min(means), max(means)]


def benjamini_hochberg(p_values):
    """Return FDR-adjusted q values in the original order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, float(p_values[original_index]) * count / rank)
        adjusted[original_index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/publication/h3_ecls_statistical_summary_v1"))
    args = parser.parse_args()
    adaptation = json.loads((
        ROOT / "results/publication/h3_ecls_adaptation_v1/results.json"
    ).read_text(encoding="utf-8"))
    temporal = json.loads((
        ROOT / "results/publication/h3_ecls_temporal_final_v1/results.json"
    ).read_text(encoding="utf-8"))
    adaptation_values = [
        float(row["summary"]["ecls_advantage"]) for row in adaptation["results"]]
    temporal_values = [
        float(row["mean_ecls_advantage"]) for row in temporal["inference_units"]]
    adaptation_test = sign_flip_p(adaptation_values)
    temporal_test = sign_flip_p(temporal_values)
    q_values = benjamini_hochberg([adaptation_test["p"], temporal_test["p"]])
    adaptation_test["bh_q_across_reported_ecls_tests"] = q_values[0]
    temporal_test["bh_q_across_reported_ecls_tests"] = q_values[1]
    output = {
        "schema_version": 1,
        "status": "post_hoc_sensitivity_from_saved_results; not_a_final_rerun",
        "decision_rule_note": (
            "Frozen ECLS decisions used bootstrap confidence intervals and preregistered gates; "
            "these P values are supplementary and were computed after evaluation."),
        "inference_note": (
            "No per-cluster hypothesis tests were performed. Each antigen cluster is one "
            "inference unit; bootstrap and sign-flip procedures operate on cluster summaries."),
        "adaptation": {
            "n_inference_units": len(adaptation_values),
            "mean": float(np.mean(adaptation_values)),
            "sign_flip": adaptation_test,
            "leave_one_cluster_out_mean_range": leave_one_out_range(adaptation_values),
        },
        "temporal_final": {
            "n_inference_units": len(temporal_values),
            "mean": float(np.mean(temporal_values)),
            "sign_flip": temporal_test,
            "leave_one_cluster_out_mean_range": leave_one_out_range(temporal_values),
            "model_forward_performed": False,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
