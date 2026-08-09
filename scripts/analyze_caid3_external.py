#!/usr/bin/env python
"""Analyze frozen CAID3 results with cluster bootstrap and metapredict baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_caid import evaluate, load_caid_reference  # noqa: E402


def bootstrap_mean(values, trials=10000, seed=6301):
    rng = random.Random(seed)
    estimates = []
    for _ in range(trials):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def cluster_values(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            grouped[row["cluster_id"]].append(float(value))
    return {cluster: float(np.mean(values)) for cluster, values in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bfn", default=str(ROOT / "results/ablation/caid3_disorder_nox_balanced_v4_s2032.json")
    )
    parser.add_argument("--caid-dir", default=str(ROOT / "data/caid3/disorder_nox"))
    parser.add_argument(
        "--cluster-manifest", default=str(ROOT / "data/caid3/disorder_nox_clusters.json")
    )
    parser.add_argument(
        "--out", default=str(ROOT / "results/ablation/caid3_external_analysis_v1.json")
    )
    args = parser.parse_args()
    try:
        import metapredict
    except ImportError as error:
        raise RuntimeError("metapredict is required for the frozen external baseline") from error

    bfn = json.loads(Path(args.bfn).read_text())
    bfn_by_id = {row["id"]: row for row in bfn["per_target"]}
    targets = {
        target["id"]: target
        for target in load_caid_reference(args.caid_dir, args.cluster_manifest)
        if target["id"] in bfn_by_id
    }
    baseline_rows = []
    for target_id, target in targets.items():
        scores = metapredict.predict_disorder(target["sequence"]).astype(float).tolist()
        metrics = evaluate(scores, target["labels"], prediction_threshold=0.5)
        baseline_rows.append(
            {
                "id": target_id,
                "cluster_id": target["cluster_id"],
                **{
                    key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                    for key, value in metrics.items()
                },
            }
        )

    bfn_clusters = cluster_values(bfn["per_target"], "auc_roc")
    baseline_clusters = cluster_values(baseline_rows, "auc_roc")
    paired_clusters = sorted(set(bfn_clusters) & set(baseline_clusters))
    bfn_values = [bfn_clusters[cluster] for cluster in sorted(bfn_clusters)]
    baseline_values = [baseline_clusters[cluster] for cluster in sorted(baseline_clusters)]
    differences = [
        bfn_clusters[cluster] - baseline_clusters[cluster] for cluster in paired_clusters
    ]
    bfn_ci = bootstrap_mean(bfn_values)
    difference_ci = bootstrap_mean(differences) if differences else [None, None]
    output = {
        "schema_version": 1,
        "status": "external_discrimination_pass_calibration_fail",
        "bfn": {
            "n_scored_targets": bfn["n_targets"],
            "n_mixed_label_clusters": len(bfn_values),
            "macro_auc_roc": float(np.mean(bfn_values)),
            "cluster_bootstrap_95_ci": bfn_ci,
            "external_discrimination_gate": len(bfn_values) >= 30 and bfn_ci[0] > 0.5,
            "mean_brier": bfn["mean_brier"],
            "mean_ece": bfn["mean_ece"],
            "calibration_gate": bfn["mean_brier"] <= 0.25 and bfn["mean_ece"] <= 0.1,
        },
        "metapredict": {
            "version": getattr(metapredict, "__version__", "unknown"),
            "n_mixed_label_clusters": len(baseline_values),
            "macro_auc_roc": float(np.mean(baseline_values)),
        },
        "paired_bfn_minus_metapredict": {
            "n_clusters": len(differences),
            "mean_auc_roc_difference": float(np.mean(differences)),
            "cluster_bootstrap_95_ci": difference_ci,
        },
        "constant_prevalence_auc_roc": 0.5,
        "bootstrap": {"trials": 10000, "seed": 6301, "unit": "UniRef50 cluster"},
        "metapredict_per_target": baseline_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "metapredict_per_target"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
