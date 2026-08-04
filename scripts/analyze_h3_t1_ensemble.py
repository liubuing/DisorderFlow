#!/usr/bin/env python
"""Summarize the frozen T1 peptide-ensemble development benchmark."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def exact_sign_flip_p(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def conformer_advantages(record):
    native = record["native"]["per_conformer_ecls"]
    controls = [row["per_conformer_ecls"] for row in record["composition_shuffles"]]
    return [
        float(np.mean([row[index] for row in controls]) - native[index])
        for index in range(len(native))
    ]


def analyze(results):
    valid = [record for record in results["results"] if record["valid_ensemble"]]
    rows = []
    for record in results["results"]:
        if not record["valid_ensemble"]:
            failed_checks = sorted({
                check for conformer in record["ensemble_audit"]["conformers"]
                for check, passed in conformer["checks"].items() if not passed
            })
            rows.append({
                "id": record["id"], "unit": "", "valid_ensemble": False,
                "accepted_conformers": record["ensemble_audit"]["accepted"],
                "mean_pairwise_rmsd": "", "mean_contact_retention": "",
                "ensemble_advantage": "", "single_advantage": "", "gain": "",
                "conformer_advantage_std": "", "failed_checks": ";".join(failed_checks),
            })
            continue
        accepted = [
            conformer for conformer in record["ensemble_audit"]["conformers"]
            if conformer["status"] == "accepted"]
        advantages = conformer_advantages(record)
        rows.append({
            "id": record["id"], "unit": record["unit"], "valid_ensemble": True,
            "accepted_conformers": len(accepted),
            "mean_pairwise_rmsd": record["ensemble_audit"]["dispersion"][
                "mean_pairwise_rmsd"],
            "mean_contact_retention": float(np.mean([
                conformer["metrics"]["native_contact_retention"]
                for conformer in accepted])),
            "ensemble_advantage": record["metrics"]["ensemble_mean_ecls_advantage"],
            "single_advantage": record["metrics"]["deposited_single_ecls_advantage"],
            "gain": record["metrics"]["ensemble_minus_single_advantage"],
            "conformer_advantage_std": float(np.std(advantages)),
            "failed_checks": "",
        })
    advantages = [float(row["ensemble_advantage"]) for row in rows
                  if row["valid_ensemble"]]
    gains = [float(row["gain"]) for row in rows if row["valid_ensemble"]]
    return rows, {
        **results["aggregate"],
        "ensemble_advantage_exact_sign_flip_p": exact_sign_flip_p(advantages),
        "ensemble_minus_single_exact_sign_flip_p": exact_sign_flip_p(gains),
        "median_ensemble_ecls_advantage": float(np.median(advantages)),
        "median_ensemble_minus_single_advantage": float(np.median(gains)),
        "interpretation": (
            "T1 local ensembles retain positive mean native-vs-shuffle ECLS but fail "
            "the frozen positive-cluster gate and underperform deposited single poses."),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, rows):
    valid = [row for row in rows if row["valid_ensemble"]]
    labels = [row["unit"] for row in valid]
    ensemble = np.asarray([row["ensemble_advantage"] for row in valid], dtype=float)
    single = np.asarray([row["single_advantage"] for row in valid], dtype=float)
    rmsd = np.asarray([row["mean_pairwise_rmsd"] for row in valid], dtype=float)
    retention = np.asarray([row["mean_contact_retention"] for row in valid], dtype=float)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    x = np.arange(len(valid))
    for index in range(len(valid)):
        axes[0].plot([x[index], x[index]], [single[index], ensemble[index]],
                     color="#999999", linewidth=1)
    axes[0].scatter(x, single, label="Deposited pose", color="#2166AC", s=32)
    axes[0].scatter(x, ensemble, label="T1 ensemble mean", color="#B2182B", s=32)
    axes[0].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set(ylabel="Native-vs-shuffle ECLS advantage",
                title="A  Single pose versus ensemble")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(rmsd, ensemble, c=retention, cmap="viridis", s=55,
                    edgecolors="black", linewidths=0.4)
    for label, x_value, y_value in zip(labels, rmsd, ensemble):
        axes[1].annotate(label, (x_value, y_value), xytext=(3, 3),
                         textcoords="offset points", fontsize=7)
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Mean pairwise peptide backbone RMSD (A)",
                ylabel="Ensemble ECLS advantage",
                title="B  Pose-neighborhood dispersion")

    accepted = [row["accepted_conformers"] for row in rows]
    colors = ["#4D9221" if row["valid_ensemble"] else "#B2182B" for row in rows]
    axes[2].bar(np.arange(len(rows)), accepted, color=colors)
    axes[2].axhline(3, color="#555555", linestyle="--", linewidth=1)
    axes[2].set_xticks(np.arange(len(rows)), [row["id"].split("_")[1] for row in rows],
                       rotation=45)
    axes[2].set(ylabel="Accepted conformers", title="C  Ensemble quality control",
                ylim=(0, 4.5))
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=str(ROOT / "results/publication/h3_t1_ensemble_dev_v3/results.json"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/publication/h3_t1_ensemble_analysis_v1"))
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows, summary = analyze(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "development_only; temporal final not accessed",
        "summary": summary,
        "per_cluster": rows,
    }, indent=2) + "\n", encoding="ascii")
    write_csv(out_dir / "main_table.csv", rows)
    plot_summary(out_dir / "t1_ensemble_summary", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
