#!/usr/bin/env python
"""Create tables and figures for the frozen T2 recovery benchmark."""

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
    permutations = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(permutations) >= observed - 1e-12))


def analyze(raw):
    rows = []
    for record in raw["results"]:
        replicas = record["t2_audit"]["replicas"]
        tier_failures = sum(not row["initial_in_t2_tier"] for row in replicas)
        numerical_failures = sum(bool(row.get("error")) for row in replicas)
        row = {
            "id": record["id"], "unit": record["unit"],
            "valid_t2": record["valid_t2"],
            "accepted_replicas": record["t2_audit"]["accepted"],
            "tier_failures": tier_failures,
            "numerical_failures": numerical_failures,
            "mean_held_out_contact_recovery": "",
            "mean_rmsd_recovery_angstrom": "",
            "final_ecls_advantage": "",
            "ecls_advantage_recovery": "",
        }
        if record["valid_t2"]:
            row.update(record["metrics"])
        rows.append(row)
    valid = [row for row in rows if row["valid_t2"]]
    contact = [row["mean_held_out_contact_recovery"] for row in valid]
    rmsd = [row["mean_rmsd_recovery_angstrom"] for row in valid]
    summary = {
        **raw["aggregate"],
        "held_out_contact_recovery_exact_sign_flip_p": exact_sign_flip_p(contact),
        "rmsd_recovery_exact_sign_flip_p": exact_sign_flip_p(rmsd),
        "total_tier_failures": sum(row["tier_failures"] for row in rows),
        "total_numerical_failures": sum(row["numerical_failures"] for row in rows),
        "interpretation": (
            "Broad supplied-contact restraints did not recover held-out contacts or "
            "native RMSD across clusters; T2 recovery claim rejected."),
    }
    return rows, summary


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(path, rows):
    labels = [row["id"].split("_")[1] for row in rows]
    accepted = [row["accepted_replicas"] for row in rows]
    valid = [row for row in rows if row["valid_t2"]]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    colors = ["#4D9221" if row["valid_t2"] else "#B2182B" for row in rows]
    axes[0].bar(np.arange(len(rows)), accepted, color=colors)
    axes[0].axhline(3, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(rows)), labels, rotation=45)
    axes[0].set(ylabel="Accepted replicas", title="A  T2 validity", ylim=(0, 4.5))

    x = np.arange(len(valid))
    axes[1].bar(
        x - 0.18, [row["mean_held_out_contact_recovery"] for row in valid],
        width=0.36, color="#2166AC", label="Held-out contacts")
    axes[1].bar(
        x + 0.18, [row["mean_rmsd_recovery_angstrom"] for row in valid],
        width=0.36, color="#B2182B", label="RMSD recovery (A)")
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, [row["unit"] for row in valid], rotation=45)
    axes[1].set(title="B  Recovery after broad restraints")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].scatter(
        [row["mean_held_out_contact_recovery"] for row in valid],
        [row["final_ecls_advantage"] for row in valid],
        color="#762A83", s=55)
    for row in valid:
        axes[2].annotate(
            row["unit"],
            (row["mean_held_out_contact_recovery"], row["final_ecls_advantage"]),
            xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[2].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[2].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[2].set(xlabel="Held-out contact recovery",
                ylabel="Final ensemble ECLS advantage",
                title="C  Structural versus ECLS recovery")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=str(ROOT / "results/publication/h3_t2_recovery_dev_v1/results.json"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/publication/h3_t2_recovery_analysis_v1"))
    args = parser.parse_args()
    raw = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows, summary = analyze(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "development_only; temporal final not accessed",
        "summary": summary, "per_cluster": rows,
    }, indent=2) + "\n", encoding="ascii")
    write_csv(out_dir / "main_table.csv", rows)
    plot(out_dir / "t2_recovery_summary", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
