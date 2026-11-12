#!/usr/bin/env python
"""Analyze the full SAbDab2-Epi H3-ECLS run.

Reads results/hf_aux_benchmark_v1/sabdab2_epi/h3ecls_full.csv and reports:
  - overall and per-split advantage
  - H3-length strata
  - cdrh3-cluster-level mean advantage and cluster bootstrap
  - ANARCI vs reference-aligned counts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hf_aux_benchmark_v1/sabdab2_epi"


def bootstrap(values, seed=20261112, trials=10000):
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(trials, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "fraction_positive": float((values > 0).mean()),
        "ci95": bootstrap(values),
    }


def main():
    frame = pd.read_csv(OUT / "h3ecls_full.csv")
    overall = describe(frame["advantage"])
    by_split = {
        split: describe(group["advantage"])
        for split, group in frame.groupby("split")
    }
    bins = pd.cut(frame["n_h3"], bins=[4, 9, 13, 18, 30],
                  labels=["5-9", "10-13", "14-18", "19-24"])
    by_length = {
        str(label): describe(group["advantage"])
        for label, group in frame.groupby(bins, observed=True)
    }
    # cluster-level: mean per cdrh3 cluster, then bootstrap clusters
    cluster_means = frame.groupby("cdrh3_cluster")["advantage"].mean()
    cluster_stats = describe(cluster_means.to_numpy())
    # unique PDB / cluster counts
    result = {
        "overall": overall,
        "by_split": by_split,
        "by_h3_length": by_length,
        "cluster_level": cluster_stats,
        "n_unique_pdb": int(frame["pdb_id"].nunique()),
        "n_unique_cdrh3": int(frame["cdrh3_cluster"].nunique()),
        "n_anarci": int((frame["h3_source"] == "anarci_imgt").sum()),
        "n_reference": int((frame["h3_source"] == "reference").sum()),
        "negative_rows": frame[frame["advantage"] < 0][
            ["pdb_id", "peptide_chain", "cdrh3_cluster", "n_h3", "advantage"]
        ].to_dict(orient="records"),
    }
    (OUT / "h3ecls_full_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# SAbDab2-Epi H3-ECLS full analysis", ""]
    lines.append(f"Rows: {overall['n']}; unique PDB: {result['n_unique_pdb']}; "
                 f"unique CDRH3 clusters: {result['n_unique_cdrh3']}")
    lines.append(f"ANARCI/IMGT: {result['n_anarci']}; reference: {result['n_reference']}")
    lines.append("")
    lines.append("| stratum | n | mean | median | fraction positive | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for label, item in [("overall", overall), *by_split.items(), *by_length.items()]:
        lines.append(f"| {label} | {item['n']} | {item['mean']:.3f} | "
                     f"{item['median']:.3f} | {item['fraction_positive']:.3f} | "
                     f"[{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] |")
    lines.append("")
    lines.append(f"Cluster-level: n={cluster_stats['n']}, mean={cluster_stats['mean']:.3f}, "
                 f"median={cluster_stats['median']:.3f}, "
                 f"fraction positive={cluster_stats['fraction_positive']:.3f}, "
                 f"CI95={cluster_stats['ci95']}")
    lines.append("")
    lines.append("Negative rows:")
    for row in result["negative_rows"]:
        lines.append(f"- {row['pdb_id']} {row['peptide_chain']} "
                     f"{row['cdrh3_cluster']} n_h3={row['n_h3']} adv={row['advantage']:.3f}")
    (OUT / "h3ecls_full_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
