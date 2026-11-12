#!/usr/bin/env python
"""AB-bind single-mutation, antibody-only replication check.

Uses the already-scored AB-bind table. Filters to:
  - one mutation per record
  - mutation chain H or L (antibody side)
Then reports within-complex Spearman, a deterministic PDB-hash half split,
and per-antigen-family consistency against the composition baseline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hf_aux_benchmark_v1/abbind"
SABDAB = ROOT / "data/sabdab_summary_all.tsv"


def parse_letters(mutation: str) -> set[str]:
    return {part.split(":")[0] for part in str(mutation).split(",") if ":" in part}


def bootstrap(values, seed=20261111, trials=10000):
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(trials, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def permutation_p(frame: pd.DataFrame, score: str, iterations=5000) -> float:
    rng = np.random.default_rng(20261111)
    complexes = [g for _, g in frame.groupby("pid")
                 if len(g) >= 5 and g["ddG"].nunique() > 1 and g[score].nunique() > 1]
    observed = np.mean([spearmanr(g[score], g["ddG"]).statistic for g in complexes])
    count = 0
    for _ in range(iterations):
        values = []
        for g in complexes:
            y = g["ddG"].to_numpy().copy()
            rng.shuffle(y)
            values.append(spearmanr(g[score], y).statistic)
        if np.mean(values) <= observed:
            count += 1
    return (count + 1) / (iterations + 1)


def summarize(frame: pd.DataFrame, label: str) -> dict:
    rows = []
    for pid, group in frame.groupby("pid"):
        if len(group) < 5 or group["ddG"].nunique() < 3 or group["dils"].nunique() < 2:
            continue
        rows.append({
            "pid": pid, "n": len(group),
            "rho_dils": spearmanr(group["dils"], group["ddG"]).statistic,
            "rho_complex": spearmanr(group["dlogp_complex"], group["ddG"]).statistic,
            "rho_composition": spearmanr(group["composition_pred"], group["ddG"]).statistic,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return {"label": label, "n_complexes": 0}
    result = {"label": label, "n_complexes": int(len(table)),
              "n_rows": int(len(frame))}
    for score in ("rho_dils", "rho_complex", "rho_composition"):
        values = table[score].dropna().to_numpy()
        result[score] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "fraction_negative": float(np.mean(values < 0)),
            "ci95": bootstrap(values),
        }
    result["permutation_p_dils"] = permutation_p(frame, "dils")
    return result, table


def main():
    scores = pd.read_csv(OUT / "scores.csv")
    scores = scores[scores["status"].eq("ok")].copy()
    scores["letters"] = scores["mutation"].map(parse_letters)
    single_ab = scores[
        scores["n_mut"].eq(1)
        & scores["letters"].map(lambda s: s <= {"H", "L"})
    ].copy()
    print("single antibody rows", len(single_ab), "complexes", single_ab["pid"].nunique())
    summary, table = summarize(single_ab, "single_antibody")
    table.to_csv(OUT / "single_antibody_per_complex.csv", index=False)

    # deterministic PDB-hash half split
    single_ab["half"] = single_ab["pid"].map(
        lambda pid: "A" if int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 2 == 0 else "B")
    halves = {}
    for half, group in single_ab.groupby("half"):
        halves[half] = summarize(group, f"half_{half}")[0]

    # antigen-family assignment from SAbDab
    sab = pd.read_csv(SABDAB, sep="\t", dtype=str)
    sab["pid"] = sab["pdb"].str.lower()
    sab = sab.drop_duplicates("pid")[["pid", "antigen_name"]]
    merged = single_ab.merge(sab, on="pid", how="left")
    families = {}
    for antigen, group in merged.groupby("antigen_name"):
        if len(group) < 5 or group["pid"].nunique() < 1:
            continue
        per = []
        for pid, comp in group.groupby("pid"):
            if len(comp) >= 5 and comp["ddG"].nunique() > 2 and comp["dils"].nunique() > 1:
                per.append(spearmanr(comp["dils"], comp["ddG"]).statistic)
        if per:
            families[str(antigen)] = {"n_complexes": len(per), "mean_rho": float(np.mean(per))}

    result = {"summary": summary, "halves": halves, "families": families}
    (OUT / "single_antibody_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
