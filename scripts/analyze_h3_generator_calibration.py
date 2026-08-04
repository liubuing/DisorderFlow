#!/usr/bin/env python
"""Calibrate and report generator-aware likelihood contrast reranking."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def candidate_nnr(row, coefficient):
    native = (
        row["native_scores"]["complex_nll"]
        - coefficient * row["native_scores"]["apo_nll"])
    candidates = [
        item["complex_nll"] - coefficient * item["apo_nll"]
        for item in row["candidates"]
    ]
    if not candidates:
        return 1.0
    better = sum(score < native for score in candidates)
    tied = sum(score == native for score in candidates)
    return 1.0 - (better + 0.5 * tied) / len(candidates)


def aggregate_units(rows, coefficient):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["unit"], []).append(candidate_nnr(row, coefficient))
    return {
        unit: float(np.mean(values))
        for unit, values in sorted(grouped.items())
    }


def bootstrap_ci(values, seed, trials):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(trials, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def exact_sign_permutation_p(values):
    """Two-sided paired sign-flip permutation test on inference-unit effects."""
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    exceedances = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(float(np.mean(values * np.asarray(signs))))
        exceedances += permuted >= observed - 1e-12
    return exceedances / total


def benjamini_hochberg(p_values):
    """Return FDR-adjusted P values in the input order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def choose_mpnn_coefficient(rows, settings):
    start = float(settings["lambda_grid_start"])
    stop = float(settings["lambda_grid_stop"])
    step = float(settings["lambda_grid_step"])
    grid = np.round(np.arange(start, stop + step / 2, step), 8)
    curve = []
    selected = None
    for coefficient in grid:
        units = aggregate_units(rows, float(coefficient))
        values = list(units.values())
        point = {
            "lambda": float(coefficient),
            "mean_nnr": float(np.mean(values)),
            "top1_fraction": float(np.mean(np.isclose(values, 1.0))),
        }
        curve.append(point)
        if (
            selected is None
            and point["mean_nnr"] >= float(settings["minimum_mean_nnr"])
            and point["top1_fraction"] >= float(settings["minimum_top1_fraction"])
        ):
            selected = float(coefficient)
    if selected is None:
        raise RuntimeError("No ProteinMPNN coefficient passed the frozen calibration rule")
    return selected, curve


def evaluate_development(rows, coefficients, seed, trials):
    generators = sorted({row["generator"] for row in rows})
    per_generator = {}
    unit_values = {}
    for generator in generators:
        generator_rows = [row for row in rows if row["generator"] == generator]
        coefficient = coefficients[generator]
        calibrated = aggregate_units(generator_rows, coefficient)
        ecls = aggregate_units(generator_rows, 1.0)
        complex_nll = aggregate_units(generator_rows, 0.0)
        values = list(calibrated.values())
        over_random = [value - 0.5 for value in values]
        per_generator[generator] = {
            "lambda": coefficient,
            "n_units": len(values),
            "mean_calibrated_nnr": float(np.mean(values)),
            "calibrated_over_random_ci95": bootstrap_ci(
                over_random, seed, trials),
            "calibrated_over_random_permutation_p": exact_sign_permutation_p(
                over_random),
            "mean_ecls_nnr": float(np.mean(list(ecls.values()))),
            "mean_complex_nll_nnr": float(np.mean(list(complex_nll.values()))),
        }
        for unit, value in calibrated.items():
            unit_values.setdefault(unit, {})[generator] = value
    generator_names = list(per_generator)
    adjusted = benjamini_hochberg([
        per_generator[name]["calibrated_over_random_permutation_p"]
        for name in generator_names
    ])
    for name, adjusted_p in zip(generator_names, adjusted):
        per_generator[name]["calibrated_over_random_bh_q"] = adjusted_p
    pooled = {
        unit: float(np.mean(list(values.values())))
        for unit, values in sorted(unit_values.items())
    }
    pooled_values = list(pooled.values())
    complex_pooled = {}
    for unit in pooled:
        values = []
        for generator in generators:
            generator_rows = [
                row for row in rows
                if row["generator"] == generator and row["unit"] == unit]
            values.append(float(np.mean([
                candidate_nnr(row, 0.0) for row in generator_rows])))
        complex_pooled[unit] = float(np.mean(values))
    gains = [pooled[unit] - complex_pooled[unit] for unit in pooled]
    over_random = [value - 0.5 for value in pooled_values]
    leave_one_out = [
        float(np.mean([value for other_index, value in enumerate(pooled_values)
                       if other_index != index]))
        for index in range(len(pooled_values))
    ]
    return {
        "generators": per_generator,
        "n_inference_units": len(pooled),
        "mean_generator_aware_nnr": float(np.mean(pooled_values)),
        "generator_aware_over_random_ci95": bootstrap_ci(
            over_random, seed + 1, trials),
        "generator_aware_over_random_permutation_p": exact_sign_permutation_p(
            over_random),
        "mean_gain_over_complex_nll": float(np.mean(gains)),
        "gain_over_complex_nll_ci95": bootstrap_ci(gains, seed + 2, trials),
        "gain_over_complex_nll_permutation_p": exact_sign_permutation_p(gains),
        "fraction_units_above_random": float(np.mean(np.asarray(pooled_values) > 0.5)),
        "leave_one_cluster_out_mean_nnr_range": [
            float(min(leave_one_out)), float(max(leave_one_out))],
        "units": [
            {
                "unit": unit,
                "generator_aware_nnr": pooled[unit],
                "complex_nll_nnr": complex_pooled[unit],
                "gain": pooled[unit] - complex_pooled[unit],
                **unit_values[unit],
            }
            for unit in pooled
        ],
    }


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main_table_rows(development):
    rows = []
    for generator, values in development["generators"].items():
        rows.append({
            "analysis": generator,
            "n_clusters": values["n_units"],
            "mean_nnr": round(values["mean_calibrated_nnr"], 6),
            "mean_complex_nll_nnr": round(values["mean_complex_nll_nnr"], 6),
            "effect_vs_random": round(values["mean_calibrated_nnr"] - 0.5, 6),
            "ci95_low": round(values["calibrated_over_random_ci95"][0], 6),
            "ci95_high": round(values["calibrated_over_random_ci95"][1], 6),
            "permutation_p": values["calibrated_over_random_permutation_p"],
            "bh_q": values["calibrated_over_random_bh_q"],
        })
    rows.append({
        "analysis": "pooled_generator_aware",
        "n_clusters": development["n_inference_units"],
        "mean_nnr": round(development["mean_generator_aware_nnr"], 6),
        "mean_complex_nll_nnr": "",
        "effect_vs_random": round(development["mean_generator_aware_nnr"] - 0.5, 6),
        "ci95_low": round(development["generator_aware_over_random_ci95"][0], 6),
        "ci95_high": round(development["generator_aware_over_random_ci95"][1], 6),
        "permutation_p": development["generator_aware_over_random_permutation_p"],
        "bh_q": "",
    })
    return rows


def plot_summary(path_base, curve, selected, development):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    axes[0].plot(
        [row["lambda"] for row in curve], [row["mean_nnr"] for row in curve],
        color="#2166AC", linewidth=2)
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].axvline(selected, color="#B2182B", linestyle=":", linewidth=1.5)
    axes[0].set(xlabel="Apo coefficient (lambda)", ylabel="Mean native NNR",
                title="A  Adaptation calibration", ylim=(-0.03, 1.03))

    generators = list(development["generators"])
    labels = {"bfn": "BFN", "esm_if": "ESM-IF", "proteinmpnn": "ProteinMPNN"}
    x = np.arange(len(generators))
    axes[1].bar(
        x - 0.2,
        [development["generators"][item]["mean_complex_nll_nnr"] for item in generators],
        width=0.4, label="Complex NLL", color="#BDBDBD")
    axes[1].bar(
        x + 0.2,
        [development["generators"][item]["mean_calibrated_nnr"] for item in generators],
        width=0.4, label="Generator-aware", color="#4D9221")
    axes[1].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, [labels[item] for item in generators], rotation=20)
    axes[1].set(ylabel="Mean native NNR", title="B  Development by generator",
                ylim=(0, 1.05))
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    units = development["units"]
    values = [row["generator_aware_nnr"] for row in units]
    axes[2].scatter(np.arange(len(units)), values, color="#762A83", s=28, zorder=3)
    axes[2].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[2].axhline(np.mean(values), color="#1B7837", linewidth=2)
    axes[2].set_xticks(np.arange(len(units)), [row["unit"] for row in units], rotation=45)
    axes[2].set(ylabel="Generator-aware NNR", title="C  Development clusters",
                ylim=(0, 1.05))
    figure.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "benchmarks" / "peptide_h3_generator_calibration_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "publication" / "h3_generator_calibration_v1"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    adaptation = json.loads(
        (ROOT / config["adaptation_results"]).read_text(encoding="utf-8"))
    development = json.loads(
        (ROOT / config["development_results"]).read_text(encoding="utf-8"))
    mpnn_rows = [
        row for row in adaptation["results"] if row["generator"] == "proteinmpnn"]
    coefficient, curve = choose_mpnn_coefficient(
        mpnn_rows, config["calibration"]["proteinmpnn"])
    coefficients = {
        "bfn": float(config["calibration"]["bfn"]["lambda"]),
        "esm_if": float(config["calibration"]["esm_if"]["lambda"]),
        "proteinmpnn": coefficient,
    }
    stats = config["statistics"]
    evaluation = evaluate_development(
        development["results"], coefficients,
        int(stats["bootstrap_seed"]), int(stats["bootstrap_trials"]))
    output = {
        "schema_version": 1,
        "status": "retrospective development analysis; temporal final not accessed",
        "score_formula": config["score"]["formula"],
        "coefficients": coefficients,
        "proteinmpnn_adaptation_curve": curve,
        "development": evaluation,
        "claim_boundary": config["claim_boundary"],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="ascii")
    write_csv(out_dir / "per_unit.csv", evaluation["units"])
    write_csv(out_dir / "proteinmpnn_calibration_curve.csv", curve)
    write_csv(out_dir / "main_table.csv", main_table_rows(evaluation))
    plot_summary(out_dir / "generator_calibration", curve, coefficient, evaluation)
    print(json.dumps({
        "coefficients": coefficients,
        "development": {key: value for key, value in evaluation.items()
                        if key not in {"generators", "units"}},
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
