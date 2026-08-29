#!/usr/bin/env python
"""Evaluate a jackknife lower-envelope ECLS selector on exposed components."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def robust_value(values, aggregation):
    values = np.asarray(values, dtype=np.float64)
    return float(
        float(aggregation["p25_weight"]) * np.percentile(values, 25)
        + float(aggregation["minimum_weight"]) * values.min()
        + float(aggregation["mean_weight"]) * values.mean()
        - float(aggregation["standard_deviation_penalty"]) * values.std()
    )


def jackknife_lower_envelope(matrix, training, sequences, aggregation):
    if len(training) < 2:
        raise ValueError("Jackknife lower envelope requires at least two training conformers")
    scores = []
    inner_scores = []
    for candidate in range(len(sequences)):
        candidate_inner = [
            robust_value(
                matrix[candidate, [index for index in training if index != deleted]],
                aggregation,
            )
            for deleted in training
        ]
        inner_scores.append(candidate_inner)
        scores.append(min(candidate_inner))
    winner = max(range(len(sequences)), key=lambda index: (scores[index], sequences[index]))
    return winner, scores, inner_scores


def bootstrap_mean(values, trials, seed):
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = generator.choice(array, size=(trials, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def exact_sign_flip_p(values):
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    estimates = [
        abs(float(np.mean(array * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ]
    return float(np.mean(np.asarray(estimates) >= observed - 1e-12))


def analyze_component(component, method_row, config):
    sequences = method_row["sequences"]
    matrix = np.asarray(method_row["epitope_conditioning_gain"], dtype=np.float64)
    aggregation = config["selector"]["robust_aggregation"]
    range_threshold = float(
        config["report_only_abstention"]["minimum_median_training_candidate_range"]
    )
    folds = []
    for holdout in range(matrix.shape[1]):
        training = [index for index in range(matrix.shape[1]) if index != holdout]
        winner, lower_scores, inner_scores = jackknife_lower_envelope(
            matrix, training, sequences, aggregation
        )
        single_winners = [int(np.argmax(matrix[:, index])) for index in training]
        single_score = float(np.median([
            matrix[index, holdout] for index in single_winners
        ]))
        ensemble_score = float(matrix[winner, holdout])
        training_ranges = [float(np.ptp(matrix[:, index])) for index in training]
        identity = all(index == winner for index in single_winners)
        low_signal = float(np.median(training_ranges)) < range_threshold
        sorted_scores = sorted(lower_scores, reverse=True)
        folds.append({
            "heldout_conformer_index": holdout,
            "winner": sequences[winner],
            "winner_index": winner,
            "winner_lower_envelope_score": lower_scores[winner],
            "winner_inner_scores": inner_scores[winner],
            "selection_margin": (
                lower_scores[winner] - sorted_scores[1]
                if len(sorted_scores) > 1 else None
            ),
            "single_state_winners": [sequences[index] for index in single_winners],
            "selector_identity_abstention": identity,
            "low_training_signal_abstention": low_signal,
            "report_only_eligible": not identity and not low_signal,
            "ensemble_score": ensemble_score,
            "single_state_score": single_score,
            "ensemble_minus_single_state": ensemble_score - single_score,
        })
    effects = [row["ensemble_minus_single_state"] for row in folds]
    eligible = [
        row["ensemble_minus_single_state"] for row in folds
        if row["report_only_eligible"]
    ]
    return {
        "component_id": component["component_id"],
        "method": method_row["method"],
        "mean_ensemble_minus_single_state": float(np.mean(effects)),
        "strictly_positive_folds": sum(value > 0 for value in effects),
        "negative_folds": sum(value < 0 for value in effects),
        "unique_winners": len({row["winner"] for row in folds}),
        "eligible_folds": len(eligible),
        "eligible_mean_ensemble_minus_single_state": (
            float(np.mean(eligible)) if eligible else None
        ),
        "folds": folds,
    }


def summarize(rows, config, seed_offset):
    values = [row["mean_ensemble_minus_single_state"] for row in rows]
    epsilon = float(config["statistics"]["mathematical_effect_epsilon"])
    positive = sum(value > epsilon for value in values)
    negative = sum(value < -epsilon for value in values)
    ci = bootstrap_mean(
        values,
        int(config["statistics"]["bootstrap_trials"]),
        int(config["statistics"]["bootstrap_seed"]) + seed_offset,
    )
    gates = config["development_gates_for_future_freeze"]
    gate_results = {
        "minimum_valid_components": len(values) >= int(gates["minimum_valid_components"]),
        "minimum_strictly_positive_components": positive
        >= int(gates["minimum_strictly_positive_components"]),
        "bootstrap_ci95_lower_above_zero": bool(ci and ci[0] > 0),
        "maximum_negative_components": negative <= int(gates["maximum_negative_components"]),
    }
    eligible_effects = [
        fold["ensemble_minus_single_state"]
        for row in rows for fold in row["folds"] if fold["report_only_eligible"]
    ]
    return {
        "valid_components": len(values),
        "mean_ensemble_minus_single_state": float(np.mean(values)),
        "median_ensemble_minus_single_state": float(np.median(values)),
        "strictly_positive_components": positive,
        "zero_components": len(values) - positive - negative,
        "negative_components": negative,
        "component_bootstrap_ci95": ci,
        "exact_sign_flip_p": exact_sign_flip_p(values),
        "full_coverage": 1.0,
        "report_only_abstention": {
            "eligible_folds": len(eligible_effects),
            "total_folds": sum(len(row["folds"]) for row in rows),
            "coverage": len(eligible_effects) / sum(len(row["folds"]) for row in rows),
            "mean_on_eligible_folds": (
                float(np.mean(eligible_effects)) if eligible_effects else None
            ),
        },
        "gate_results": gate_results,
        "passed_for_future_freeze": all(gate_results.values()),
        "components": rows,
    }


def write_report(path, output):
    primary = output["method_summary"][output["primary_method"]]
    lines = [
        "# ECLS v2 Jackknife Development", "",
        "This evaluates an exposed-panel development mechanism and is not confirmation.", "",
        "## Primary Result", "",
        f"- Mean ensemble-minus-single: {primary['mean_ensemble_minus_single_state']:.8f}",
        f"- Strictly positive components: {primary['strictly_positive_components']}/12",
        f"- Negative components: {primary['negative_components']}",
        f"- Component bootstrap CI95: {primary['component_bootstrap_ci95']}",
        f"- Future-freeze gate: {primary['passed_for_future_freeze']}", "",
        "## Components", "",
        "| Component | Mean delta | Positive folds | Negative folds | Eligible folds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary["components"]:
        lines.append(
            f"| {row['component_id']} | {row['mean_ensemble_minus_single_state']:.8f} | "
            f"{row['strictly_positive_folds']} | {row['negative_folds']} | "
            f"{row['eligible_folds']}/5 |"
        )
    lines.extend(["", f"Claim boundary: {output['claim_boundary']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_ecls_v2_development_v1.yml"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    analyses = []
    provenance = {}
    for name, item in config["inputs"].items():
        path = ROOT / item["path"]
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {path}")
        analyses.append(json.loads(path.read_text(encoding="utf-8")))
        provenance[name] = {"path": item["path"], "sha256": digest}

    by_method = {method: [] for method in config["methods"]}
    for analysis in analyses:
        for component in analysis["component_scores"]:
            for method_row in component["methods"]:
                if method_row["method"] in by_method:
                    by_method[method_row["method"]].append(
                        analyze_component(component, method_row, config)
                    )
    method_summary = {
        method: summarize(rows, config, index)
        for index, (method, rows) in enumerate(by_method.items())
    }
    primary = config["primary_method"]
    output = {
        "schema_version": 1,
        "status": "candidate_for_untouched_freeze"
        if method_summary[primary]["passed_for_future_freeze"]
        else "v2_development_gate_failed",
        "classification": config["classification"],
        "primary_method": primary,
        "selector": config["selector"],
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "inputs": provenance,
        },
        "method_summary": method_summary,
        "claim_boundary": config["claim_boundary"],
    }
    json_path = ROOT / config["output"]["json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    write_report(ROOT / config["output"]["report"], output)
    print(json.dumps({
        "status": output["status"],
        "primary": {
            key: method_summary[primary][key]
            for key in (
                "mean_ensemble_minus_single_state",
                "strictly_positive_components",
                "negative_components",
                "component_bootstrap_ci95",
                "report_only_abstention",
                "passed_for_future_freeze",
            )
        },
    }, indent=2))


if __name__ == "__main__":
    main()
