#!/usr/bin/env python
"""Measure jackknife-consensus abstention coverage after ECLS v2 failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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


def robust_winner(matrix, conformers, sequences, aggregation):
    scores = [robust_value(matrix[index, conformers], aggregation) for index in range(len(sequences))]
    winner = max(range(len(sequences)), key=lambda index: (scores[index], sequences[index]))
    alternatives = [score for index, score in enumerate(scores) if index != winner]
    margin = scores[winner] - max(alternatives) if alternatives else None
    return winner, margin


def consensus_selection(matrix, training, sequences, aggregation):
    inner = []
    for deleted in training:
        conformers = [index for index in training if index != deleted]
        winner, margin = robust_winner(matrix, conformers, sequences, aggregation)
        inner.append({
            "deleted_training_conformer": deleted,
            "winner_index": winner,
            "winner": sequences[winner],
            "strict_margin": margin,
        })
    winners = {row["winner_index"] for row in inner}
    positive_margins = all(
        row["strict_margin"] is None or row["strict_margin"] > 0 for row in inner
    )
    consensus = len(winners) == 1 and positive_margins
    return (next(iter(winners)) if consensus else None), inner


def analyze_component(component, method_row, config):
    sequences = method_row["sequences"]
    matrix = np.asarray(method_row["epitope_conditioning_gain"], dtype=np.float64)
    aggregation = config["selector"]["robust_aggregation"]
    range_threshold = float(
        config["secondary_abstention"]["minimum_median_training_candidate_range"]
    )
    folds = []
    for holdout in range(matrix.shape[1]):
        training = [index for index in range(matrix.shape[1]) if index != holdout]
        consensus_winner, inner = consensus_selection(
            matrix, training, sequences, aggregation
        )
        single_winners = [int(np.argmax(matrix[:, index])) for index in training]
        identity = consensus_winner is not None and all(
            index == consensus_winner for index in single_winners
        )
        training_ranges = [float(np.ptp(matrix[:, index])) for index in training]
        low_signal = float(np.median(training_ranges)) < range_threshold
        consensus_effect = None
        if consensus_winner is not None:
            single_score = float(np.median([
                matrix[index, holdout] for index in single_winners
            ]))
            consensus_effect = float(matrix[consensus_winner, holdout]) - single_score
        strict_eligible = consensus_winner is not None and not identity and not low_signal
        folds.append({
            "heldout_conformer_index": holdout,
            "inner_jackknife": inner,
            "consensus": consensus_winner is not None,
            "consensus_winner": (
                sequences[consensus_winner] if consensus_winner is not None else None
            ),
            "selector_identity": identity,
            "low_training_signal": low_signal,
            "strict_eligible": strict_eligible,
            "ensemble_minus_single_state": consensus_effect,
        })
    consensus_effects = [
        row["ensemble_minus_single_state"] for row in folds if row["consensus"]
    ]
    strict_effects = [
        row["ensemble_minus_single_state"] for row in folds if row["strict_eligible"]
    ]
    return {
        "component_id": component["component_id"],
        "method": method_row["method"],
        "attempted_folds": len(folds),
        "consensus_folds": len(consensus_effects),
        "strict_eligible_folds": len(strict_effects),
        "consensus_mean_effect": (
            float(np.mean(consensus_effects)) if consensus_effects else None
        ),
        "strict_eligible_mean_effect": (
            float(np.mean(strict_effects)) if strict_effects else None
        ),
        "consensus_negative_folds": sum(value < 0 for value in consensus_effects),
        "strict_eligible_negative_folds": sum(value < 0 for value in strict_effects),
        "folds": folds,
    }


def summarize(rows):
    all_folds = [fold for row in rows for fold in row["folds"]]
    consensus = [fold for fold in all_folds if fold["consensus"]]
    strict = [fold for fold in all_folds if fold["strict_eligible"]]
    consensus_effects = [fold["ensemble_minus_single_state"] for fold in consensus]
    strict_effects = [fold["ensemble_minus_single_state"] for fold in strict]
    abstention_reasons = Counter()
    for fold in all_folds:
        if not fold["consensus"]:
            abstention_reasons["jackknife_disagreement"] += 1
        elif fold["selector_identity"]:
            abstention_reasons["selector_identity"] += 1
        elif fold["low_training_signal"]:
            abstention_reasons["low_training_signal"] += 1
    return {
        "components": len(rows),
        "attempted_folds": len(all_folds),
        "consensus_folds": len(consensus),
        "consensus_coverage": len(consensus) / len(all_folds),
        "consensus_mean_effect": (
            float(np.mean(consensus_effects)) if consensus_effects else None
        ),
        "consensus_negative_folds": sum(value < 0 for value in consensus_effects),
        "strict_eligible_folds": len(strict),
        "strict_coverage": len(strict) / len(all_folds),
        "strict_mean_effect": float(np.mean(strict_effects)) if strict_effects else None,
        "strict_negative_folds": sum(value < 0 for value in strict_effects),
        "abstention_reasons": dict(abstention_reasons),
        "component_results": rows,
    }


def write_report(path, output):
    primary = output["method_summary"][output["primary_method"]]
    lines = [
        "# ECLS v2 Stability Abstention", "",
        "This is an exposed-panel post-failure diagnostic, not confirmation.", "",
        f"- Consensus coverage: {primary['consensus_coverage']:.3f}",
        f"- Consensus mean effect: {primary['consensus_mean_effect']}",
        f"- Strict coverage: {primary['strict_coverage']:.3f}",
        f"- Strict mean effect: {primary['strict_mean_effect']}", "",
        "| Component | Consensus folds | Strict folds | Consensus mean | Negative folds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary["component_results"]:
        lines.append(
            f"| {row['component_id']} | {row['consensus_folds']}/5 | "
            f"{row['strict_eligible_folds']}/5 | {row['consensus_mean_effect']} | "
            f"{row['consensus_negative_folds']} |"
        )
    lines.extend(["", f"Claim boundary: {output['claim_boundary']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/idp_ensemble_ecls_v2_stability_abstention_dev_v1.yml",
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
        method: summarize(rows) for method, rows in by_method.items()
    }
    output = {
        "schema_version": 1,
        "status": "stability_abstention_diagnostic_complete",
        "classification": config["classification"],
        "primary_method": config["primary_method"],
        "selector": config["selector"],
        "development_decision": config["development_decision"],
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
    primary = method_summary[config["primary_method"]]
    print(json.dumps({
        key: primary[key] for key in (
            "consensus_coverage", "consensus_mean_effect", "consensus_negative_folds",
            "strict_coverage", "strict_mean_effect", "strict_negative_folds",
            "abstention_reasons",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
