#!/usr/bin/env python
"""Decompose ECLS gate failures into selector, signal, and leverage mechanisms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
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


def hamming(left, right):
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length sequences")
    return sum(a != b for a, b in zip(left, right, strict=True))


def diversity_metrics(sequences, raw_sequences):
    if not raw_sequences:
        raise ValueError("Raw candidate provenance is required")
    if set(raw_sequences) != set(sequences):
        raise ValueError("Raw and ECLS-deduplicated candidate pools do not match")
    distances = [hamming(left, right) for left, right in combinations(sequences, 2)]
    length = len(sequences[0])
    if distances:
        distance_summary = {
            "minimum": int(np.min(distances)),
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "maximum": int(np.max(distances)),
            "mean_fraction": float(np.mean(distances) / length),
        }
    else:
        distance_summary = {
            "minimum": 0, "mean": 0.0, "median": 0.0,
            "maximum": 0, "mean_fraction": 0.0,
        }
    return {
        "raw_candidates": len(raw_sequences),
        "unique_candidates": len(sequences),
        "duplicate_fraction": 1.0 - len(set(raw_sequences)) / len(raw_sequences),
        "sequence_length": length,
        "pairwise_hamming": distance_summary,
        "collapsed": len(sequences) == 1,
        "near_clonal": len(sequences) > 1 and distance_summary["minimum"] == 1,
    }


def winner_index(matrix, indices, sequences, aggregation):
    return max(
        range(len(sequences)),
        key=lambda index: (robust_value(matrix[index, indices], aggregation), sequences[index]),
    )


def classify_effect(mean_effect, strict_identity_folds, fold_count, thresholds):
    epsilon = float(thresholds["mathematical_effect_epsilon"])
    practical = float(thresholds["practical_neutral_absolute_effect"])
    if mean_effect < -epsilon:
        return "negative_transfer"
    if abs(mean_effect) <= epsilon:
        return "selector_identity" if strict_identity_folds == fold_count else "score_equivalence"
    if abs(mean_effect) < practical:
        return "numerical_near_tie"
    return "positive_effect"


def analyze_component_method(
    component, method_row, raw_sequences, stored_rows, aggregation, thresholds
):
    sequences = method_row["sequences"]
    matrix = np.asarray(method_row["epitope_conditioning_gain"], dtype=np.float64)
    if matrix.shape != (len(sequences), 5):
        raise ValueError(f"{component['component_id']} requires a candidate x 5 matrix")
    all_conformers = list(range(matrix.shape[1]))
    full_winner = winner_index(matrix, all_conformers, sequences, aggregation)
    folds = []
    tolerance = float(thresholds["stored_loo_reproduction_tolerance"])
    for holdout in all_conformers:
        training = [index for index in all_conformers if index != holdout]
        ensemble_index = winner_index(matrix, training, sequences, aggregation)
        single_indices = [int(np.argmax(matrix[:, conformer])) for conformer in training]
        single_scores = [float(matrix[index, holdout]) for index in single_indices]
        ensemble_score = float(matrix[ensemble_index, holdout])
        single_score = float(np.median(single_scores))
        effect = ensemble_score - single_score
        stored = stored_rows[holdout]
        if stored["ensemble_candidate"] != sequences[ensemble_index]:
            raise ValueError("Stored ensemble candidate reproduction failed")
        if abs(float(stored["ensemble_minus_single_state"]) - effect) > tolerance:
            raise ValueError("Stored LOO score reproduction failed")
        deletion_margin = robust_value(
            matrix[ensemble_index, training], aggregation
        ) - robust_value(matrix[full_winner, training], aggregation)
        folds.append({
            "heldout_conformer_index": holdout,
            "ensemble_candidate": sequences[ensemble_index],
            "ensemble_candidate_index": ensemble_index,
            "single_state_candidates": [sequences[index] for index in single_indices],
            "single_state_candidate_indices": single_indices,
            "same_as_any_single_winner": ensemble_index in single_indices,
            "same_as_all_single_winners": all(
                index == ensemble_index for index in single_indices
            ),
            "ensemble_score": ensemble_score,
            "single_state_score": single_score,
            "ensemble_minus_single_state": effect,
            "negative_transfer": effect
            < -float(thresholds["mathematical_effect_epsilon"]),
            "changes_full_panel_winner": ensemble_index != full_winner,
            "deletion_margin": deletion_margin,
        })

    effects = [row["ensemble_minus_single_state"] for row in folds]
    strict_identity = sum(row["same_as_all_single_winners"] for row in folds)
    influential = [row for row in folds if row["changes_full_panel_winner"]]
    leverage_epsilon = float(thresholds["leverage_margin_epsilon"])
    if len(influential) == 1 and influential[0]["deletion_margin"] > leverage_epsilon:
        leverage = "isolated_anomalous_leverage"
    elif len(influential) >= 2:
        leverage = "diffuse_instability"
    elif influential:
        leverage = "numerical_near_tie_leverage"
    else:
        leverage = "stable_full_panel_winner"

    candidate_variances = np.var(matrix, axis=1)
    q1, q3 = np.percentile(candidate_variances, [25, 75])
    winner_variance = float(candidate_variances[full_winner])
    mean_effect = float(np.mean(effects))
    outcome = classify_effect(mean_effect, strict_identity, len(folds), thresholds)
    return {
        "component_id": component["component_id"],
        "method": method_row["method"],
        "native_h3": component["native_h3"],
        "diversity": diversity_metrics(sequences, raw_sequences),
        "score_signal": {
            "candidate_variance_median": float(np.median(candidate_variances)),
            "candidate_variance_iqr": [float(q1), float(q3)],
            "full_winner_variance": winner_variance,
            "full_winner_high_variance_outlier": bool(
                winner_variance > q3 + 1.5 * (q3 - q1)
            ),
            "per_conformer_candidate_score_range": [
                float(np.ptp(matrix[:, index])) for index in all_conformers
            ],
            "all_candidate_scores_conformer_invariant": bool(
                np.all(np.ptp(matrix, axis=1)
                       <= float(thresholds["mathematical_effect_epsilon"]))
            ),
        },
        "selection": {
            "full_panel_winner": sequences[full_winner],
            "loo_unique_ensemble_winners": len({
                row["ensemble_candidate"] for row in folds
            }),
            "winner_turnover_rate": (
                len({row["ensemble_candidate"] for row in folds}) - 1
            ) / (len(folds) - 1),
            "strict_selector_identity_folds": strict_identity,
            "any_selector_identity_folds": sum(
                row["same_as_any_single_winner"] for row in folds
            ),
            "full_winner_switch_holdouts": [
                row["heldout_conformer_index"] for row in influential
            ],
            "leverage_class": leverage,
        },
        "effect": {
            "mean_ensemble_minus_single_state": mean_effect,
            "minimum_fold_effect": float(np.min(effects)),
            "maximum_fold_effect": float(np.max(effects)),
            "negative_transfer_folds": [
                row["heldout_conformer_index"] for row in folds
                if row["negative_transfer"]
            ],
            "classification": outcome,
        },
        "folds": folds,
    }


def write_csv(path, diagnostics):
    fields = [
        "panel", "component_id", "method", "effect_classification",
        "mean_ensemble_minus_single_state", "negative_transfer_folds",
        "strict_selector_identity_folds", "loo_unique_ensemble_winners",
        "leverage_class", "full_winner_switch_holdouts", "raw_candidates",
        "unique_candidates", "duplicate_fraction", "minimum_hamming",
        "mean_hamming_fraction", "candidate_variance_median",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in diagnostics:
            writer.writerow({
                "panel": row["panel"],
                "component_id": row["component_id"],
                "method": row["method"],
                "effect_classification": row["effect"]["classification"],
                "mean_ensemble_minus_single_state": row["effect"]["mean_ensemble_minus_single_state"],
                "negative_transfer_folds": ";".join(map(str, row["effect"]["negative_transfer_folds"])),
                "strict_selector_identity_folds": row["selection"]["strict_selector_identity_folds"],
                "loo_unique_ensemble_winners": row["selection"]["loo_unique_ensemble_winners"],
                "leverage_class": row["selection"]["leverage_class"],
                "full_winner_switch_holdouts": ";".join(map(str, row["selection"]["full_winner_switch_holdouts"])),
                "raw_candidates": row["diversity"]["raw_candidates"],
                "unique_candidates": row["diversity"]["unique_candidates"],
                "duplicate_fraction": row["diversity"]["duplicate_fraction"],
                "minimum_hamming": row["diversity"]["pairwise_hamming"]["minimum"],
                "mean_hamming_fraction": row["diversity"]["pairwise_hamming"]["mean_fraction"],
                "candidate_variance_median": row["score_signal"]["candidate_variance_median"],
            })


def write_report(path, output):
    primary = [
        row for row in output["diagnostics"]
        if row["method"] == output["primary_method"]
    ]
    lines = [
        "# ECLS Failure Decomposition v1", "",
        "This is a post-gate diagnostic and does not alter the failed predeclared result.", "",
        "## Primary ProteinMPNN Components", "",
        "| Component | Panel | Effect | Mean delta | Identity folds | LOO winners | Leverage | Negative folds |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in primary:
        lines.append(
            f"| {row['component_id']} | {row['panel']} | "
            f"{row['effect']['classification']} | "
            f"{row['effect']['mean_ensemble_minus_single_state']:.8f} | "
            f"{row['selection']['strict_selector_identity_folds']}/5 | "
            f"{row['selection']['loo_unique_ensemble_winners']} | "
            f"{row['selection']['leverage_class']} | "
            f"{','.join(map(str, row['effect']['negative_transfer_folds'])) or 'none'} |"
        )
    lines.extend(["", "## v2 Priorities", ""])
    for priority in output["v2_priorities"]:
        lines.append(f"{priority['priority']}. {priority['action']}")
    lines.extend(["", f"Claim boundary: {output['claim_boundary']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_ecls_failure_decomposition_v1.yml"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded = {}
    provenance = {}
    for name, item in config["inputs"].items():
        path = ROOT / item["path"]
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {path}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        provenance[name] = {"path": item["path"], "sha256": digest}

    raw_by_key = defaultdict(list)
    for input_name in ("development_generation", "expansion_generation"):
        panel = config["inputs"][input_name]["panel"]
        for slot in loaded[input_name]["slots"]:
            raw_by_key[(panel, slot["component_id"], slot["method"])].extend(
                candidate["sequence"] for candidate in slot["candidates"]
            )

    diagnostics = []
    for input_name in ("development_analysis", "expansion_analysis"):
        analysis = loaded[input_name]
        panel = config["inputs"][input_name]["panel"]
        stored_by_key = defaultdict(dict)
        for row in analysis["leave_one_conformer_out"]:
            stored_by_key[(row["component_id"], row["method"])][
                int(row["heldout_conformer_index"])
            ] = row
        for component in analysis["component_scores"]:
            for method_row in component["methods"]:
                method = method_row["method"]
                if method not in config["methods"]:
                    continue
                row = analyze_component_method(
                    component,
                    method_row,
                    raw_by_key[(panel, component["component_id"], method)],
                    stored_by_key[(component["component_id"], method)],
                    config["aggregation"],
                    config["thresholds"],
                )
                diagnostics.append({"panel": panel, **row})

    primary_rows = [
        row for row in diagnostics if row["method"] == config["primary_method"]
    ]
    classifications = Counter(row["effect"]["classification"] for row in primary_rows)
    leverage = Counter(row["selection"]["leverage_class"] for row in primary_rows)
    output = {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "classification": config["classification"],
        "primary_method": config["primary_method"],
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "inputs": provenance,
        },
        "thresholds": config["thresholds"],
        "primary_summary": {
            "components": len(primary_rows),
            "effect_classifications": dict(classifications),
            "leverage_classifications": dict(leverage),
        },
        "v2_priorities": [
            {
                "priority": 1,
                "action": "Address isolated 5CSZ conformer leverage with predeclared conformer quality weighting or influence caps.",
            },
            {
                "priority": 2,
                "action": "Add a conformer-information and selector-identity abstention rule for zero-effect components.",
            },
            {
                "priority": 3,
                "action": "Treat effects below the diagnostic practical-neutral threshold as abstentions, not positive evidence.",
            },
            {
                "priority": 4,
                "action": "Develop v2 only on these now-exposed 12 components, then freeze a new untouched panel.",
            },
        ],
        "diagnostics": diagnostics,
        "claim_boundary": config["claim_boundary"],
    }
    json_path = ROOT / config["output"]["json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    write_csv(ROOT / config["output"]["csv"], diagnostics)
    write_report(ROOT / config["output"]["report"], output)
    print(json.dumps({
        "status": output["status"],
        "primary_summary": output["primary_summary"],
        "output": config["output"],
    }, indent=2))


if __name__ == "__main__":
    main()
