#!/usr/bin/env python
"""Analyze frozen v2 component-level endpoints under the preregistered validity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values, trials, seed):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(trials, dtype=np.float64)
    for index in range(trials):
        estimates[index] = generator.choice(array, size=len(array), replace=True).mean()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/multiscaffold_confirmatory_v2_analysis.yml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {
        name: ROOT / config[name]
        for name in ("selection", "af2_results", "prodigy_results")}
    for name, path in paths.items():
        if sha256(path) != config[f"{name}_sha256"]:
            raise ValueError(f"Frozen hash mismatch: {path}")
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    af2 = json.loads(paths["af2_results"].read_text(encoding="utf-8"))
    prodigy = json.loads(paths["prodigy_results"].read_text(encoding="utf-8"))

    components = sorted({row["component_id"] for row in selection["selections"]})
    candidate_arms = [config["primary"]["proposed_arm"], *config["primary"]["baseline_arms"]]
    selection_by_group = defaultdict(list)
    for row in selection["selections"]:
        selection_by_group[(row["component_id"], row["arm"])].append(row)
    prodigy_entities = prodigy["entities"]
    component_arm = []
    component_arm_scores = {}
    required_candidates = int(config["validity"]["arm_component_requires_selected_unique_candidates"])
    required_seeds = int(config["validity"]["candidate_requires_successful_seeds"])
    for component in components:
        for arm in candidate_arms:
            slots = selection_by_group[(component, arm)]
            selected_slots = [row for row in slots if row["status"] == "selected"]
            candidate_values = []
            invalid_entities = []
            for slot in selected_slots:
                entity = prodigy_entities.get(slot["selection_id"])
                if not entity or entity["n_successful_seeds"] != required_seeds:
                    invalid_entities.append(slot["selection_id"])
                else:
                    candidate_values.append(entity["maximum_worst_delta_g_kcal_mol"])
            valid = (
                len(selected_slots) == required_candidates
                and len(candidate_values) == required_candidates
                and not invalid_entities)
            score = float(np.mean(candidate_values)) if valid else None
            component_arm_scores[(component, arm)] = score
            component_arm.append({
                "component_id": component,
                "arm": arm,
                "selected_slots": len(selected_slots),
                "valid_candidate_scores": len(candidate_values),
                "invalid_entities": invalid_entities,
                "valid": valid,
                "mean_candidate_worst_delta_g_kcal_mol": score,
            })

    proposed = config["primary"]["proposed_arm"]
    baselines = config["primary"]["baseline_arms"]
    primary_rows = []
    for component in components:
        proposed_score = component_arm_scores[(component, proposed)]
        baseline_scores = {
            arm: component_arm_scores[(component, arm)] for arm in baselines}
        valid = proposed_score is not None and all(
            value is not None for value in baseline_scores.values())
        oracle_arm = None
        oracle_score = None
        delta = None
        if valid:
            oracle_arm, oracle_score = min(
                baseline_scores.items(), key=lambda item: (item[1], item[0]))
            delta = oracle_score - proposed_score
        primary_rows.append({
            "component_id": component,
            "valid": valid,
            "proposed_score": proposed_score,
            "baseline_scores": baseline_scores,
            "oracle_baseline_arm": oracle_arm,
            "oracle_baseline_score": oracle_score,
            "positive_delta": delta,
        })
    deltas = [row["positive_delta"] for row in primary_rows if row["valid"]]
    valid_count = len(deltas)
    valid_fraction = valid_count / len(components)
    positive_fraction = sum(value > 0 for value in deltas) / valid_count if deltas else 0.0
    ci95 = bootstrap_mean(
        deltas, int(config["statistics"]["bootstrap_trials"]),
        int(config["statistics"]["bootstrap_seed"]))
    gates = config["gates"]
    gate_results = {
        "minimum_valid_components": valid_count >= int(gates["minimum_valid_components"]),
        "minimum_valid_fraction": valid_fraction >= float(gates["minimum_valid_fraction"]),
        "minimum_positive_component_fraction": (
            positive_fraction >= float(gates["minimum_positive_component_fraction"])),
        "bootstrap_ci95_lower_above_zero": bool(ci95 and ci95[0] > 0),
    }
    primary_passed = all(gate_results.values())

    arm_summary = {}
    for arm in candidate_arms:
        values = [
            component_arm_scores[(component, arm)] for component in components
            if component_arm_scores[(component, arm)] is not None]
        arm_summary[arm] = {
            "valid_components": len(values),
            "mean_component_score": float(np.mean(values)) if values else None,
            "median_component_score": float(np.median(values)) if values else None,
        }

    af2_by_entity = defaultdict(list)
    for row in af2["results"]:
        if row["status"] == "success":
            af2_by_entity[row["entity_id"]].append(row)
    af2_arm_values = defaultdict(lambda: defaultdict(list))
    for entity in af2["entities"]:
        rows = af2_by_entity[entity["entity_id"]]
        if len(rows) != len(af2["requested"]["seeds"]):
            continue
        for metric in config["secondary"]["af2_metrics"]:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            if len(values) == len(rows):
                af2_arm_values[entity["arm"]][metric].append(float(np.median(values)))
    af2_summary = {
        arm: {
            metric: {
                "entities": len(values),
                "mean_entity_median": float(np.mean(values)) if values else None,
                "median_entity_median": float(np.median(values)) if values else None,
            }
            for metric, values in metrics.items()
        }
        for arm, metrics in af2_arm_values.items()
    }

    output = {
        "schema_version": 1,
        "status": "primary_gate_passed" if primary_passed else "primary_gate_failed",
        "config": args.config,
        "config_sha256": sha256(config_path),
        "inputs": {name: sha256(path) for name, path in paths.items()},
        "primary": {
            "passed": primary_passed,
            "valid_components": valid_count,
            "total_components": len(components),
            "valid_fraction": valid_fraction,
            "mean_positive_delta": float(np.mean(deltas)) if deltas else None,
            "median_positive_delta": float(np.median(deltas)) if deltas else None,
            "positive_component_fraction": positive_fraction,
            "component_bootstrap_ci95": ci95,
            "gate_results": gate_results,
            "rows": primary_rows,
        },
        "arm_summary": arm_summary,
        "component_arm": component_arm,
        "af2_secondary": af2_summary,
        "claim_boundary": config["claim_boundary"],
    }
    out_path = ROOT / (args.out or config["output"])
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": output["status"],
        "primary": {key: value for key, value in output["primary"].items() if key != "rows"},
        "arm_summary": output["arm_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
