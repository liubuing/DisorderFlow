#!/usr/bin/env python
"""Select balanced development checkpoints and evaluate precommitted gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("plddt", "iptm", "pae")
MAE_GATES = {"plddt": 0.05, "iptm": 0.05, "pae": 0.10}
MIN_RELIABLE_PAIRS = 30
MIN_PAIR_SCAFFOLDS = 3
MAX_SCAFFOLD_PAIR_FRACTION = 0.50


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left, right):
    left, right = rankdata(left), rankdata(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metric_view(evaluation, name):
    return evaluation["metrics"][name]["entity_aggregated"]


def selection_key(evaluation):
    metrics = [metric_view(evaluation, name) for name in OUTPUTS]
    correlations = [
        metric["within_scaffold"]["mean_group_spearman"] for metric in metrics]
    correlations = [
        value if value is not None else float("-inf") for value in correlations]
    maes = [metric["mae"] for metric in metrics]
    passed_mae = sum(
        value <= MAE_GATES[name] for name, value in zip(OUTPUTS, maes, strict=True))
    normalized_mae = sum(
        value / MAE_GATES[name] for name, value in zip(OUTPUTS, maes, strict=True))
    return min(correlations), float(np.mean(correlations)), passed_mae, -normalized_mae


def pair_outcomes(rows, name):
    outcomes = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            target_delta = rows[left][f"target_{name}"] - rows[right][f"target_{name}"]
            noise = float(np.hypot(
                rows[left][f"noise_{name}"], rows[right][f"noise_{name}"]))
            if abs(target_delta) <= noise:
                continue
            prediction_delta = (
                rows[left][f"pred_{name}"] - rows[right][f"pred_{name}"])
            outcomes.append(float(np.sign(target_delta) == np.sign(prediction_delta)))
    return outcomes


def pair_statistics(records, name, bootstrap_seed):
    groups = defaultdict(list)
    for row in records:
        groups[row["scaffold_family"]].append(row)
    outcomes_by_scaffold = {
        scaffold: pair_outcomes(rows, name)
        for scaffold, rows in groups.items()
    }
    outcomes = [
        outcome for scaffold_outcomes in outcomes_by_scaffold.values()
        for outcome in scaffold_outcomes
    ]
    contributing_scaffolds = sum(
        bool(scaffold_outcomes)
        for scaffold_outcomes in outcomes_by_scaffold.values())
    max_scaffold_pair_fraction = (
        max(map(len, outcomes_by_scaffold.values()), default=0) / len(outcomes)
        if outcomes else None)
    if not outcomes:
        return {
            "pairs": 0,
            "contributing_scaffolds": 0,
            "max_scaffold_pair_fraction": None,
            "accuracy": None,
            "bootstrap_95_lower": None,
            "bootstrap_unit": "scaffold_then_entity",
        }
    outcomes = np.asarray(outcomes)
    generator = np.random.default_rng(bootstrap_seed)
    scaffolds = list(groups)
    bootstrap = []
    for _ in range(10000):
        sampled_scaffolds = generator.choice(
            scaffolds, size=len(scaffolds), replace=True)
        replicate = []
        for scaffold in sampled_scaffolds:
            rows = groups[scaffold]
            sampled_indices = generator.choice(
                len(rows), size=len(rows), replace=True)
            sampled_rows = [rows[index] for index in sampled_indices]
            replicate.extend(pair_outcomes(sampled_rows, name))
        if replicate:
            bootstrap.append(float(np.mean(replicate)))
    return {
        "pairs": len(outcomes),
        "contributing_scaffolds": contributing_scaffolds,
        "max_scaffold_pair_fraction": max_scaffold_pair_fraction,
        "accuracy": float(outcomes.mean()),
        "bootstrap_95_lower": (
            float(np.quantile(bootstrap, 0.025)) if bootstrap else None),
        "bootstrap_unit": "scaffold_then_entity",
    }


def scaffold_correlations(records, name):
    groups = defaultdict(list)
    for row in records:
        groups[row["scaffold_family"]].append(row)
    output = {}
    for scaffold, rows in groups.items():
        output[scaffold] = spearman(
            [row[f"pred_{name}"] for row in rows],
            [row[f"target_{name}"] for row in rows])
    return output


def dispersion_ratios(prediction_std, target_std):
    if target_std <= 0:
        return None, None
    std_ratio = prediction_std / target_std
    return std_ratio, std_ratio ** 2


def summarize_selection(evaluation, bootstrap_seed):
    records = evaluation["entity_aggregated_records"]
    summary = {
        "checkpoint": evaluation["checkpoint"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "iteration": evaluation["iteration"],
        "weights_kind": evaluation["weights_kind"],
        "metrics": {},
    }
    all_pass = True
    for name in OUTPUTS:
        metric = metric_view(evaluation, name)
        pairs = pair_statistics(records, name, bootstrap_seed)
        correlations = scaffold_correlations(records, name)
        defined_correlations = [value for value in correlations.values() if value is not None]
        median_spearman = (
            float(np.median(defined_correlations)) if defined_correlations else None)
        stability = evaluation["metrics"][name]["condition_stability"]
        replicate_std = stability["mean_target_std_across_conditions"]
        condition_std_ratio, condition_variance_ratio = dispersion_ratios(
            stability["mean_prediction_std_across_conditions"], replicate_std)
        entity_std_ratio, entity_variance_ratio = dispersion_ratios(
            metric["prediction_std"], metric["target_std"])
        gates = {
            "mae": metric["mae"] <= MAE_GATES[name],
            "pair_evidence": (
                pairs["pairs"] >= MIN_RELIABLE_PAIRS
                and pairs["contributing_scaffolds"] >= MIN_PAIR_SCAFFOLDS
                and pairs["max_scaffold_pair_fraction"] is not None
                and pairs["max_scaffold_pair_fraction"]
                <= MAX_SCAFFOLD_PAIR_FRACTION),
            "pair_accuracy": (
                pairs["accuracy"] is not None and pairs["accuracy"] >= 0.65),
            "pair_bootstrap_lower": (
                pairs["bootstrap_95_lower"] is not None
                and pairs["bootstrap_95_lower"] > 0.50),
            "median_scaffold_spearman": (
                median_spearman is not None and median_spearman >= 0.50),
            "entity_variance_ratio": (
                entity_variance_ratio is not None
                and 0.5 <= entity_variance_ratio <= 2.0),
            "cross_condition_variance_ratio": (
                condition_variance_ratio is not None
                and condition_variance_ratio <= 2.0),
            "no_negative_scaffold": (
                len(defined_correlations) == len(correlations)
                and all(value >= 0 for value in defined_correlations)),
        }
        all_pass &= all(gates.values())
        summary["metrics"][name] = {
            "mae": metric["mae"],
            "pairwise": pairs,
            "scaffold_spearman": correlations,
            "median_scaffold_spearman": median_spearman,
            "entity_prediction_to_target_std_ratio": entity_std_ratio,
            "entity_prediction_to_target_variance_ratio": entity_variance_ratio,
            "cross_condition_prediction_to_replicate_std_ratio": condition_std_ratio,
            "cross_condition_prediction_to_replicate_variance_ratio": (
                condition_variance_ratio),
            "gates": gates,
        }
    summary["deployment_gates_passed"] = all_pass
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluations", nargs="+", default=[
            "results/confidence_candidate_interface_multiscaffold_dual_sem_v1/evaluation_s2041.json",
            "results/confidence_candidate_interface_multiscaffold_dual_sem_v1/evaluation_s2053.json",
            "results/confidence_candidate_interface_multiscaffold_dual_sem_v1/evaluation_s2069.json",
        ])
    parser.add_argument(
        "--output",
        default=(
            "results/confidence_candidate_interface_multiscaffold_dual_sem_v1/"
            "selection_summary_semantics_v2.json"))
    args = parser.parse_args()

    selections = []
    sources = []
    dataset_hashes = set()
    for index, relative_path in enumerate(args.evaluations):
        path = ROOT / relative_path
        document = json.loads(path.read_text(encoding="utf-8"))
        dataset_hashes.add(document["dataset_manifest_sha256"])
        candidates = [
            row for row in document["evaluations"] if row["weights_kind"] == "raw"]
        selected = max(candidates, key=selection_key)
        selections.append(summarize_selection(selected, bootstrap_seed=2041 + index))
        sources.append({"path": relative_path, "sha256": sha256(path)})

    if len(dataset_hashes) != 1:
        raise ValueError("Evaluation files do not share one dataset manifest")
    output = {
        "schema_version": "candidate_interface_multiseed_selection_v2",
        "classification": "development_only",
        "deployment_gate_passed": all(
            row["deployment_gates_passed"] for row in selections),
        "selection_rule": (
            "raw periodic checkpoints only; lexicographically maximize minimum "
            "output scaffold Spearman, mean output scaffold Spearman, number of "
            "absolute MAE gates passed, then minimize normalized MAE"),
        "dataset_manifest_sha256": dataset_hashes.pop(),
        "evaluation_sources": sources,
        "selections": selections,
        "final_test_evaluated": False,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
