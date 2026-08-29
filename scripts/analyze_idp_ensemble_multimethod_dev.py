#!/usr/bin/env python
"""Analyze the exposed six-component IDP ensemble development pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values, trials, seed):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = generator.choice(array, size=(trials, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def exact_sign_flip_p(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    estimates = [
        abs(float(np.mean(array * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ]
    return float(np.mean(np.asarray(estimates) >= observed - 1e-12))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_fold_rows(abeta_rows, nonabeta_rows):
    rows = []
    for row in abeta_rows:
        component = f"ABETA_{row['reference_pdb']}"
        fold = row["heldout_conformer"]
        scores = {
            "contact_ensemble_robust": float(row["ensemble_holdout_score"]),
            "contact_single_state": float(row["single_state_selection_median"]),
            "native": float(row["native_holdout_score"]),
            "composition_matched_random": float(row["random_selection_median"]),
        }
        rows.extend(
            {"component_id": component, "fold": fold, "method": method, "score": score}
            for method, score in scores.items()
        )
    target_components = {"tau": "TAU_5MP3", "alpha_synuclein": "ASYN_8B9V"}
    for row in nonabeta_rows:
        component = target_components[row["target"]]
        fold = str(row["heldout_conformer"])
        rows.extend([
            {
                "component_id": component,
                "fold": fold,
                "method": "contact_ensemble_robust",
                "score": float(row["heldout_score"]),
            },
            {
                "component_id": component,
                "fold": fold,
                "method": "native",
                "score": float(row["native_score"]),
            },
        ])
    return rows


def component_method_means(fold_rows):
    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[(row["component_id"], row["method"])].append(float(row["score"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_summary(component_scores, proposed, baseline, components, trials, seed):
    rows = []
    for component in components:
        proposed_score = component_scores.get((component, proposed))
        baseline_score = component_scores.get((component, baseline))
        valid = proposed_score is not None and baseline_score is not None
        rows.append({
            "component_id": component,
            "valid": valid,
            "proposed_score": proposed_score,
            "baseline_score": baseline_score,
            "delta": proposed_score - baseline_score if valid else None,
        })
    deltas = [row["delta"] for row in rows if row["valid"]]
    return {
        "proposed": proposed,
        "baseline": baseline,
        "valid_components": len(deltas),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "median_delta": float(np.median(deltas)) if deltas else None,
        "positive_component_fraction": (
            sum(value > 0 for value in deltas) / len(deltas) if deltas else 0.0
        ),
        "component_bootstrap_ci95": bootstrap_mean(deltas, trials, seed),
        "exact_sign_flip_p": exact_sign_flip_p(deltas),
        "rows": rows,
    }


def analyze(config, manifest, abeta_rows, nonabeta_rows):
    components = [row["component_id"] for row in manifest["components"]]
    fold_rows = build_fold_rows(abeta_rows, nonabeta_rows)
    scores = component_method_means(fold_rows)
    trials = int(config["statistics"]["bootstrap_trials"])
    seed = int(config["statistics"]["bootstrap_seed"])
    primary = paired_summary(
        scores, "contact_ensemble_robust", "contact_single_state", components, trials, seed
    )
    secondary = {
        "versus_native": paired_summary(
            scores, "contact_ensemble_robust", "native", components, trials, seed + 1
        ),
        "versus_random": paired_summary(
            scores,
            "contact_ensemble_robust",
            "composition_matched_random",
            components,
            trials,
            seed + 2,
        ),
    }
    method_availability = {
        config["methods"]["proposed"]["id"]: config["methods"]["proposed"]["availability"]
    }
    method_availability.update({row["id"]: row["availability"] for row in config["methods"]["baselines"]})
    gates = config["gates"]
    gate_results = {
        "minimum_valid_components": primary["valid_components"]
        >= int(gates["minimum_valid_components"]),
        "minimum_positive_component_fraction": primary["positive_component_fraction"]
        >= float(gates["minimum_positive_component_fraction"]),
        "bootstrap_ci95_lower_above_zero": bool(
            primary["component_bootstrap_ci95"]
            and primary["component_bootstrap_ci95"][0] > 0
        ),
        "complete_matched_arms": all(
            method_availability.get(method) == "complete_on_6_components"
            for method in gates["require_complete_matched_arms"]
        ),
    }
    passed = all(gate_results.values())
    return {
        "schema_version": 1,
        "status": "development_gate_passed" if passed else "development_gate_failed",
        "classification": config["classification"],
        "component_count": len(components),
        "target_count": len({row["target"] for row in manifest["components"]}),
        "ensemble_source_count": len({row["ensemble_source"] for row in manifest["components"]}),
        "method_availability": method_availability,
        "primary": {**primary, "gate_results": gate_results, "passed": passed},
        "secondary": secondary,
        "fold_rows": fold_rows,
        "claim_boundary": config["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_multimethod_dev_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["component_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len({row["component_id"] for row in manifest["components"]}) != len(manifest["components"]):
        raise ValueError("Component IDs must be unique")

    input_paths = {name: ROOT / item["path"] for name, item in config["inputs"].items()}
    for name, path in input_paths.items():
        expected = config["inputs"][name]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Frozen hash mismatch: {path}")
    for component in manifest["components"]:
        path = ROOT / component["reference_path"]
        if sha256(path) != component["reference_sha256"]:
            raise ValueError(f"Reference hash mismatch: {path}")

    output = analyze(
        config,
        manifest,
        read_csv(input_paths["abeta_evaluation"]),
        read_csv(input_paths["nonabeta_evaluation"]),
    )
    output["provenance"] = {
        "config": args.config,
        "config_sha256": sha256(config_path),
        "component_manifest": config["component_manifest"],
        "component_manifest_sha256": sha256(manifest_path),
        "inputs": {name: sha256(path) for name, path in input_paths.items()},
    }
    out_path = ROOT / (args.out or config["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": output["status"],
        "component_count": output["component_count"],
        "primary": {key: value for key, value in output["primary"].items() if key != "rows"},
        "output": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
