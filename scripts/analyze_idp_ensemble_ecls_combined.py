#!/usr/bin/env python
"""Apply the predeclared 12-component ECLS gate without endpoint retuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_idp_ensemble_matched_candidates import bootstrap_mean, exact_sign_flip_p  # noqa: E402,I001

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component_effects(analysis, method):
    rows = analysis["method_summary"][method]["components"]
    return {
        row["component_id"]: float(row["mean_ensemble_minus_single_state"])
        for row in rows
    }


def summarize(effects, config, seed_offset=0):
    values = list(effects.values())
    statistics = config["statistics"]
    gates = config["predeclared_gates"]
    ci = bootstrap_mean(
        values,
        int(statistics["bootstrap_trials"]),
        int(statistics["bootstrap_seed"]) + seed_offset,
    )
    positive = sum(value > 0 for value in values)
    zero = sum(value == 0 for value in values)
    negative = sum(value < 0 for value in values)
    gate_results = {
        "minimum_valid_components": len(values) >= int(gates["minimum_valid_components"]),
        "minimum_strictly_positive_components": positive
        >= int(gates["minimum_strictly_positive_components"]),
        "bootstrap_ci95_lower_above_zero": bool(ci and ci[0] > 0),
    }
    return {
        "valid_components": len(values),
        "mean_ensemble_minus_single_state": float(np.mean(values)),
        "median_ensemble_minus_single_state": float(np.median(values)),
        "strictly_positive_components": positive,
        "zero_components": zero,
        "negative_components": negative,
        "positive_component_fraction": positive / len(values),
        "component_bootstrap_ci95": ci,
        "exact_sign_flip_p": exact_sign_flip_p(values),
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "components": [
            {"component_id": component_id, "mean_ensemble_minus_single_state": value}
            for component_id, value in effects.items()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_ecls_combined_12_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = {}
    analyses = {}
    for name in ("development_analysis", "expansion_analysis"):
        path = ROOT / config[name]["path"]
        digest = sha256(path)
        if digest != config[name]["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {path}")
        inputs[name] = {"path": config[name]["path"], "sha256": digest}
        analyses[name] = json.loads(path.read_text(encoding="utf-8"))

    method_summary = {}
    sensitivity_summary = {}
    excluded = config["sensitivity"]["excluded_component"]
    for method_index, method in enumerate(config["methods"]):
        development = component_effects(analyses["development_analysis"], method)
        expansion = component_effects(analyses["expansion_analysis"], method)
        overlap = set(development) & set(expansion)
        if overlap:
            raise ValueError(f"Component overlap between panels: {sorted(overlap)}")
        combined = {**development, **expansion}
        method_summary[method] = summarize(combined, config, method_index)
        sensitivity = {key: value for key, value in combined.items() if key != excluded}
        sensitivity_config = {
            **config,
            "predeclared_gates": {
                **config["predeclared_gates"],
                "minimum_valid_components": 11,
                "minimum_strictly_positive_components": 8,
            },
        }
        sensitivity_summary[method] = summarize(
            sensitivity, sensitivity_config, 100 + method_index
        )

    primary = config["primary_method"]
    output = {
        "schema_version": 1,
        "status": "predeclared_primary_gate_passed"
        if method_summary[primary]["passed"] else "predeclared_primary_gate_failed",
        "classification": config["classification"],
        "primary_method": primary,
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "inputs": inputs,
        },
        "method_summary": method_summary,
        "sensitivity_excluding_synthetic_component": {
            "excluded_component": excluded,
            "methods": sensitivity_summary,
        },
        "claim_boundary": config["claim_boundary"],
    }
    output_path = ROOT / (args.out or config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": output["status"],
        "primary": method_summary[primary],
        "sensitivity": sensitivity_summary[primary],
    }, indent=2))


if __name__ == "__main__":
    main()
