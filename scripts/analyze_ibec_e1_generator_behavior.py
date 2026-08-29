#!/usr/bin/env python
"""Analyze frozen BFN and ProteinMPNN generator behavior for iBEC E1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hamming_fraction(first, second):
    if len(first) != len(second):
        raise ValueError("H3 sequences have different lengths within one component")
    return sum(left != right for left, right in zip(first, second, strict=True)) / len(first)


def position_entropy(sequences):
    if not sequences:
        return float("nan")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("Position entropy requires matched sequence lengths")
    values = []
    for index in range(length):
        counts = Counter(sequence[index] for sequence in sequences)
        probabilities = np.asarray(list(counts.values()), dtype=float) / len(sequences)
        values.append(float(-(probabilities * np.log(probabilities)).sum()))
    return float(np.mean(values))


def mean_pairwise_hamming(sequences):
    distances = [hamming_fraction(first, second) for first, second in combinations(sequences, 2)]
    return float(np.mean(distances)) if distances else 0.0


def native_recovery(sequence, native):
    return 1.0 - hamming_fraction(sequence, native)


def summarize_group(rows, native, score_sequence):
    sequences = [row["sequence"] for row in rows]
    risks = [score_sequence(row["full_heavy_sequence"])["sequence_risk"] for row in rows]
    seeds = sorted({row["seed"] for row in rows})
    seed_unique = []
    seed_entropy = []
    for seed in seeds:
        seed_sequences = [row["sequence"] for row in rows if row["seed"] == seed]
        seed_unique.append(len(set(seed_sequences)) / len(seed_sequences))
        seed_entropy.append(position_entropy(seed_sequences))
    return {
        "n_attempts": len(rows),
        "n_unique": len(set(sequences)),
        "unique_fraction": len(set(sequences)) / len(sequences),
        "mean_position_entropy_nats": position_entropy(sequences),
        "mean_pairwise_hamming_fraction": mean_pairwise_hamming(sequences),
        "developability_pass_fraction_at_0_55": float(np.mean(np.asarray(risks) <= 0.55)),
        "mean_full_heavy_developability_risk": float(np.mean(risks)),
        "mean_native_recovery": float(np.mean([native_recovery(sequence, native) for sequence in sequences])),
        "unique_fraction_seed_std": float(np.std(seed_unique)),
        "position_entropy_seed_std": float(np.std(seed_entropy)),
    }


def paired_bootstrap(values, trials, seed):
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(trials, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def interpretation(config, differences):
    diversity_metrics = [
        "unique_fraction",
        "mean_position_entropy_nats",
        "mean_pairwise_hamming_fraction",
    ]
    established = sum(differences[metric]["ci95"][0] > 0 for metric in diversity_metrics)
    point_positive = sum(differences[metric]["mean_difference"] > 0 for metric in diversity_metrics)
    developability_difference = differences["developability_pass_fraction_at_0_55"]
    noninferior = (
        developability_difference["ci95"][0]
        >= float(config["interpretation"]["developability_noninferiority_margin"])
    )
    required = int(config["interpretation"]["diversity_advantage_requires_metrics"])
    if established >= required and noninferior:
        decision = "higher_diversity_with_noninferior_developability"
    elif point_positive >= required and not noninferior:
        decision = "expanded_exploration_with_developability_tradeoff"
    elif point_positive >= required:
        decision = "descriptive_diversity_increase_not_statistically_established"
    else:
        decision = "no_bfn_diversity_advantage"
    return {
        "decision": decision,
        "diversity_metrics_ci_lower_above_zero": established,
        "diversity_metrics_point_positive": point_positive,
        "developability_noninferior": noninferior,
        "allowed_claim": {
            "higher_diversity_with_noninferior_developability": (
                "BFN generated a more diverse H3 distribution on the frozen 20-component panel "
                "while meeting the preregistered developability noninferiority margin."
            ),
            "expanded_exploration_with_developability_tradeoff": (
                "BFN expanded H3 sequence exploration but showed a developability tradeoff."
            ),
            "descriptive_diversity_increase_not_statistically_established": (
                "BFN showed descriptive diversity increases that were not established by component-level CIs."
            ),
            "no_bfn_diversity_advantage": (
                "The frozen panel did not establish a BFN diversity advantage over ProteinMPNN."
            ),
        }[decision],
    }


def build_report(summary):
    lines = [
        "# iBEC E1 Generator Behavior Audit",
        "",
        "## Frozen Panel",
        "",
        f"- {summary['panel']['components']} independent antibody-peptide components.",
        f"- {summary['panel']['attempts']} raw generation attempts; {summary['panel']['successes']} successful.",
        "- Three seeds and eight attempts per component/arm.",
        "- Self-perplexity values are not compared across model families.",
        "",
        "## Primary Comparison",
        "",
        f"Decision: **{summary['interpretation']['decision']}**",
        "",
        summary["interpretation"]["allowed_claim"],
        "",
        "| Metric | BFN mean | ProteinMPNN mean | Paired difference | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    primary = summary["arm_means"]
    for metric, result in summary["primary_differences"].items():
        lines.append(
            f"| {metric} | {primary[summary['primary_bfn_arm']][metric]:.4f} | "
            f"{primary[summary['primary_baseline']][metric]:.4f} | "
            f"{result['mean_difference']:.4f} | [{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}] |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "Diversity is a generator-distribution property, not evidence of binding, affinity, or biological quality. The original phrase 'higher diversity and lower perplexity' must be replaced by the allowed claim above.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/ibec_e1_generator_behavior_v1.yml"
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_paths = {
        name: ROOT / config["inputs"][name]
        for name in ("raw_generation", "raw_generation_audit", "holdout_manifest")
    }
    for name, path in input_paths.items():
        if sha256(path) != config["inputs"][f"{name}_sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    raw = json.loads(input_paths["raw_generation"].read_text(encoding="utf-8"))
    audit = json.loads(input_paths["raw_generation_audit"].read_text(encoding="utf-8"))
    holdout = json.loads(input_paths["holdout_manifest"].read_text(encoding="utf-8"))
    if raw["status"] != "raw_generation_complete" or audit["status"] != "raw_generation_integrity_passed":
        raise ValueError("Raw generation integrity did not pass")
    if raw["requested"]["attempts"] != int(config["panel"]["components"]) * 5 * 3 * 8:
        raise ValueError("Frozen attempt count differs from E1 contract")
    from scripts.pipeline.score_shortlist_developability import score_sequence

    native_by_component = {
        component["component_id"]: component["representative"]["cdr_h3_sequence"]
        for component in holdout["components"]
    }
    grouped = defaultdict(list)
    for row in raw["attempts"]:
        if row["status"] == "success":
            grouped[(row["component_id"], row["arm"])].append(row)
    arms = raw["requested"]["arms"]
    component_rows = []
    metrics_by_component_arm = {}
    for component in raw["requested"]["components"]:
        for arm in arms:
            rows = grouped[(component, arm)]
            if len(rows) != 24:
                raise ValueError(f"Incomplete group: {component} {arm}")
            metrics = summarize_group(
                rows, native_by_component[component], score_sequence
            )
            metrics_by_component_arm[(component, arm)] = metrics
            component_rows.append({"component_id": component, "arm": arm, **metrics})
    numeric_metrics = [
        key for key in component_rows[0]
        if key not in {"component_id", "arm", "n_attempts", "n_unique"}
    ]
    arm_means = {
        arm: {
            metric: float(np.mean([
                metrics_by_component_arm[(component, arm)][metric]
                for component in raw["requested"]["components"]
            ]))
            for metric in numeric_metrics
        }
        for arm in arms
    }
    primary = config["comparison"]["primary_bfn_arm"]
    baseline = config["comparison"]["primary_baseline"]
    differences = {}
    for metric in numeric_metrics:
        paired = [
            metrics_by_component_arm[(component, primary)][metric]
            - metrics_by_component_arm[(component, baseline)][metric]
            for component in raw["requested"]["components"]
        ]
        differences[metric] = {
            "mean_difference": float(np.mean(paired)),
            "ci95": paired_bootstrap(
                paired,
                int(config["statistics"]["bootstrap_trials"]),
                int(config["statistics"]["bootstrap_seed"]),
            ),
            "component_differences": paired,
        }
    result_interpretation = interpretation(config, differences)
    summary = {
        "schema_version": 1,
        "status": "ibec_e1_generator_behavior_complete",
        "config": args.config,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "panel": {
            "components": len(raw["requested"]["components"]),
            "arms": len(arms),
            "attempts": raw["summary"]["recorded_attempts"],
            "successes": raw["summary"]["successes"],
            "seeds": raw["requested"]["seeds"],
            "samples_per_seed": raw["requested"]["samples_per_seed"],
        },
        "primary_bfn_arm": primary,
        "primary_baseline": baseline,
        "arm_means": arm_means,
        "primary_differences": differences,
        "interpretation": result_interpretation,
        "self_perplexity_comparison_performed": False,
        "claim_boundary": config["claim_boundary"],
    }
    output_dir = ROOT / (args.out or config["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / config["outputs"]["component_metrics"]
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)
    summary["component_metrics_sha256"] = sha256(csv_path)
    (output_dir / config["outputs"]["summary"]).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / config["outputs"]["report"]).write_text(
        build_report(summary), encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "panel": summary["panel"],
        "decision": result_interpretation["decision"],
        "allowed_claim": result_interpretation["allowed_claim"],
        "primary_differences": {
            metric: {"mean": value["mean_difference"], "ci95": value["ci95"]}
            for metric, value in differences.items()
            if metric in {
                "unique_fraction", "mean_position_entropy_nats",
                "mean_pairwise_hamming_fraction", "developability_pass_fraction_at_0_55",
            }
        },
    }, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
