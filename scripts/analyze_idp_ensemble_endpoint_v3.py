#!/usr/bin/env python
"""Evaluate native-constrained complex NLL plus ECLS gain as separate axes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gain_robust(values, aggregation):
    values = np.asarray(values, dtype=np.float64)
    return float(
        float(aggregation["p25_weight"]) * np.percentile(values, 25)
        + float(aggregation["minimum_weight"]) * values.min()
        + float(aggregation["mean_weight"]) * values.mean()
        - float(aggregation["standard_deviation_penalty"]) * values.std()
    )


def complex_robust_cost(values, aggregation):
    values = np.asarray(values, dtype=np.float64)
    return float(
        float(aggregation["p75_weight"]) * np.percentile(values, 75)
        + float(aggregation["maximum_weight"]) * values.max()
        + float(aggregation["mean_weight"]) * values.mean()
        + float(aggregation["standard_deviation_penalty"]) * values.std()
    )


def sequence_nll(log_probs, sequence, indices, alphabet):
    return -sum(
        float(log_probs[index, alphabet.index(aa)])
        for index, aa in zip(indices, sequence, strict=True)
    ) / len(sequence)


def load_conformer_states(conformer, native_h3, alphabet):
    states = {}
    indices = None
    for state_name in ("apo", "complex"):
        record = conformer[state_name]
        path = ROOT / record["npz"]
        if sha256(path) != record["npz_sha256"]:
            raise ValueError(f"Conditional NPZ hash mismatch: {path}")
        payload = np.load(path)
        state_indices = [
            int(index) for index in np.flatnonzero(np.asarray(payload["design_mask"]) > 0)
        ]
        if indices is None:
            indices = state_indices
        elif indices != state_indices:
            raise ValueError("Apo and complex H3 design masks differ")
        if len(indices) != len(native_h3):
            raise ValueError("H3 design mask length mismatch")
        decoded = "".join(alphabet[int(payload["S"][index])] for index in indices)
        if decoded != native_h3:
            raise ValueError("Conditional NPZ native H3 mismatch")
        log_p = np.asarray(payload["log_p"][0], dtype=np.float64)
        if not np.isfinite(log_p[indices]).all():
            raise ValueError("Non-finite H3 conditional probabilities")
        if not np.allclose(np.exp(log_p[indices]).sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError("H3 conditional probabilities are not normalized")
        states[state_name] = log_p
    return states, indices


def build_state_matrices(component, method_row, config):
    native = component["native_h3"]
    generated = list(method_row["sequences"])
    sequences = list(generated)
    if native not in sequences:
        sequences.append(native)
    native_index = sequences.index(native)
    conformer_states = []
    indices = None
    for conformer in component["conformers"]:
        states, conformer_indices = load_conformer_states(
            conformer, native, config["proteinmpnn_alphabet"]
        )
        if indices is None:
            indices = conformer_indices
        elif indices != conformer_indices:
            raise ValueError("Conformer H3 indices differ")
        conformer_states.append(states)
    apo = np.zeros((len(sequences), len(conformer_states)), dtype=np.float64)
    complex_nll = np.zeros_like(apo)
    for sequence_index, sequence in enumerate(sequences):
        if len(sequence) != len(native):
            raise ValueError("Candidate H3 length mismatch")
        for conformer_index, states in enumerate(conformer_states):
            apo[sequence_index, conformer_index] = sequence_nll(
                states["apo"], sequence, indices, config["proteinmpnn_alphabet"]
            )
            complex_nll[sequence_index, conformer_index] = sequence_nll(
                states["complex"], sequence, indices, config["proteinmpnn_alphabet"]
            )
    gain = apo - complex_nll
    stored_gain = np.asarray(method_row["epitope_conditioning_gain"], dtype=np.float64)
    generated_indices = [sequences.index(sequence) for sequence in generated]
    residual = float(np.max(np.abs(gain[generated_indices] - stored_gain)))
    if residual > 1e-12:
        raise ValueError(f"Stored gain matrix reconstruction residual {residual}")
    native_residual = float(np.max(np.abs(
        gain[native_index] - np.asarray(component["native_epitope_conditioning_gain"])
    )))
    if native_residual > 1e-12:
        raise ValueError(f"Native gain reconstruction residual {native_residual}")
    return {
        "sequences": sequences,
        "generated_sequences": set(generated),
        "native_index": native_index,
        "apo": apo,
        "complex": complex_nll,
        "gain": gain,
        "matrix_reconstruction_residual": residual,
        "native_reconstruction_residual": native_residual,
    }


def select_ensemble(matrices, training, config):
    sequences = matrices["sequences"]
    native_index = matrices["native_index"]
    tolerance = float(config["selection"]["constraint_tolerance"])
    costs = [
        complex_robust_cost(
            matrices["complex"][index, training],
            config["selection"]["complex_cost_aggregation"],
        )
        for index in range(len(sequences))
    ]
    mode = config["selection"]["complex_constraint"]
    if mode == "candidate_robust_complex_nll_not_above_native_robust_complex_nll":
        threshold = costs[native_index]
    elif mode == "top_fraction_by_robust_complex_nll":
        count = min(
            len(costs),
            max(
                int(config["selection"]["minimum_feasible_candidates"]),
                math.ceil(float(config["selection"]["complex_top_fraction"]) * len(costs)),
            ),
        )
        threshold = sorted(costs)[count - 1]
    else:
        raise ValueError(f"Unsupported complex constraint: {mode}")
    feasible = [index for index, cost in enumerate(costs) if cost <= threshold + tolerance]
    winner = max(
        feasible,
        key=lambda index: (
            gain_robust(
                matrices["gain"][index, training],
                config["selection"]["gain_aggregation"],
            ),
            sequences[index],
        ),
    )
    return winner, feasible, costs, threshold


def select_single(matrices, conformer, config):
    sequences = matrices["sequences"]
    native_index = matrices["native_index"]
    tolerance = float(config["selection"]["constraint_tolerance"])
    costs = [float(matrices["complex"][index, conformer]) for index in range(len(sequences))]
    mode = config["selection"]["complex_constraint"]
    if mode == "candidate_robust_complex_nll_not_above_native_robust_complex_nll":
        threshold = costs[native_index]
    elif mode == "top_fraction_by_robust_complex_nll":
        count = min(
            len(costs),
            max(
                int(config["selection"]["minimum_feasible_candidates"]),
                math.ceil(float(config["selection"]["complex_top_fraction"]) * len(costs)),
            ),
        )
        threshold = sorted(costs)[count - 1]
    else:
        raise ValueError(f"Unsupported complex constraint: {mode}")
    feasible = [index for index, cost in enumerate(costs) if cost <= threshold + tolerance]
    winner = max(
        feasible,
        key=lambda index: (matrices["gain"][index, conformer], sequences[index]),
    )
    return winner, feasible, threshold


def median_provenance(records, metric):
    ordered = sorted(
        records,
        key=lambda row: (row[metric], row["training_conformer"], row["candidate"]),
    )
    midpoint = len(ordered) // 2
    contributors = (
        ordered[midpoint:midpoint + 1]
        if len(ordered) % 2
        else ordered[midpoint - 1:midpoint + 1]
    )
    return float(np.median([row[metric] for row in records])), contributors


def analyze_component_method(component, method_row, config):
    matrices = build_state_matrices(component, method_row, config)
    folds = []
    for holdout in range(matrices["gain"].shape[1]):
        training = [index for index in range(matrices["gain"].shape[1]) if index != holdout]
        ensemble, feasible, costs, native_threshold = select_ensemble(
            matrices, training, config
        )
        single_records = []
        for conformer in training:
            winner, single_feasible, single_threshold = select_single(
                matrices, conformer, config
            )
            single_records.append({
                "training_conformer": conformer,
                "candidate_index": winner,
                "candidate": matrices["sequences"][winner],
                "selected_native": winner == matrices["native_index"],
                "feasible_pool_size": len(single_feasible),
                "native_complex_threshold": single_threshold,
                "gain": float(matrices["gain"][winner, holdout]),
                "complex_nll": float(matrices["complex"][winner, holdout]),
                "apo_nll": float(matrices["apo"][winner, holdout]),
            })
        gain_median, gain_middle = median_provenance(single_records, "gain")
        complex_median, complex_middle = median_provenance(single_records, "complex_nll")
        apo_median, apo_middle = median_provenance(single_records, "apo_nll")
        ensemble_gain = float(matrices["gain"][ensemble, holdout])
        ensemble_complex = float(matrices["complex"][ensemble, holdout])
        ensemble_apo = float(matrices["apo"][ensemble, holdout])
        native_index = matrices["native_index"]
        folds.append({
            "heldout_conformer_index": holdout,
            "ensemble_candidate_index": ensemble,
            "ensemble_candidate": matrices["sequences"][ensemble],
            "ensemble_selected_native": ensemble == native_index,
            "ensemble_feasible_pool_size": len(feasible),
            "ensemble_native_complex_threshold": native_threshold,
            "ensemble_selected_complex_cost": costs[ensemble],
            "single_observations": single_records,
            "gain_median_observations": gain_middle,
            "complex_median_observations": complex_middle,
            "apo_median_observations": apo_middle,
            "ensemble_gain": ensemble_gain,
            "single_median_gain": gain_median,
            "gain_advantage": ensemble_gain - gain_median,
            "ensemble_complex_nll": ensemble_complex,
            "single_median_complex_nll": complex_median,
            "complex_nll_advantage": complex_median - ensemble_complex,
            "ensemble_apo_nll": ensemble_apo,
            "single_median_apo_nll": apo_median,
            "apo_nll_advantage": apo_median - ensemble_apo,
            "ensemble_gain_minus_native": ensemble_gain
            - float(matrices["gain"][native_index, holdout]),
            "ensemble_complex_advantage_over_native": float(
                matrices["complex"][native_index, holdout]
            ) - ensemble_complex,
        })
    return {
        "component_id": component["component_id"],
        "method": method_row["method"],
        "conformer_count": len(component["conformers"]),
        "generated_pool_size": len(method_row["sequences"]),
        "v3_pool_size": len(matrices["sequences"]),
        "native_was_generated": component["native_h3"] in method_row["sequences"],
        "matrix_reconstruction_residual": matrices["matrix_reconstruction_residual"],
        "native_reconstruction_residual": matrices["native_reconstruction_residual"],
        "mean_gain_advantage": float(np.mean([row["gain_advantage"] for row in folds])),
        "mean_complex_nll_advantage": float(np.mean([
            row["complex_nll_advantage"] for row in folds
        ])),
        "mean_apo_nll_advantage": float(np.mean([
            row["apo_nll_advantage"] for row in folds
        ])),
        "ensemble_native_selection_fraction": sum(
            row["ensemble_selected_native"] for row in folds
        ) / len(folds),
        "single_native_selection_fraction": sum(
            observation["selected_native"]
            for row in folds for observation in row["single_observations"]
        ) / sum(len(row["single_observations"]) for row in folds),
        "joint_favorable_folds": sum(
            row["gain_advantage"] > 0 and row["complex_nll_advantage"] >= 0
            for row in folds
        ),
        "folds": folds,
    }


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


def target_from_component(component_id):
    if component_id.startswith("ABETA_"):
        return "amyloid_beta"
    if component_id.startswith("TAU_"):
        return "tau"
    if component_id.startswith("ASYN_"):
        return "alpha_synuclein"
    if component_id.startswith("PRION_"):
        return "prion_protein"
    return "other"


def grouped_summary(rows, field):
    output = {}
    for value in sorted({row[field] for row in rows}):
        selected = [row for row in rows if row[field] == value]
        output[value] = {
            "components": len(selected),
            "mean_gain_advantage": float(np.mean([
                row["mean_gain_advantage"] for row in selected
            ])),
            "mean_complex_nll_advantage": float(np.mean([
                row["mean_complex_nll_advantage"] for row in selected
            ])),
        }
    return output


def subset_summary(rows, config, seed_offset=0):
    epsilon = float(config["statistics"]["mathematical_effect_epsilon"])
    gain = [row["mean_gain_advantage"] for row in rows]
    complex_values = [row["mean_complex_nll_advantage"] for row in rows]
    margin = float(
        config["development_gates_for_future_freeze"]["complex_noninferiority_margin"]
    )
    seed = int(config["statistics"]["bootstrap_seed"]) + seed_offset
    return {
        "components": len(rows),
        "mean_gain_advantage": float(np.mean(gain)),
        "gain_positive_components": sum(value > epsilon for value in gain),
        "gain_negative_components": sum(value < -epsilon for value in gain),
        "gain_component_bootstrap_ci95": bootstrap_mean(
            gain, int(config["statistics"]["bootstrap_trials"]), seed
        ),
        "gain_exact_sign_flip_p": exact_sign_flip_p(gain),
        "mean_complex_nll_advantage": float(np.mean(complex_values)),
        "complex_noninferior_components": sum(
            value >= -margin for value in complex_values
        ),
        "complex_component_bootstrap_ci95": bootstrap_mean(
            complex_values,
            int(config["statistics"]["bootstrap_trials"]),
            seed + 100,
        ),
        "complex_exact_sign_flip_p": exact_sign_flip_p(complex_values),
        "by_target": grouped_summary(rows, "target"),
        "by_epitope_cluster": grouped_summary(rows, "epitope_cluster"),
    }


def cluster_summary(rows, config, seed_offset=0):
    by_cluster = grouped_summary(rows, "epitope_cluster")
    cluster_rows = list(by_cluster.values())
    gain = [row["mean_gain_advantage"] for row in cluster_rows]
    complex_values = [row["mean_complex_nll_advantage"] for row in cluster_rows]
    seed = int(config["statistics"]["bootstrap_seed"]) + seed_offset
    trials = int(config["statistics"]["bootstrap_trials"])
    return {
        "inference_unit": "epitope_overlap_cluster",
        "clusters": len(cluster_rows),
        "mean_of_cluster_mean_gain_advantage": float(np.mean(gain)),
        "gain_cluster_bootstrap_ci95": bootstrap_mean(gain, trials, seed),
        "gain_exact_cluster_sign_flip_p": exact_sign_flip_p(gain),
        "mean_of_cluster_mean_complex_nll_advantage": float(np.mean(complex_values)),
        "complex_cluster_bootstrap_ci95": bootstrap_mean(
            complex_values, trials, seed + 100
        ),
        "complex_exact_cluster_sign_flip_p": exact_sign_flip_p(complex_values),
        "by_epitope_cluster": by_cluster,
    }


def summarize_method(rows, config, method_index):
    epsilon = float(config["statistics"]["mathematical_effect_epsilon"])
    gain_values = [row["mean_gain_advantage"] for row in rows]
    complex_values = [row["mean_complex_nll_advantage"] for row in rows]
    gain_ci = bootstrap_mean(
        gain_values,
        int(config["statistics"]["bootstrap_trials"]),
        int(config["statistics"]["bootstrap_seed"]) + method_index,
    )
    complex_ci = bootstrap_mean(
        complex_values,
        int(config["statistics"]["bootstrap_trials"]),
        int(config["statistics"]["bootstrap_seed"]) + 100 + method_index,
    )
    gain_positive = sum(value > epsilon for value in gain_values)
    gain_negative = sum(value < -epsilon for value in gain_values)
    margin = float(
        config["development_gates_for_future_freeze"]["complex_noninferiority_margin"]
    )
    complex_noninferior = sum(value >= -margin for value in complex_values)
    gates = config["development_gates_for_future_freeze"]
    gate_results = {
        "minimum_valid_components": len(rows) >= int(gates["minimum_valid_components"]),
        "minimum_gain_positive_components": gain_positive
        >= int(gates["minimum_gain_positive_components"]),
        "gain_bootstrap_ci95_lower_above_zero": gain_ci[0] > 0,
        "maximum_gain_negative_components": gain_negative
        <= int(gates["maximum_gain_negative_components"]),
        "minimum_complex_noninferior_components": complex_noninferior
        >= int(gates["minimum_complex_noninferior_components"]),
        "complex_bootstrap_noninferior": complex_ci[0] >= -margin,
        "full_fold_coverage": all(
            len(row["folds"]) == row["conformer_count"] for row in rows
        ),
    }
    return {
        "valid_components": len(rows),
        "gain_axis": {
            "mean_advantage": float(np.mean(gain_values)),
            "median_advantage": float(np.median(gain_values)),
            "positive_components": gain_positive,
            "zero_components": len(rows) - gain_positive - gain_negative,
            "negative_components": gain_negative,
            "component_bootstrap_ci95": gain_ci,
            "exact_sign_flip_p": exact_sign_flip_p(gain_values),
        },
        "complex_axis": {
            "mean_nll_advantage": float(np.mean(complex_values)),
            "median_nll_advantage": float(np.median(complex_values)),
            "noninferior_components": complex_noninferior,
            "component_bootstrap_ci95": complex_ci,
            "exact_sign_flip_p": exact_sign_flip_p(complex_values),
        },
        "selection": {
            "ensemble_native_selection_fraction": float(np.mean([
                row["ensemble_native_selection_fraction"] for row in rows
            ])),
            "single_native_selection_fraction": float(np.mean([
                row["single_native_selection_fraction"] for row in rows
            ])),
            "joint_favorable_folds": sum(row["joint_favorable_folds"] for row in rows),
            "total_folds": sum(len(row["folds"]) for row in rows),
        },
        "by_target": grouped_summary(rows, "target"),
        "by_epitope_cluster": grouped_summary(rows, "epitope_cluster"),
        "gate_results": gate_results,
        "passed_for_future_freeze": all(gate_results.values()),
        "components": rows,
    }


def write_csv(path, summaries):
    fields = [
        "method", "component_id", "mean_gain_advantage",
        "mean_complex_nll_advantage", "mean_apo_nll_advantage",
        "ensemble_native_selection_fraction", "single_native_selection_fraction",
        "joint_favorable_folds", "generated_pool_size", "v3_pool_size",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, summary in summaries.items():
            for row in summary["components"]:
                writer.writerow({field: row.get(field, method if field == "method" else None)
                                 for field in fields})


def write_report(path, output):
    primary = output["method_summary"][output["primary_method"]]
    component_count = primary["valid_components"]
    clean = output["method_clean_subset_summary"][output["primary_method"]]
    clusters = output["method_cluster_summary"][output["primary_method"]]
    lines = [
        f"# IDP Ensemble {output['endpoint_id']} Development", "",
        "Gain and absolute complex NLL remain separate axes.", "",
        "## Primary ProteinMPNN", "",
        f"- Gain mean: {primary['gain_axis']['mean_advantage']}",
        f"- Gain positive components: {primary['gain_axis']['positive_components']}/{component_count}",
        f"- Gain CI95: {primary['gain_axis']['component_bootstrap_ci95']}",
        f"- Complex NLL mean advantage: {primary['complex_axis']['mean_nll_advantage']}",
        f"- Complex noninferior components: {primary['complex_axis']['noninferior_components']}/{component_count}",
        f"- Complex CI95: {primary['complex_axis']['component_bootstrap_ci95']}",
        f"- Native selection fraction: {primary['selection']['ensemble_native_selection_fraction']}",
        f"- Future-freeze gate: {primary['passed_for_future_freeze']}", "",
        "## Clean Primary Subset", "",
        f"- Components: {clean['components']}",
        f"- Gain mean: {clean['mean_gain_advantage']}",
        f"- Gain positive components: {clean['gain_positive_components']}/{clean['components']}",
        f"- Gain CI95: {clean['gain_component_bootstrap_ci95']}",
        f"- Complex NLL mean advantage: {clean['mean_complex_nll_advantage']}",
        f"- Complex noninferior components: {clean['complex_noninferior_components']}/{clean['components']}",
        f"- Complex CI95: {clean['complex_component_bootstrap_ci95']}", "",
        "## Epitope-Cluster Sensitivity", "",
        f"- Clusters: {clusters['clusters']}",
        f"- Gain mean of cluster means: {clusters['mean_of_cluster_mean_gain_advantage']}",
        f"- Gain cluster CI95: {clusters['gain_cluster_bootstrap_ci95']}",
        f"- Complex mean of cluster means: {clusters['mean_of_cluster_mean_complex_nll_advantage']}",
        f"- Complex cluster CI95: {clusters['complex_cluster_bootstrap_ci95']}", "",
        "| Component | Gain advantage | Complex NLL advantage | Native fraction | Joint folds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary["components"]:
        lines.append(
            f"| {row['component_id']} | {row['mean_gain_advantage']:.8f} | "
            f"{row['mean_complex_nll_advantage']:.8f} | "
            f"{row['ensemble_native_selection_fraction']:.2f} | "
            f"{row['joint_favorable_folds']}/{row['conformer_count']} |"
        )
    lines.extend(["", f"Claim boundary: {output['claim_boundary']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_endpoint_v3_development_v1.yml"
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
            raise ValueError(f"Frozen analysis hash mismatch: {path}")
        analyses.append(json.loads(path.read_text(encoding="utf-8")))
        provenance[name] = {"path": item["path"], "sha256": digest}
    component_metadata = {}
    registry_item = config.get("component_registry")
    if registry_item:
        registry_path = ROOT / registry_item["path"]
        registry_digest = sha256(registry_path)
        if registry_digest != registry_item["sha256"]:
            raise ValueError(f"Frozen registry hash mismatch: {registry_path}")
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        component_metadata = {
            row["component_id"]: row for row in registry["components"]
        }
        provenance["component_registry"] = {
            "path": registry_item["path"], "sha256": registry_digest
        }
    rows_by_method = defaultdict(list)
    for analysis in analyses:
        for component in analysis["component_scores"]:
            for method_row in component["methods"]:
                if method_row["method"] in config["methods"]:
                    row = analyze_component_method(component, method_row, config)
                    metadata = component_metadata.get(component["component_id"], {})
                    row.update({
                        "target": metadata.get(
                            "target", target_from_component(component["component_id"])
                        ),
                        "epitope_cluster": metadata.get(
                            "epitope_cluster", component["component_id"]
                        ),
                        "sensitivity_only": bool(metadata.get("sensitivity_only", False)),
                    })
                    rows_by_method[method_row["method"]].append(row)
    summaries = {
        method: summarize_method(rows_by_method[method], config, index)
        for index, method in enumerate(config["methods"])
    }
    clean_summaries = {
        method: subset_summary(
            [row for row in summaries[method]["components"] if not row["sensitivity_only"]],
            config,
            200 + index,
        )
        for index, method in enumerate(config["methods"])
    }
    cluster_summaries = {
        method: cluster_summary(summaries[method]["components"], config, 400 + index)
        for index, method in enumerate(config["methods"])
    }
    clean_cluster_summaries = {
        method: cluster_summary(
            [row for row in summaries[method]["components"] if not row["sensitivity_only"]],
            config,
            600 + index,
        )
        for index, method in enumerate(config["methods"])
    }
    primary = config["primary_method"]
    output = {
        "schema_version": 1,
        "endpoint_id": config["endpoint_id"],
        "status": f"{config['endpoint_id']}_candidate_for_untouched_freeze"
        if summaries[primary]["passed_for_future_freeze"]
        else f"{config['endpoint_id']}_development_gate_failed",
        "classification": config["classification"],
        "primary_method": primary,
        "selection_contract": config["selection"],
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "inputs": provenance,
        },
        "method_summary": summaries,
        "method_clean_subset_summary": clean_summaries,
        "method_cluster_summary": cluster_summaries,
        "method_clean_subset_cluster_summary": clean_cluster_summaries,
        "primary_sensitivity_excluding_synthetic_TAU_6PXR": subset_summary(
            [
                row for row in summaries[primary]["components"]
                if row["component_id"] != "TAU_6PXR"
            ],
            config,
            800,
        ),
        "claim_boundary": config["claim_boundary"],
    }
    json_path = ROOT / config["output"]["json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    write_csv(ROOT / config["output"]["csv"], summaries)
    write_report(ROOT / config["output"]["report"], output)
    summary = summaries[primary]
    print(json.dumps({
        "status": output["status"],
        "gain_axis": summary["gain_axis"],
        "complex_axis": summary["complex_axis"],
        "selection": summary["selection"],
        "gate_results": summary["gate_results"],
    }, indent=2))


if __name__ == "__main__":
    main()
