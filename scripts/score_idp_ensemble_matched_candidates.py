#!/usr/bin/env python
"""Score matched-generator H3 pools by leave-one-conformer-out selection."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.h3_interface_contacts import extract_h3_interface, score_h3_sequence  # noqa: E402


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


def candidate_pools(raw_generation, methods):
    grouped = defaultdict(list)
    for slot in raw_generation["slots"]:
        if slot["status"] != "success" or slot["method"] not in methods:
            continue
        grouped[(slot["component_id"], slot["method"])].extend(slot["candidates"])
    pools = {}
    for key, rows in grouped.items():
        sequences = []
        seen = set()
        for row in rows:
            sequence = row["sequence"]
            if sequence not in seen:
                seen.add(sequence)
                sequences.append(sequence)
        pools[key] = sequences
    return pools


def pose_rows_by_component(components, abeta_audit, nonabeta_audit):
    output = {}
    for component in components:
        if component["target"] == "amyloid_beta":
            rows = [
                row for row in abeta_audit["entries"]
                if row["reference_pdb"] == component["reference_pdb"] and row["status"] == "pass"
            ]
        else:
            rows = [
                row for row in nonabeta_audit["entries"]
                if row["reference_id"] == component["reference_pdb"]
                and row["status"] == "pass"
                and row.get("selected_for_panel")
            ]
        output[component["component_id"]] = sorted(rows, key=lambda row: str(row["conformer"]))
    return output


def score_matrix(component, pose_rows, sequences, scoring):
    interfaces = [
        extract_h3_interface(
            str(ROOT / row["pose_pdb"]),
            component["heavy_chain"],
            "P",
            light_chain=component["light_chain"],
            expected_h3_sequence=component["native_h3"],
            contact_cutoff=float(scoring["contact_cutoff_angstrom"]),
            sidechain_cutoff=float(scoring["sidechain_cutoff_angstrom"]),
        )
        for row in pose_rows
    ]
    matrix = np.asarray([
        [score_h3_sequence(sequence, interface)["chemistry_score"] for interface in interfaces]
        for sequence in sequences
    ], dtype=np.float64)
    native = np.asarray([
        score_h3_sequence(component["native_h3"], interface)["chemistry_score"]
        for interface in interfaces
    ], dtype=np.float64)
    return matrix, native, [
        {
            "conformer": row["conformer"],
            "pose_pdb": row["pose_pdb"],
            "pose_sha256": sha256(ROOT / row["pose_pdb"]),
            "h3_geometry_contacts": len(interface["contacts"]),
            "h3_sidechain_contacts": sum(
                contact.min_sidechain_distance is not None for contact in interface["contacts"]
            ),
        }
        for row, interface in zip(pose_rows, interfaces, strict=True)
    ]


def leave_one_out(component_id, method, sequences, matrix, native, aggregation):
    rows = []
    for holdout in range(matrix.shape[1]):
        training = [index for index in range(matrix.shape[1]) if index != holdout]
        ensemble_index = max(
            range(len(sequences)),
            key=lambda index: (robust_value(matrix[index, training], aggregation), sequences[index]),
        )
        single_indices = [int(np.argmax(matrix[:, conformer])) for conformer in training]
        ensemble_score = float(matrix[ensemble_index, holdout])
        single_scores = [float(matrix[index, holdout]) for index in single_indices]
        single_score = float(np.median(single_scores))
        native_score = float(native[holdout])
        rows.append({
            "component_id": component_id,
            "method": method,
            "heldout_conformer_index": holdout,
            "candidate_pool_size": len(sequences),
            "ensemble_candidate": sequences[ensemble_index],
            "ensemble_score": ensemble_score,
            "single_state_score": single_score,
            "native_score": native_score,
            "ensemble_minus_single_state": ensemble_score - single_score,
            "ensemble_minus_native": ensemble_score - native_score,
        })
    return rows


def summarize(rows, methods, components, statistics, gates):
    by_component_method = []
    effects = {}
    for method in methods:
        for component in components:
            selected = [
                row for row in rows
                if row["method"] == method and row["component_id"] == component
            ]
            if not selected:
                continue
            delta = float(np.mean([row["ensemble_minus_single_state"] for row in selected]))
            native_delta = float(np.mean([row["ensemble_minus_native"] for row in selected]))
            effects[(component, method)] = delta
            by_component_method.append({
                "component_id": component,
                "method": method,
                "heldout_conformers": len(selected),
                "mean_ensemble_minus_single_state": delta,
                "mean_ensemble_minus_native": native_delta,
            })
    by_method = {}
    for method_index, method in enumerate(methods):
        values = [effects[(component, method)] for component in components if (component, method) in effects]
        ci = bootstrap_mean(
            values,
            int(statistics["bootstrap_trials"]),
            int(statistics["bootstrap_seed"]) + method_index,
        )
        gate_results = {
            "minimum_valid_components": len(values) >= int(gates["minimum_valid_components_per_generator"]),
            "minimum_positive_component_fraction": (
                sum(value > 0 for value in values) / len(values) if values else 0.0
            ) >= float(gates["minimum_positive_component_fraction"]),
            "bootstrap_ci95_lower_above_zero": bool(ci and ci[0] > 0),
        }
        by_method[method] = {
            "valid_components": len(values),
            "mean_ensemble_minus_single_state": float(np.mean(values)) if values else None,
            "median_ensemble_minus_single_state": float(np.median(values)) if values else None,
            "positive_component_fraction": (
                sum(value > 0 for value in values) / len(values) if values else 0.0
            ),
            "component_bootstrap_ci95": ci,
            "exact_sign_flip_p": exact_sign_flip_p(values),
            "gate_results": gate_results,
            "passed": all(gate_results.values()),
        }
    return by_component_method, by_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_matched_scoring_dev_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = {
        "component_manifest": config["component_manifest"],
        "raw_generation": config["raw_generation"],
        "abeta_pose_audit": config["pose_audits"]["abeta"],
        "nonabeta_pose_audit": config["pose_audits"]["nonabeta"],
    }
    paths = {name: ROOT / item["path"] for name, item in frozen.items()}
    for name, path in paths.items():
        if sha256(path) != frozen[name]["sha256"]:
            raise ValueError(f"Frozen hash mismatch: {path}")

    manifest = json.loads(paths["component_manifest"].read_text(encoding="utf-8"))
    raw = json.loads(paths["raw_generation"].read_text(encoding="utf-8"))
    if raw["status"] != "complete" or raw["failures"]:
        raise ValueError("Matched generation must be complete with zero failures")
    components = manifest["components"]
    methods = config["candidate_pool"]["methods"]
    pools = candidate_pools(raw, methods)
    poses = pose_rows_by_component(
        components,
        json.loads(paths["abeta_pose_audit"].read_text(encoding="utf-8")),
        json.loads(paths["nonabeta_pose_audit"].read_text(encoding="utf-8")),
    )

    rows = []
    matrices = []
    invalid_component_methods = []
    expected_conformers = int(config["scoring"]["conformers_per_component"])
    for component in components:
        component_id = component["component_id"]
        if len(poses[component_id]) != expected_conformers:
            raise ValueError(f"{component_id} does not have {expected_conformers} held-out poses")
        for method in methods:
            sequences = pools.get((component_id, method), [])
            if not sequences:
                raise ValueError(f"Missing candidate pool: {component_id} {method}")
            matrix, native, pose_provenance = score_matrix(
                component, poses[component_id], sequences, config["scoring"]
            )
            minimum_sidechain_contacts = int(
                config["scoring"]["minimum_sidechain_contacts_per_conformer"]
            )
            invalid_poses = [
                row for row in pose_provenance
                if row["h3_sidechain_contacts"] < minimum_sidechain_contacts
            ]
            if invalid_poses:
                invalid_component_methods.append({
                    "component_id": component_id,
                    "method": method,
                    "reason": "insufficient_h3_sidechain_contacts",
                    "invalid_conformers": [row["conformer"] for row in invalid_poses],
                })
                continue
            rows.extend(leave_one_out(
                component_id,
                method,
                sequences,
                matrix,
                native,
                config["scoring"]["ensemble_aggregation"],
            ))
            matrices.append({
                "component_id": component_id,
                "method": method,
                "sequences": sequences,
                "scores": matrix.tolist(),
                "native_scores": native.tolist(),
                "poses": pose_provenance,
            })

    component_ids = [component["component_id"] for component in components]
    component_summary, method_summary = summarize(
        rows, methods, component_ids, config["statistics"], config["development_gates"]
    )
    output = {
        "schema_version": 1,
        "status": (
            "all_generator_development_gates_passed"
            if all(row["passed"] for row in method_summary.values())
            else "development_gate_failed"
        ),
        "classification": config["classification"],
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "inputs": {name: sha256(path) for name, path in paths.items()},
        },
        "method_summary": method_summary,
        "component_method_summary": component_summary,
        "invalid_component_methods": invalid_component_methods,
        "leave_one_conformer_out": rows,
        "score_matrices": matrices,
        "claim_boundary": config["claim_boundary"],
    }
    out_path = ROOT / (args.out or config["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": output["status"], "method_summary": method_summary}, indent=2))


if __name__ == "__main__":
    main()
