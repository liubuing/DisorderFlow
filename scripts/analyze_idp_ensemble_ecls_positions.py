#!/usr/bin/env python
"""Decompose negative ECLS folds into apo, complex, and H3-position terms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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


def residue_profile(log_probs, sequence, indices, alphabet):
    length = len(sequence)
    apo_or_complex = [
        -float(log_probs[index, alphabet.index(aa)]) / length
        for index, aa in zip(indices, sequence, strict=True)
    ]
    return np.asarray(apo_or_complex, dtype=np.float64)


def candidate_state_profiles(apo_logp, complex_logp, sequence, indices, alphabet):
    apo = residue_profile(apo_logp, sequence, indices, alphabet)
    complex_state = residue_profile(complex_logp, sequence, indices, alphabet)
    return {"apo": apo, "complex": complex_state, "gain": apo - complex_state}


def contribution_mechanism(apo_difference, complex_difference, contribution):
    if contribution >= 0:
        return "positive_or_neutral"
    apo_term = apo_difference
    complex_term = -complex_difference
    if apo_term < 0 <= complex_term:
        return "apo_baseline_dominance"
    if complex_term < 0 <= apo_term:
        return "complex_support_deficit"
    if apo_term < 0 and complex_term < 0:
        return (
            "mixed_apo_dominant" if abs(apo_term) >= abs(complex_term)
            else "mixed_complex_dominant"
        )
    return "offsetting_terms"


def middle_comparator_records(records):
    ordered = sorted(records, key=lambda row: (row["heldout_score"], row["training_conformer"]))
    selected = ordered[1:3]
    selected_ids = {(row["training_conformer"], row["candidate_index"]) for row in selected}
    boundary_scores = {row["heldout_score"] for row in selected}
    alternatives = [
        row for row in ordered
        if row["heldout_score"] in boundary_scores
        and (row["training_conformer"], row["candidate_index"]) not in selected_ids
        and row["candidate_index"] not in {item["candidate_index"] for item in selected}
    ]
    return selected, ordered, bool(alternatives), alternatives


def load_state_npz(path, expected_hash, indices, native_h3, alphabet, tolerances):
    if sha256(path) != expected_hash:
        raise ValueError(f"Conditional NPZ hash mismatch: {path}")
    payload = np.load(path)
    required = {"log_p", "S", "mask", "design_mask"}
    if not required <= set(payload.files):
        raise ValueError(f"Conditional NPZ lacks required arrays: {path}")
    log_p = np.asarray(payload["log_p"][0], dtype=np.float64)
    if not np.isfinite(log_p).all():
        raise ValueError(f"Non-finite conditional probabilities: {path}")
    normalized = np.exp(log_p[indices]).sum(axis=-1)
    if not np.allclose(
        normalized, 1.0, atol=float(tolerances["probability_normalization_tolerance"])
    ):
        raise ValueError(f"Conditional probabilities are not normalized: {path}")
    for index in indices:
        if payload["mask"][index] <= 0 or payload["design_mask"][index] <= 0:
            raise ValueError(f"Invalid H3 mask at index {index}: {path}")
    decoded = "".join(alphabet[int(payload["S"][index])] for index in indices)
    if decoded != native_h3:
        raise ValueError(f"Native H3 mapping mismatch in {path}: {decoded}")
    return log_p


def select_ensemble(matrix, sequences, training, aggregation):
    return max(
        range(len(sequences)),
        key=lambda index: (robust_value(matrix[index, training], aggregation), sequences[index]),
    )


def analyze_component(spec, config, analysis):
    cache_path = ROOT / spec["score_cache"]
    if sha256(cache_path) != spec["score_cache_sha256"]:
        raise ValueError(f"Score cache hash mismatch: {cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    component = cache["component"]
    scores = cache["scores"]
    if component["component_id"] != spec["component_id"]:
        raise ValueError("Score cache component mismatch")
    start, end = component["h3_positions_1_indexed"]
    indices = list(range(start - 1, end))
    native_h3 = component["native_h3"]
    alphabet = config["proteinmpnn_alphabet"]
    method_row = next(row for row in scores["methods"] if row["method"] == config["method"])
    sequences = method_row["sequences"]
    stored_matrix = np.asarray(method_row["epitope_conditioning_gain"], dtype=np.float64)
    state_profiles = []
    npz_provenance = []
    for conformer_index, conformer in enumerate(scores["conformers"]):
        state = {}
        provenance = {"conformer_index": conformer_index, "conformer": conformer["conformer"]}
        for state_name in ("apo", "complex"):
            record = conformer[state_name]
            path = ROOT / record["npz"]
            state[state_name] = load_state_npz(
                path,
                record["npz_sha256"],
                indices,
                native_h3,
                alphabet,
                config["thresholds"],
            )
            provenance[state_name] = {
                "npz": record["npz"],
                "sha256": record["npz_sha256"],
            }
        state_profiles.append(state)
        npz_provenance.append(provenance)

    rebuilt = np.zeros_like(stored_matrix)
    all_profiles = []
    for candidate_index, sequence in enumerate(sequences):
        candidate_profiles = []
        for conformer_index, state in enumerate(state_profiles):
            profile = candidate_state_profiles(
                state["apo"], state["complex"], sequence, indices, alphabet
            )
            rebuilt[candidate_index, conformer_index] = float(profile["gain"].sum())
            candidate_profiles.append(profile)
        all_profiles.append(candidate_profiles)
    residual = float(np.max(np.abs(rebuilt - stored_matrix)))
    if residual > float(config["thresholds"]["reconstruction_tolerance"]):
        raise ValueError(f"Candidate matrix reconstruction residual {residual}")

    stored_loo = {
        int(row["heldout_conformer_index"]): row
        for row in analysis["leave_one_conformer_out"]
        if row["component_id"] == spec["component_id"] and row["method"] == config["method"]
    }
    folds = []
    position_rows = []
    for holdout in range(stored_matrix.shape[1]):
        training = [index for index in range(stored_matrix.shape[1]) if index != holdout]
        ensemble_index = select_ensemble(
            stored_matrix, sequences, training, config["aggregation"]
        )
        single_indices = [int(np.argmax(stored_matrix[:, index])) for index in training]
        comparator_records = [
            {
                "training_conformer": training_index,
                "candidate_index": candidate_index,
                "candidate": sequences[candidate_index],
                "heldout_score": float(stored_matrix[candidate_index, holdout]),
            }
            for training_index, candidate_index in zip(training, single_indices, strict=True)
        ]
        middle, ordered, nonunique, alternatives = middle_comparator_records(comparator_records)
        comparator_score = float(np.mean([row["heldout_score"] for row in middle]))
        ensemble_score = float(stored_matrix[ensemble_index, holdout])
        effect = ensemble_score - comparator_score
        stored = stored_loo[holdout]
        tolerance = float(config["thresholds"]["reconstruction_tolerance"])
        if sequences[ensemble_index] != stored["ensemble_candidate"]:
            raise ValueError("Stored ensemble candidate reproduction failed")
        if abs(effect - float(stored["ensemble_minus_single_state"])) > tolerance:
            raise ValueError("Stored fold effect reproduction failed")
        fold = {
            "heldout_conformer_index": holdout,
            "heldout_conformer": scores["conformers"][holdout]["conformer"],
            "ensemble_candidate_index": ensemble_index,
            "ensemble_candidate": sequences[ensemble_index],
            "single_winner_observations": ordered,
            "middle_comparator_observations": middle,
            "positional_attribution_nonunique": nonunique,
            "attribution_alternatives": alternatives,
            "ensemble_score": ensemble_score,
            "single_state_score": comparator_score,
            "ensemble_minus_single_state": effect,
            "negative_transfer": effect < float(config["thresholds"]["negative_fold_effect"]),
        }
        if fold["negative_transfer"]:
            ensemble_profile = all_profiles[ensemble_index][holdout]
            comparator_profiles = [
                all_profiles[row["candidate_index"]][holdout] for row in middle
            ]
            comparator = {
                key: np.mean([profile[key] for profile in comparator_profiles], axis=0)
                for key in ("apo", "complex", "gain")
            }
            contributions = []
            for offset, tensor_index in enumerate(indices):
                apo_difference = float(ensemble_profile["apo"][offset] - comparator["apo"][offset])
                complex_difference = float(
                    ensemble_profile["complex"][offset] - comparator["complex"][offset]
                )
                contribution = apo_difference - complex_difference
                row = {
                    "panel": spec["panel"],
                    "component_id": spec["component_id"],
                    "method": config["method"],
                    "heldout_conformer_index": holdout,
                    "heldout_conformer": fold["heldout_conformer"],
                    "h3_offset_0_based": offset,
                    "h3_position_1_based": start + offset,
                    "tensor_index_0_based": tensor_index,
                    "ensemble_candidate": sequences[ensemble_index],
                    "ensemble_residue": sequences[ensemble_index][offset],
                    "middle_low_training_conformer": middle[0]["training_conformer"],
                    "middle_low_candidate": middle[0]["candidate"],
                    "middle_low_residue": middle[0]["candidate"][offset],
                    "middle_high_training_conformer": middle[1]["training_conformer"],
                    "middle_high_candidate": middle[1]["candidate"],
                    "middle_high_residue": middle[1]["candidate"][offset],
                    "ensemble_apo_nll": float(ensemble_profile["apo"][offset]),
                    "ensemble_complex_nll": float(ensemble_profile["complex"][offset]),
                    "ensemble_ecls_gain": float(ensemble_profile["gain"][offset]),
                    "comparator_apo_nll": float(comparator["apo"][offset]),
                    "comparator_complex_nll": float(comparator["complex"][offset]),
                    "comparator_ecls_gain": float(comparator["gain"][offset]),
                    "apo_difference": apo_difference,
                    "complex_difference": complex_difference,
                    "ensemble_minus_median_single_contribution": contribution,
                    "mechanism": contribution_mechanism(
                        apo_difference, complex_difference, contribution
                    ),
                }
                contributions.append(row)
                position_rows.append(row)
            contribution_sum = float(sum(
                row["ensemble_minus_median_single_contribution"] for row in contributions
            ))
            if abs(contribution_sum - effect) > tolerance:
                raise ValueError("Per-position fold contribution reconstruction failed")
            fold["position_contributions"] = contributions
            fold["position_contribution_sum"] = contribution_sum
            fold["reconstruction_residual"] = contribution_sum - effect
        folds.append(fold)

    negative_rows = [row for row in position_rows]
    by_position = []
    for offset in range(len(native_h3)):
        selected = [row for row in negative_rows if row["h3_offset_0_based"] == offset]
        by_position.append({
            "h3_offset_0_based": offset,
            "h3_position_1_based": start + offset,
            "native_residue": native_h3[offset],
            "negative_folds": len(selected),
            "mean_apo_difference": float(np.mean([row["apo_difference"] for row in selected])),
            "mean_complex_difference": float(np.mean([
                row["complex_difference"] for row in selected
            ])),
            "mean_effect_contribution": float(np.mean([
                row["ensemble_minus_median_single_contribution"] for row in selected
            ])),
            "sum_effect_contribution": float(np.sum([
                row["ensemble_minus_median_single_contribution"] for row in selected
            ])),
            "mechanism": contribution_mechanism(
                float(np.mean([row["apo_difference"] for row in selected])),
                float(np.mean([row["complex_difference"] for row in selected])),
                float(np.mean([
                    row["ensemble_minus_median_single_contribution"] for row in selected
                ])),
            ),
        })
    return {
        "component_id": spec["component_id"],
        "panel": spec["panel"],
        "h3_positions_1_indexed": [start, end],
        "native_h3": native_h3,
        "candidate_pool_size": len(sequences),
        "matrix_reconstruction_max_abs_residual": residual,
        "npz_provenance": npz_provenance,
        "negative_folds": [row["heldout_conformer_index"] for row in folds if row["negative_transfer"]],
        "folds": folds,
        "position_summary": sorted(by_position, key=lambda row: row["mean_effect_contribution"]),
    }, position_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, output):
    lines = [
        "# ECLS H3 Position Decomposition", "",
        "Per-position values are ProteinMPNN likelihood contributions, not binding energies.", "",
    ]
    for component in output["components"]:
        lines.extend([
            f"## {component['component_id']}", "",
            f"Negative folds: {component['negative_folds']}", "",
            "| H3 position | Native | Mean apo difference | Mean complex difference | Mean effect contribution | Mechanism |",
            "|---:|:---:|---:|---:|---:|---|",
        ])
        for row in component["position_summary"]:
            lines.append(
                f"| {row['h3_position_1_based']} | {row['native_residue']} | "
                f"{row['mean_apo_difference']:.8f} | "
                f"{row['mean_complex_difference']:.8f} | "
                f"{row['mean_effect_contribution']:.8f} | {row['mechanism']} |"
            )
        lines.append("")
    lines.extend(["## Endpoint v3 Priorities", ""])
    for priority in output["endpoint_v3_priorities"]:
        lines.append(f"{priority['priority']}. {priority['action']} {priority['reason']}")
    lines.append("")
    lines.extend([f"Claim boundary: {output['claim_boundary']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/idp_ensemble_ecls_h3_position_decomposition_v1.yml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    analyses = {}
    provenance = {}
    for name, item in config["analysis_inputs"].items():
        path = ROOT / item["path"]
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Analysis input hash mismatch: {path}")
        analyses[name] = json.loads(path.read_text(encoding="utf-8"))
        provenance[name] = {"path": item["path"], "sha256": digest}
    components = []
    all_rows = []
    for spec in config["components"]:
        component, rows = analyze_component(spec, config, analyses[spec["panel"]])
        components.append(component)
        all_rows.extend(rows)
    output = {
        "schema_version": 1,
        "status": "position_decomposition_complete",
        "classification": config["classification"],
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "analysis_inputs": provenance,
        },
        "components": components,
        "endpoint_v3_priorities": [
            {
                "priority": 1,
                "action": "Replace scalar gain-only ranking with a predeclared two-axis objective that retains absolute complex NLL and apo-to-complex gain separately.",
                "reason": "Apo-baseline dominance can penalize candidates that have better absolute complex likelihood.",
            },
            {
                "priority": 2,
                "action": "Require an absolute complex-compatibility constraint before optimizing state contrast.",
                "reason": "Prevents high gain caused primarily by poor apo likelihood from winning.",
            },
            {
                "priority": 3,
                "action": "Report per-position influence and test a predeclared contribution cap or distributed-support requirement on the exposed panel.",
                "reason": "TAU_6LRA negative transfer is concentrated at one H3 position.",
            },
        ],
        "claim_boundary": config["claim_boundary"],
    }
    json_path = ROOT / config["output"]["json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    write_csv(ROOT / config["output"]["csv"], all_rows)
    write_report(ROOT / config["output"]["report"], output)
    print(json.dumps({
        "status": output["status"],
        "components": [
            {
                "component_id": row["component_id"],
                "negative_folds": row["negative_folds"],
                "top_negative_positions": row["position_summary"][:3],
            }
            for row in components
        ],
        "per_position_rows": len(all_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
