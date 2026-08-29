#!/usr/bin/env python
"""Freeze binder-confidence features and fail closed without experimental labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def optional_float(value):
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite measurement: {value}")
    return number


def optional_bool(value):
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass"}:
        return True
    if normalized in {"false", "0", "no", "fail"}:
        return False
    raise ValueError(f"Invalid boolean measurement: {value}")


def validate_inputs(config):
    loaded = {}
    for name, source in config["inputs"].items():
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
        loaded[name] = path
    return loaded


def candidate_maps(paths):
    prospective = json.loads(paths["prospective_results"].read_text(encoding="utf-8"))
    af2 = json.loads(paths["af2_analysis"].read_text(encoding="utf-8"))
    expression = json.loads(paths["expression_package"].read_text(encoding="utf-8"))
    prospective_by_sequence = {
        row["sequence"]: row for row in prospective["candidates"]
    }
    af2_by_id = {
        row["entity_id"]: row for row in af2["shortlist"]
    }
    expression_by_id = {
        row["construct_id"]: row for row in expression["constructs"]
        if row["role"] == "candidate"
    }
    return prospective_by_sequence, af2_by_id, expression_by_id


def build_feature_rows(config, paths):
    prospective, af2, expression = candidate_maps(paths)
    rows = []
    for construct_id in sorted(expression, key=lambda value: expression[value]["computational_rank"]):
        construct = expression[construct_id]
        af2_row = af2[construct_id]
        sequence = construct["h3_sequence"]
        prospective_row = prospective[sequence]
        scoring = prospective_row["scoring"]
        developability = prospective_row["developability"]
        row = {
            "construct_id": construct_id,
            "target_id": config["panel"]["target_id"],
            "scaffold_id": config["panel"]["scaffold_id"],
            "h3_sequence": sequence,
            "h3_sequence_sha256": sequence_sha256(sequence),
            "computational_rank": construct["computational_rank"],
            "state_mean_complex_minus_stripped": scoring["mean_complex_minus_stripped"],
            "state_std_complex_minus_stripped": scoring["std_complex_minus_stripped"],
            "state_worst_complex_minus_stripped": scoring["worst_complex_minus_stripped"],
            "state_poses_better_than_native": scoring["poses_better_than_native"],
            "state_mean_contact_probability": scoring["mean_contact_probability"],
            "mutation_count": prospective_row["mutation_count"],
            "sequence_developability_risk": developability["sequence_risk"],
            "sequence_hydrophobic_fraction": developability["hydrophobic_fraction"],
            "sequence_net_charge": developability["net_charge"],
            "af2_median_iptm": af2_row["median_iptm"],
            "af2_worst_iptm": af2_row["worst_iptm"],
            "af2_iptm_seed_range": af2_row["iptm_seed_range"],
            "af2_median_interface_pae": af2_row["median_interface_pae"],
            "prodigy_median_delta_g_kcal_mol": af2_row[
                "median_prodigy_delta_g_kcal_mol"],
        }
        if set(config["features"]["forbidden_as_primary_features"]) & set(row):
            raise ValueError("Forbidden legacy confidence feature entered the table")
        rows.append(row)
    if len(rows) != int(config["panel"]["candidate_count"]):
        raise ValueError("Feature row count differs from the frozen panel")
    return rows


def write_feature_artifacts(config, config_path, paths, rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / config["outputs"]["feature_table"]
    fields = list(rows[0])
    with feature_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "status": "pre_experiment_features_frozen",
        "config": str(config_path.relative_to(ROOT).as_posix()),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "feature_table": str(feature_path.relative_to(ROOT).as_posix()),
        "feature_table_sha256": sha256(feature_path),
        "rows": len(rows),
        "targets": sorted({row["target_id"] for row in rows}),
        "scaffolds": sorted({row["scaffold_id"] for row in rows}),
        "feature_columns": config["features"]["columns"],
        "forbidden_primary_features": config["features"]["forbidden_as_primary_features"],
        "source_hashes": {
            name: sha256(path) for name, path in paths.items()
        },
        "label_columns_present": False,
        "probability_output_permitted": False,
        "claim_boundary": config["claim_boundary"],
    }
    manifest_path = output_dir / config["outputs"]["feature_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return feature_path, manifest_path, manifest


def expression_labels(config, rows):
    required_batches = int(config["labels"]["expression_pass"]["required_batches"])
    grouped = defaultdict(list)
    for row in rows:
        construct_id = row.get("construct_id", "").strip()
        if construct_id:
            grouped[construct_id].append(row)
    labels = {}
    audits = {}
    for construct_id, records in grouped.items():
        valid = []
        for row in records:
            batch = row.get("expression_batch", "").strip()
            yield_value = optional_float(row.get("yield_mg_per_l"))
            monomer = optional_float(row.get("sec_monomer_fraction"))
            qc = optional_bool(row.get("qc_pass"))
            if batch and yield_value is not None and monomer is not None and qc is not None:
                valid.append((batch, yield_value, monomer, qc))
        distinct_batches = {record[0] for record in valid}
        complete = len(distinct_batches) >= required_batches
        passed = complete and all(
            record[1] >= float(config["labels"]["expression_pass"]["minimum_yield_mg_per_l"])
            and record[2] >= float(config["labels"]["expression_pass"]["minimum_sec_monomer_fraction"])
            and record[3]
            for record in valid
        )
        labels[construct_id] = passed if complete else None
        audits[construct_id] = {
            "valid_batches": len(distinct_batches),
            "required_batches": required_batches,
            "complete": complete,
        }
    return labels, audits


def bli_aggregates(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("operator_exclusion_reason", "").strip():
            continue
        blind_id = row.get("blind_sample_id", "").strip()
        analyte = row.get("analyte", "").strip()
        response = optional_float(row.get("blank_subtracted_response"))
        fit_quality = row.get("fit_quality", "").strip()
        if blind_id and analyte and response is not None and fit_quality:
            grouped[(blind_id, analyte)].append(response)
    output = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        cv = float(array.std(ddof=1) / abs(mean)) if len(array) > 1 and mean != 0 else None
        output[key] = {"n": len(values), "mean": mean, "cv": cv}
    return output


def assay_and_candidate_labels(
    config, aggregates, blind_key, candidate_ids, expression_label_by_construct
):
    construct_by_blind = {row["blind_sample_id"]: row["construct_id"] for row in blind_key}
    blind_by_construct = {value: key for key, value in construct_by_blind.items()}
    target = config["labels"]["binding_screen_hit"]["target_analyte"]
    controls = config["labels"]["specificity_pass"]["control_analytes"]
    min_reps = int(config["labels"]["binding_screen_hit"]["minimum_technical_replicates"])
    max_cv = float(config["labels"]["binding_screen_hit"]["maximum_replicate_cv"])
    external_id = config["assay_validity"]["external_negative_blind_id"]
    mock_id = config["assay_validity"]["mock_blind_id"]

    def complete(blind_id, analyte):
        row = aggregates.get((blind_id, analyte))
        return bool(row and row["n"] >= min_reps and row["cv"] is not None and row["cv"] <= max_cv)

    external_complete = all(complete(external_id, analyte) for analyte in [target, *controls])
    mock_complete = all(complete(mock_id, analyte) for analyte in [target, *controls])
    native_blind = blind_by_construct.get(config["assay_validity"]["native_control_id"])
    native_complete = bool(native_blind and complete(native_blind, target))
    external_target = aggregates.get((external_id, target), {}).get("mean")
    native_target = aggregates.get((native_blind, target), {}).get("mean") if native_blind else None
    native_signal_pass = bool(
        native_complete and external_target is not None and native_target is not None
        and native_target / max(abs(external_target), 1e-12) >= float(
            config["labels"]["binding_screen_hit"][
                "minimum_signal_ratio_vs_external_negative"])
    )
    native_expression_pass = expression_label_by_construct.get(
        config["assay_validity"]["native_control_id"]
    ) is True
    assay_valid = (
        external_complete and mock_complete and native_signal_pass and native_expression_pass
    )
    labels = {}
    audits = {}
    for construct_id, blind_id in blind_by_construct.items():
        if construct_id not in candidate_ids:
            continue
        target_row = aggregates.get((blind_id, target))
        control_rows = [aggregates.get((blind_id, analyte)) for analyte in controls]
        complete_candidate = complete(blind_id, target) and all(
            complete(blind_id, analyte) for analyte in controls
        )
        if not assay_valid or not complete_candidate or external_target is None:
            labels[construct_id] = {"binding_screen_hit": None, "specificity_pass": None}
        else:
            denominator = max(abs(external_target), 1e-12)
            signal_ratio = target_row["mean"] / denominator
            binding = signal_ratio >= float(
                config["labels"]["binding_screen_hit"]["minimum_signal_ratio_vs_external_negative"])
            target_denominator = max(abs(target_row["mean"]), 1e-12)
            specificity = all(
                abs(row["mean"]) / target_denominator <= float(
                    config["labels"]["specificity_pass"]["maximum_control_fraction_of_target"])
                for row in control_rows
            )
            labels[construct_id] = {
                "binding_screen_hit": binding,
                "specificity_pass": specificity,
            }
        audits[construct_id] = {
            "blind_sample_id": blind_id,
            "assay_valid": assay_valid,
            "candidate_measurements_complete": complete_candidate,
        }
    assay = {
        "valid": assay_valid,
        "native_target_complete": native_complete,
        "native_target_signal_pass": native_signal_pass,
        "native_expression_pass": native_expression_pass,
        "external_negative_complete": external_complete,
        "mock_complete": mock_complete,
    }
    return labels, audits, assay


def training_readiness(config, labeled_rows):
    qc_rows = [row for row in labeled_rows if row.get("expression_pass") is not None]
    positives = sum(row.get("binding_screen_hit") is True for row in labeled_rows)
    scaffolds = {row["scaffold_id"] for row in labeled_rows}
    targets = {row["target_id"] for row in labeled_rows}
    gates = config["training_gates"]
    checks = {
        "minimum_qc_candidates": len(qc_rows) >= int(gates["minimum_qc_candidates"]),
        "minimum_positive_binding_hits": positives >= int(gates["minimum_positive_binding_hits"]),
        "minimum_scaffolds": len(scaffolds) >= int(gates["minimum_scaffolds"]),
        "minimum_targets": len(targets) >= int(gates["minimum_targets"]),
    }
    return {
        "eligible": all(checks.values()) and not gates["panel_only_model_forbidden"],
        "checks": checks,
        "observed_qc_candidates": len(qc_rows),
        "observed_positive_binding_hits": positives,
        "observed_scaffolds": len(scaffolds),
        "observed_targets": len(targets),
        "decision": "train_regularized_model" if all(checks.values()) else "abstain_insufficient_data",
    }


def analyze_measurements(config, feature_rows, expression_path, bli_path, blind_key_path):
    expression_rows = read_csv(expression_path)
    bli_rows = read_csv(bli_path)
    blind_key = read_csv(blind_key_path)
    expression, expression_audit = expression_labels(config, expression_rows)
    aggregates = bli_aggregates(bli_rows)
    candidate_ids = {row["construct_id"] for row in feature_rows}
    assay_labels, assay_audit, assay = assay_and_candidate_labels(
        config, aggregates, blind_key, candidate_ids, expression
    )
    labeled = []
    for feature in feature_rows:
        construct_id = feature["construct_id"]
        expression_pass = expression.get(construct_id)
        binding = assay_labels.get(construct_id, {}).get("binding_screen_hit")
        specificity = assay_labels.get(construct_id, {}).get("specificity_pass")
        overall = (
            expression_pass and binding and specificity
            if None not in (expression_pass, binding, specificity) else None
        )
        labeled.append({
            **feature,
            "expression_pass": expression_pass,
            "binding_screen_hit": binding,
            "specificity_pass": specificity,
            "overall_screen_pass": overall,
            "affinity_kd_molar": None,
            "affinity_label_status": "abstain_requires_multiconcentration_bli_or_spr",
        })
    complete_labels = sum(row["overall_screen_pass"] is not None for row in labeled)
    status = (
        "panel_labels_complete_model_training_still_gated"
        if complete_labels == len(labeled) and assay["valid"]
        else "abstain_no_experimental_labels"
    )
    return {
        "status": status,
        "assay_validity": assay,
        "complete_candidate_labels": complete_labels,
        "expression_audit": expression_audit,
        "assay_audit": assay_audit,
        "labeled_rows": labeled,
        "training_readiness": training_readiness(config, labeled),
    }


def write_labeled_table(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/binder_confidence_v1.yml")
    parser.add_argument("--expression-measurements")
    parser.add_argument("--bli-measurements")
    parser.add_argument("--blind-key")
    parser.add_argument("--out")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = validate_inputs(config)
    feature_rows = build_feature_rows(config, paths)
    output_dir = ROOT / (args.out or config["outputs"]["directory"])
    feature_path, manifest_path, manifest = write_feature_artifacts(
        config, config_path, paths, feature_rows, output_dir
    )
    expression_path = ROOT / (args.expression_measurements or
        "results/prospective/abeta_4hix_expression_prep_v1/expression_batch_tracking.csv")
    bli_path = ROOT / (args.bli_measurements or
        "results/prospective/abeta_4hix_expression_prep_v1/bli_results_template.csv")
    blind_key_path = ROOT / (args.blind_key or config["inputs"]["blinding_key"]["path"])
    analysis = analyze_measurements(
        config, feature_rows, expression_path, bli_path, blind_key_path
    )
    labeled_path = output_dir / config["outputs"]["labeled_table"]
    write_labeled_table(labeled_path, analysis.pop("labeled_rows"))
    status = {
        "schema_version": 1,
        **analysis,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "feature_manifest": str(manifest_path.relative_to(ROOT).as_posix()),
        "feature_manifest_sha256": sha256(manifest_path),
        "feature_table_sha256": sha256(feature_path),
        "labeled_table_sha256": sha256(labeled_path),
        "measurement_inputs": {
            "expression": str(expression_path.relative_to(ROOT).as_posix()),
            "expression_sha256": sha256(expression_path),
            "bli": str(bli_path.relative_to(ROOT).as_posix()),
            "bli_sha256": sha256(bli_path),
            "blind_key_sha256": sha256(blind_key_path),
        },
        "probability_output": None,
        "probability_status": "abstain_not_trained",
        "claim_boundary": config["claim_boundary"],
    }
    status_path = output_dir / config["outputs"]["measurement_status"]
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="ascii")
    panel_path = output_dir / config["outputs"]["panel_analysis"]
    panel_path.write_text(json.dumps({
        "schema_version": 1,
        "status": status["status"],
        "training_readiness": status["training_readiness"],
        "probability_status": status["probability_status"],
        "note": "Computational scores are frozen features, never experimental labels.",
    }, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": status["status"],
        "features": len(feature_rows),
        "complete_candidate_labels": status["complete_candidate_labels"],
        "training_eligible": status["training_readiness"]["eligible"],
        "probability_status": status["probability_status"],
    }, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
