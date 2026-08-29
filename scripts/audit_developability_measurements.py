#!/usr/bin/env python
"""Create templates or audit measured candidate developability records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def candidate_template(candidate_path):
    source = json.loads(candidate_path.read_text(encoding="ascii"))
    rows = []
    seen = set()
    for component in source["components"]:
        sequence = component["candidate_sequence"]
        key = (component["component_id"], sequence)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "candidate_id": component["component_id"] + "|" + sequence_sha256(sequence)[:12],
            "component_id": component["component_id"],
            "sequence_sha256": sequence_sha256(sequence),
            "assay_batch": None,
            "replicates": None,
            "expression_mg_per_l": None,
            "sec_monomer_fraction": None,
            "aggregate_fraction": None,
            "polyspecificity_normalized": None,
            "melting_temperature_c": None,
        })
    return {
        "schema_version": 1,
        "status": "developability_measurement_template",
        "records": rows,
        "note": "Populate measured values; computational proxies are not accepted",
    }


def audit(config_path, measurements_path, output_path):
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    measurements = json.loads(measurements_path.read_text(encoding="ascii"))
    thresholds = config["required_measurements"]
    required_fields = (
        "replicates", "expression_mg_per_l", "sec_monomer_fraction",
        "aggregate_fraction", "polyspecificity_normalized",
    )
    records = []
    for row in measurements["records"]:
        missing = [field for field in required_fields if row.get(field) is None]
        checks = {
            "measurements_complete": not missing,
            "minimum_replicates": (
                row.get("replicates") is not None
                and int(row["replicates"]) >= int(thresholds["minimum_replicates"])),
            "expression": (
                row.get("expression_mg_per_l") is not None
                and float(row["expression_mg_per_l"])
                >= float(thresholds["minimum_expression_mg_per_l"])),
            "sec_monomer": (
                row.get("sec_monomer_fraction") is not None
                and float(row["sec_monomer_fraction"])
                >= float(thresholds["minimum_sec_monomer_fraction"])),
            "aggregation": (
                row.get("aggregate_fraction") is not None
                and float(row["aggregate_fraction"])
                <= float(thresholds["maximum_aggregate_fraction"])),
            "polyspecificity": (
                row.get("polyspecificity_normalized") is not None
                and float(row["polyspecificity_normalized"])
                <= float(thresholds["maximum_polyspecificity_normalized"])),
        }
        records.append({
            **row, "missing_measurements": missing,
            "checks": checks, "developability_pass": all(checks.values()),
        })
    payload = {
        "schema_version": 1,
        "status": (
            "developability_measurements_complete"
            if records and all(row["checks"]["measurements_complete"] for row in records)
            else "developability_measurements_blocked"),
        "classification": config["classification"],
        "candidate_count": len(records),
        "complete_count": sum(row["checks"]["measurements_complete"] for row in records),
        "pass_count": sum(row["developability_pass"] for row in records),
        "records": records,
        "decision": "use_measured_qc_only_never_replace_with_proxy",
        "claim_boundary": config["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/benchmarks/developability_measurement_gate_v1.yml"))
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--template-from", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.measurements) == bool(args.template_from):
        parser.error("Choose exactly one of --measurements or --template-from")
    if args.template_from:
        output = ROOT / (args.output or Path(
            "reviewer_outputs/developability_measurement_gate_v1/template.json"))
        payload = candidate_template(ROOT / args.template_from)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    else:
        config_path = ROOT / args.config
        config = yaml.safe_load(config_path.read_text(encoding="ascii"))
        output = ROOT / (args.output or Path(config["output"]))
        payload = audit(config_path, ROOT / args.measurements, output)
    print(json.dumps({
        "status": payload["status"], "records": len(payload["records"]),
        "output": str(output.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
