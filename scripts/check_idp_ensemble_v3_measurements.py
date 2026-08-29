#!/usr/bin/env python
"""Validate blinded v3 binding, specificity, expression, and QC measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED_COLUMNS = {
    "construct_id",
    "component_id",
    "binding_kd_molar",
    "expression_mg_per_l",
    "specificity_pass",
    "qc_pass",
}


def check(measurements, panel_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite measurement status: {output}")
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    component_ids = {row["component_id"] for row in panel.get("components", [])}
    if not measurements.is_file():
        rows, columns = [], set()
    else:
        with measurements.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
    valid_rows = [
        row for row in rows
        if row.get("component_id") in component_ids
        and row.get("construct_id")
        and row.get("binding_kd_molar")
        and row.get("expression_mg_per_l")
        and row.get("specificity_pass") in {"True", "False"}
        and row.get("qc_pass") in {"True", "False"}
    ]
    measured_components = {row["component_id"] for row in valid_rows}
    ready = REQUIRED_COLUMNS <= columns and measured_components == component_ids
    payload = {
        "schema_version": 1,
        "status": "measurements_complete" if ready else "measurements_blocked",
        "classification": "blinded_experimental_measurement_readiness",
        "required_columns": sorted(REQUIRED_COLUMNS),
        "observed_columns": sorted(columns),
        "required_components": len(component_ids),
        "measured_components": len(measured_components),
        "valid_rows": len(valid_rows),
        "decision": "allow_frozen_analysis" if ready else "acquire_missing_blinded_measurements",
        "claim_boundary": "measurement completeness only; no efficacy or therapeutic claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.measurements, args.panel, args.output), indent=2))


if __name__ == "__main__":
    main()
