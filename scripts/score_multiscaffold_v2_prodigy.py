#!/usr/bin/env python
"""Score every frozen v2 AF2 structure with independent PRODIGY."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/multiscaffold_confirmatory_v2_prodigy.yml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    af2_config_path = ROOT / config["af2_config"]
    af2_path = ROOT / config["af2_results"]
    for path, expected in [
        (af2_config_path, config["af2_config_sha256"]),
        (af2_path, config["af2_results_sha256"]),
    ]:
        if sha256(path) != expected:
            raise ValueError(f"Frozen hash mismatch: {path}")
    installed_version = importlib.metadata.version(config["scorer"]["package"])
    if installed_version != config["scorer"]["version"]:
        raise ValueError(f"PRODIGY version {installed_version} differs from frozen config")
    af2 = json.loads(af2_path.read_text(encoding="utf-8"))
    if af2["status"] != "af2_complete" or any(
        row["status"] != "success" for row in af2["results"]
    ):
        raise ValueError("AF2 artifact is incomplete")
    pdb_dir = ROOT / "results/multiscaffold_confirmatory_v2/af2/pdb"
    executable = Path(sys.executable).parent / "Scripts" / "prodigy.exe"
    if not executable.exists():
        executable = Path(sys.executable).parent / "prodigy.exe"
    if not executable.exists():
        user_executable = Path.home() / (
            "AppData/Roaming/Python/Python314/Scripts/prodigy.exe")
        executable = user_executable
    command = [
        str(executable), str(pdb_dir), "--selection",
        *config["scorer"]["selection_arguments"], "--temperature",
        str(config["scorer"]["temperature_celsius"]), "-q", "-np",
        str(config["scorer"]["processors"]),
    ]
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=7200)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    energies = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            energies[fields[0].removesuffix("_model0")] = float(fields[1])
        except ValueError:
            continue

    rows = []
    for af2_row in af2["results"]:
        stem = Path(af2_row["pdb"]).stem
        energy = energies.get(stem)
        rows.append({
            "prediction_id": af2_row["prediction_id"],
            "entity_id": af2_row["entity_id"],
            "component_id": af2_row["component_id"],
            "arm": af2_row["arm"],
            "entity_type": af2_row["entity_type"],
            "af2_seed": af2_row["af2_seed"],
            "status": "success" if energy is not None else "failed",
            "predicted_delta_g_kcal_mol": energy,
            "error": None if energy is not None else "missing_prodigy_output",
            "source_pdb": af2_row["pdb"],
            "source_pdb_sha256": af2_row["pdb_sha256"],
        })
    by_entity = defaultdict(list)
    entity_metadata = {}
    for row in rows:
        if row["status"] == "success":
            by_entity[row["entity_id"]].append(row["predicted_delta_g_kcal_mol"])
        entity_metadata[row["entity_id"]] = {
            "component_id": row["component_id"],
            "arm": row["arm"],
            "entity_type": row["entity_type"],
        }
    entities = {}
    for entity_id, metadata in entity_metadata.items():
        values = by_entity[entity_id]
        entities[entity_id] = {
            **metadata,
            "n_successful_seeds": len(values),
            "median_delta_g_kcal_mol": float(np.median(values)) if values else None,
            "maximum_worst_delta_g_kcal_mol": float(np.max(values)) if values else None,
            "minimum_best_delta_g_kcal_mol": float(np.min(values)) if values else None,
            "range_delta_g_kcal_mol": float(np.ptp(values)) if values else None,
        }
    output = {
        "schema_version": 1,
        "status": "independent_interface_scoring_complete",
        "config": args.config,
        "config_sha256": sha256(config_path),
        "af2_results_sha256": sha256(af2_path),
        "scorer": config["scorer"],
        "summary": {
            "expected_slots": int(config["expected_slots"]),
            "recorded_slots": len(rows),
            "successes": sum(row["status"] == "success" for row in rows),
            "failures": sum(row["status"] == "failed" for row in rows),
            "entities": len(entities),
        },
        "rows": rows,
        "entities": entities,
        "claim_boundary": config["claim_boundary"],
    }
    out_path = ROOT / (args.out or config["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
