#!/usr/bin/env python
"""Run frozen multi-seed AF2-Multimer evaluation for v2 selections and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    temporary.replace(path)


def shuffled_h3(native, seed):
    values = list(native)
    generator = random.Random(seed)
    for _attempt in range(100):
        generator.shuffle(values)
        candidate = "".join(values)
        if candidate != native:
            return candidate
    raise ValueError("Cannot construct a non-native composition shuffle")


def replace_h3(heavy, indices, h3):
    if len(indices) != len(h3):
        raise ValueError("Control H3 length differs from frozen positions")
    output = list(heavy)
    for index, amino_acid in zip(indices, h3, strict=True):
        output[index] = amino_acid
    return "".join(output)


def build_entities(selection, holdout, shuffle_seed):
    representatives = {
        component["component_id"]: component["representative"]
        for component in holdout["components"]}
    entities = []
    for row in selection["selections"]:
        if row["status"] != "selected":
            continue
        representative = representatives[row["component_id"]]
        entities.append({
            "entity_id": row["selection_id"],
            "entity_type": "candidate",
            "component_id": row["component_id"],
            "arm": row["arm"],
            "selection_slot": row["selection_slot"],
            "source_attempt_id": row["source_attempt_id"],
            "h3_sequence": row["sequence"],
            "heavy_sequence": row["full_heavy_sequence"],
            "light_sequence": representative["light_sequence"],
            "antigen_sequence": representative["antigen_sequence"],
        })
    for component_index, component in enumerate(holdout["components"]):
        representative = component["representative"]
        native = representative["cdr_h3_sequence"]
        shuffle = shuffled_h3(native, shuffle_seed + component_index)
        for control, h3 in (("native", native), ("composition_shuffle", shuffle)):
            entities.append({
                "entity_id": f"{component['component_id']}|control|{control}",
                "entity_type": "control",
                "component_id": component["component_id"],
                "arm": control,
                "selection_slot": None,
                "source_attempt_id": None,
                "h3_sequence": h3,
                "heavy_sequence": replace_h3(
                    representative["heavy_sequence"],
                    representative["h3_heavy_indices_zero_based"], h3),
                "light_sequence": representative["light_sequence"],
                "antigen_sequence": representative["antigen_sequence"],
            })
    return entities


def windows_to_wsl(path):
    resolved = path.resolve().as_posix()
    if len(resolved) >= 3 and resolved[1:3] == ":/":
        return f"/mnt/{resolved[0].lower()}{resolved[2:]}"
    raise ValueError(f"Cannot map path to WSL: {path}")


def run_chunk(jobs, config, timeout):
    af2 = config["af2"]
    command = (
        f"cd {shlex.quote(windows_to_wsl(ROOT))} && "
        f"source {shlex.quote(af2['environment'])}/bin/activate && "
        "python scripts/utils/af2_wsl_batch.py "
        f"--recycle {int(af2['recycles'])}"
    )
    payload = "".join(json.dumps(job) + "\n" for job in jobs)
    completed = subprocess.run(
        ["wsl.exe", "-d", af2["wsl_distribution"], "--", "bash", "-lc", command],
        input=payload, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:] or f"WSL exited {completed.returncode}")
    rows = []
    for line in completed.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in row:
            rows.append(row)
    return rows, completed.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/multiscaffold_confirmatory_v2_af2.yml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--components", nargs="*")
    parser.add_argument("--max-entities", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection_path = ROOT / config["selection"]
    holdout_path = ROOT / config["holdout_manifest"]
    selection_config_path = ROOT / config["selection_config"]
    parameter_path = Path(config["af2"]["model_parameters"])
    for path, expected in [
        (selection_path, config["selection_sha256"]),
        (holdout_path, config["holdout_manifest_sha256"]),
        (selection_config_path, config["selection_config_sha256"]),
        (parameter_path, config["af2"]["model_parameters_sha256"]),
    ]:
        if sha256(path) != expected:
            raise ValueError(f"Frozen hash mismatch: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    entities = build_entities(
        selection, holdout, int(config["controls"]["composition_shuffle_seed"]))
    if args.components:
        requested = set(args.components)
        entities = [row for row in entities if row["component_id"] in requested]
    entities.sort(key=lambda row: (
        len(row["heavy_sequence"]) + len(row["light_sequence"]) + len(row["antigen_sequence"]),
        row["component_id"], row["entity_id"]))
    if args.max_entities is not None:
        entities = entities[:args.max_entities]
    seeds = config["af2"]["seeds"]
    output_path = ROOT / (args.out or config["output"]["results"])
    structure_dir = ROOT / (args.structures or config["output"]["structures"])
    structure_dir.mkdir(parents=True, exist_ok=True)
    requested = {
        "entities": [row["entity_id"] for row in entities],
        "seeds": seeds,
        "prediction_slots": len(entities) * len(seeds),
    }
    output = {
        "schema_version": 1,
        "status": "af2_in_progress",
        "config": args.config,
        "config_sha256": sha256(config_path),
        "selection_sha256": sha256(selection_path),
        "requested": requested,
        "entities": entities,
        "results": [],
        "claim_boundary": config["claim_boundary"],
    }
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "selection_sha256", "requested"):
            if previous[key] != output[key]:
                raise ValueError(f"Resume contract mismatch: {key}")
        output = previous
        output["status"] = "af2_in_progress"
    completed_ids = {
        row["prediction_id"] for row in output["results"]
        if not args.retry_failures or row["status"] == "success"}
    entity_by_id = {row["entity_id"]: row for row in entities}
    jobs = []
    for entity in entities:
        for seed in seeds:
            prediction_id = f"{entity['entity_id']}|af2_seed_{seed}"
            if prediction_id in completed_ids:
                continue
            safe_name = hashlib.sha256(prediction_id.encode("ascii")).hexdigest()[:20]
            pdb_path = structure_dir / f"{safe_name}.pdb"
            jobs.append({
                "id": prediction_id,
                "seq": f"{entity['heavy_sequence']}:{entity['light_sequence']}",
                "epi_seq": entity["antigen_sequence"],
                "seed": seed,
                "recycle": int(config["af2"]["recycles"]),
                "output_pdb": windows_to_wsl(pdb_path),
                "return_pae": bool(config["af2"]["retain_full_pae_matrix"]),
            })

    chunk_size = args.chunk_size or int(config["af2"]["chunk_size"])
    for chunk_start in range(0, len(jobs), chunk_size):
        chunk = jobs[chunk_start:chunk_start + chunk_size]
        started = time.perf_counter()
        error = None
        try:
            returned, stderr = run_chunk(chunk, config, timeout=600 + 600 * len(chunk))
        except Exception as chunk_error:  # noqa: BLE001
            returned = []
            stderr = ""
            error = f"{type(chunk_error).__name__}: {chunk_error}"
        returned_by_id = {row["id"]: row for row in returned}
        elapsed = time.perf_counter() - started
        chunk_ids = {job["id"] for job in chunk}
        output["results"] = [
            row for row in output["results"] if row["prediction_id"] not in chunk_ids]
        for job in chunk:
            result = returned_by_id.get(job["id"])
            entity_id, seed_text = job["id"].rsplit("|af2_seed_", 1)
            entity = entity_by_id[entity_id]
            success = bool(result and result.get("success"))
            plddt_seq = result.get("plddt_seq", []) if result else []
            antibody_length = len(entity["heavy_sequence"]) + len(entity["light_sequence"])
            pdb_path = Path(job["output_pdb"].replace("/mnt/c/", "C:/"))
            row = {
                "prediction_id": job["id"],
                "entity_id": entity_id,
                "component_id": entity["component_id"],
                "arm": entity["arm"],
                "entity_type": entity["entity_type"],
                "af2_seed": int(seed_text),
                "status": "success" if success else "failed",
                "iptm": result.get("iptm") if result else None,
                "ptm": result.get("ptm") if result else None,
                "plddt": result.get("plddt") if result else None,
                "antibody_plddt": (
                    sum(plddt_seq[:antibody_length]) / antibody_length
                    if len(plddt_seq) >= antibody_length else None),
                "antigen_plddt": (
                    sum(plddt_seq[antibody_length:]) / len(entity["antigen_sequence"])
                    if len(plddt_seq) == antibody_length + len(entity["antigen_sequence"])
                    else None),
                "interface_pae": result.get("interface_pae") if result else None,
                "max_pae": result.get("max_pae") if result else None,
                "elapsed": result.get("elapsed") if result else None,
                "chunk_wall_seconds": elapsed,
                "error": None if success else (
                    (result or {}).get("error") or error or "missing worker result"),
                "worker_stderr": stderr[-2000:] or None,
            }
            if success and pdb_path.exists():
                row["pdb"] = pdb_path.relative_to(ROOT).as_posix()
                row["pdb_sha256"] = sha256(pdb_path)
            output["results"].append(row)
        atomic_json(output_path, output)
        successes = sum(row["status"] == "success" for row in output["results"])
        print(
            f"AF2 chunk {chunk_start // chunk_size + 1}: "
            f"recorded={len(output['results'])}/{requested['prediction_slots']} "
            f"successes={successes}", flush=True)

    output["results"].sort(key=lambda row: row["prediction_id"])
    output["summary"] = {
        "expected_prediction_slots": requested["prediction_slots"],
        "recorded_prediction_slots": len(output["results"]),
        "successes": sum(row["status"] == "success" for row in output["results"]),
        "failures": sum(row["status"] == "failed" for row in output["results"]),
    }
    output["status"] = (
        "af2_complete" if len(output["results"]) == requested["prediction_slots"]
        else "af2_incomplete")
    atomic_json(output_path, output)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
