#!/usr/bin/env python
"""Run frozen multi-seed AF2-Multimer triage for the 3D6 H3 shortlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import time
from pathlib import Path

import yaml
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    temporary.replace(path)


def windows_to_wsl(path):
    value = Path(path).resolve().as_posix()
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}{value[2:]}"
    raise ValueError(f"Cannot map path to WSL: {path}")


def chain_sequences(path):
    model = PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))[0]
    return {
        chain.id: "".join(AA3[residue.resname] for residue in chain if residue.resname in AA3)
        for chain in model
    }


def shuffled_h3(native, seed):
    rng = random.Random(seed)
    values = list(native)
    for _attempt in range(100):
        rng.shuffle(values)
        sequence = "".join(values)
        if sequence != native:
            return sequence
    raise ValueError("Could not construct a non-native composition shuffle")


def replace_h3(heavy, h3):
    if len(h3) != 12:
        raise ValueError(f"Expected 12-residue H3, got {h3}")
    return heavy[:95] + h3 + heavy[107:]


def validate_contract(config):
    source = ROOT / config["input"]["prospective_results"]
    scaffold = ROOT / config["input"]["scaffold"]
    mpnn = ROOT / config["input"]["proteinmpnn_control"]
    worker = ROOT / config["af2"]["worker"]
    for path, expected in (
        (source, config["input"]["prospective_results_sha256"]),
        (scaffold, config["input"]["scaffold_sha256"]),
        (mpnn, config["input"]["proteinmpnn_control_sha256"]),
        (worker, config["af2"]["worker_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")


def build_entities(config):
    source = json.loads((ROOT / config["input"]["prospective_results"]).read_text())
    if len(source["shortlist"]) != int(config["input"]["expected_candidates"]):
        raise ValueError("Prospective shortlist count changed")
    scaffold_sequences = chain_sequences(ROOT / config["input"]["scaffold"])
    native = config["input"]["native_h3"]
    if scaffold_sequences["H"][95:107] != native:
        raise ValueError("Frozen H3 mapping changed")
    entities = []
    for row in source["shortlist"]:
        rank = int(row["rank"])
        entities.append({
            "entity_id": f"3D6-H3-PV1-{rank:02d}",
            "entity_type": "candidate",
            "pre_af2_rank": rank,
            "h3_sequence": row["sequence"],
            "heavy_sequence": replace_h3(scaffold_sequences["H"], row["sequence"]),
            "light_sequence": scaffold_sequences["L"],
            "antigen_sequence": scaffold_sequences["A"],
        })
    mpnn = json.loads((ROOT / config["input"]["proteinmpnn_control"]).read_text())
    best_mpnn = min(mpnn["results"], key=lambda row: (row["score"], row["seed"], row["sample_index"]))
    controls = [
        ("native", native),
        ("proteinmpnn_best", best_mpnn["sequence"]),
        ("composition_shuffle", shuffled_h3(
            native, int(config["controls"]["composition_shuffle_seed"]))),
    ]
    for control_id, h3 in controls:
        entities.append({
            "entity_id": control_id,
            "entity_type": "control",
            "pre_af2_rank": None,
            "h3_sequence": h3,
            "heavy_sequence": replace_h3(scaffold_sequences["H"], h3),
            "light_sequence": scaffold_sequences["L"],
            "antigen_sequence": scaffold_sequences["A"],
        })
    return entities


def run_chunk(jobs, config):
    af2 = config["af2"]
    command = (
        f"cd {shlex.quote(windows_to_wsl(ROOT))} && "
        f"source {shlex.quote(af2['environment'])}/bin/activate && "
        f"python {shlex.quote(af2['worker'])} --recycle {int(af2['recycles'])}"
    )
    payload = "".join(json.dumps(job) + "\n" for job in jobs)
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    env["WSLENV"] = ":".join(filter(None, [
        env.get("WSLENV", ""), "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION"
    ]))
    completed = subprocess.run(
        ["wsl.exe", "-d", af2["wsl_distribution"], "--", "bash", "-lc", command],
        input=payload, capture_output=True, text=True, env=env,
        timeout=900 + 600 * len(jobs),
    )
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
    parser.add_argument("--config", default="configs/benchmarks/abeta_4hix_af2_gate_v1.yml")
    parser.add_argument("--out")
    parser.add_argument("--structures")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--max-entities", type=int)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_contract(config)
    entities = build_entities(config)
    if args.max_entities is not None:
        entities = entities[:args.max_entities]
    seeds = args.seeds or config["af2"]["seeds"]
    output_path = ROOT / (args.out or config["output"]["af2_results"])
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
        "runner_sha256": sha256(Path(__file__)),
        "smoke": args.smoke,
        "requested": requested,
        "entities": entities,
        "results": [],
        "claim_boundary": config["claim_boundary"],
    }
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text())
        for key in ("config_sha256", "runner_sha256", "requested"):
            if previous[key] != output[key]:
                raise ValueError(f"Resume contract mismatch: {key}")
        output = previous
        output["status"] = "af2_in_progress"
    completed_ids = {
        row["prediction_id"] for row in output["results"]
        if not args.retry_failures or row["status"] == "success"
    }
    jobs = []
    entity_by_id = {row["entity_id"]: row for row in entities}
    for entity in entities:
        for seed in seeds:
            prediction_id = f"{entity['entity_id']}|af2_seed_{seed}"
            if prediction_id in completed_ids:
                continue
            pdb_name = hashlib.sha256(prediction_id.encode("ascii")).hexdigest()[:20] + ".pdb"
            jobs.append({
                "id": prediction_id,
                "seq": f"{entity['heavy_sequence']}:{entity['light_sequence']}",
                "epi_seq": entity["antigen_sequence"],
                "seed": seed,
                "recycle": int(config["af2"]["recycles"]),
                "output_pdb": windows_to_wsl(structure_dir / pdb_name),
                "return_pae": bool(config["af2"]["retain_full_pae_matrix"]),
            })
    chunk_size = args.chunk_size or int(config["af2"]["chunk_size"])
    for chunk_start in range(0, len(jobs), chunk_size):
        chunk = jobs[chunk_start:chunk_start + chunk_size]
        started = time.perf_counter()
        error = None
        try:
            returned, stderr = run_chunk(chunk, config)
        except Exception as exc:  # noqa: BLE001
            returned, stderr = [], ""
            error = f"{type(exc).__name__}: {exc}"
        returned_by_id = {row["id"]: row for row in returned}
        chunk_ids = {job["id"] for job in chunk}
        output["results"] = [
            row for row in output["results"] if row["prediction_id"] not in chunk_ids
        ]
        for job in chunk:
            result = returned_by_id.get(job["id"])
            entity_id, seed_text = job["id"].rsplit("|af2_seed_", 1)
            entity = entity_by_id[entity_id]
            success = bool(result and result.get("success"))
            pdb_path = structure_dir / Path(job["output_pdb"]).name
            row = {
                "prediction_id": job["id"],
                "entity_id": entity_id,
                "entity_type": entity["entity_type"],
                "h3_sequence": entity["h3_sequence"],
                "pre_af2_rank": entity["pre_af2_rank"],
                "af2_seed": int(seed_text),
                "status": "success" if success else "failed",
                "iptm": result.get("iptm") if result else None,
                "ptm": result.get("ptm") if result else None,
                "plddt": result.get("plddt") if result else None,
                "interface_pae": result.get("interface_pae") if result else None,
                "max_pae": result.get("max_pae") if result else None,
                "elapsed": result.get("elapsed") if result else None,
                "chunk_wall_seconds": time.perf_counter() - started,
                "error": None if success else ((result or {}).get("error") or error or "missing result"),
                "worker_stderr": stderr[-2000:] or None,
            }
            if success and pdb_path.exists():
                row["pdb"] = pdb_path.relative_to(ROOT).as_posix()
                row["pdb_sha256"] = sha256(pdb_path)
            output["results"].append(row)
        atomic_json(output_path, output)
        print(
            f"AF2 chunk {chunk_start // chunk_size + 1}: "
            f"recorded={len(output['results'])}/{requested['prediction_slots']} "
            f"successes={sum(row['status'] == 'success' for row in output['results'])}",
            flush=True,
        )
    output["results"].sort(key=lambda row: row["prediction_id"])
    output["summary"] = {
        "expected_prediction_slots": requested["prediction_slots"],
        "recorded_prediction_slots": len(output["results"]),
        "successes": sum(row["status"] == "success" for row in output["results"]),
        "failures": sum(row["status"] == "failed" for row in output["results"]),
    }
    output["status"] = (
        "smoke_complete" if args.smoke and len(output["results"]) == requested["prediction_slots"]
        else "af2_complete" if len(output["results"]) == requested["prediction_slots"]
        else "af2_incomplete"
    )
    atomic_json(output_path, output)
    print(json.dumps(output["summary"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
