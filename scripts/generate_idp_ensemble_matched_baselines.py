#!/usr/bin/env python
"""Generate matched H3 candidates for IDP ensemble benchmark components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_candidate_reranking import (  # noqa: E402
    generate_esmif,
    load_esmif,
    validate_candidate,
)

from modules.h3_interface_contacts import write_standardized_backbone  # noqa: E402

METHODS = ("proteinmpnn", "single_conformer_bfn", "esm_if")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def canonicalize_component(component, output_path):
    source = ROOT / component["reference_path"]
    source_chains = [component["heavy_chain"]]
    canonical_chains = ["H"]
    if component.get("light_chain"):
        source_chains.append(component["light_chain"])
        canonical_chains.append("L")
    source_chains.append(component["antigen_chain"])
    canonical_chains.append("P")
    temporary = output_path.with_suffix(".source_chains.pdb")
    original = write_standardized_backbone(str(source), str(temporary), source_chains,
                                           skip_backbone_check_chains=[source_chains[-1]])
    role_map = dict(zip(source_chains, canonical_chains, strict=True))
    lines = []
    for line in temporary.read_text(encoding="ascii").splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            line = f"{line[:21]}{role_map[line[21]]}{line[22:]}"
        lines.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="ascii")
    temporary.unlink()
    chains = {
        canonical: original["chains"][source_chain]
        for source_chain, canonical in role_map.items()
    }
    start, end = component["h3_positions_1_indexed"]
    native = chains["H"]["sequence"][start - 1:end]
    if native != component["native_h3"]:
        raise ValueError(
            f"{component['component_id']} H3 mapping mismatch: {native} != {component['native_h3']}"
        )
    return {
        "source_pdb": component["reference_path"],
        "source_sha256": sha256(source),
        "canonical_pdb": display_path(output_path),
        "canonical_sha256": sha256(output_path),
        "source_to_canonical_chain": role_map,
        "chains": chains,
        "h3_positions_1_indexed": [start, end],
        "native_h3": native,
    }


def parse_mpnn_fasta(path, component, chain_manifest, seed):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chain_order = [chain for chain in ("H", "L", "P") if chain in chain_manifest]
    expected = [chain_manifest[chain]["sequence"] for chain in chain_order]
    if len(lines) < 2 or lines[1].split("/") != expected:
        raise ValueError(
            f"{component['component_id']} ProteinMPNN chain order is not "
            f"{'/'.join(chain_order)}"
        )
    start, end = component["h3_positions_1_indexed"]
    rows = []
    for index in range(2, len(lines), 2):
        header = lines[index]
        segments = lines[index + 1].split("/")
        if len(segments) != len(chain_order):
            raise ValueError("ProteinMPNN output chain count mismatch")
        for chain_index, chain in enumerate(chain_order):
            native = chain_manifest[chain]["sequence"]
            if len(segments[chain_index]) != len(native):
                raise ValueError(f"ProteinMPNN changed {chain} chain length")
            if chain != "H" and segments[chain_index] != native:
                raise ValueError(f"ProteinMPNN changed fixed {chain} chain")
        heavy = segments[0]
        mutable = set(range(start - 1, end))
        if any(
            heavy[position] != expected[0][position]
            for position in range(len(heavy))
            if position not in mutable
        ):
            raise ValueError("ProteinMPNN changed fixed heavy-chain positions")
        fields = {}
        for field in header[1:].split(","):
            if "=" in field:
                key, value = field.split("=", 1)
                fields[key.strip()] = value.strip()
        sequence = validate_candidate(heavy[start - 1:end], end - start + 1)
        rows.append({
            "seed": seed,
            "sample_index": len(rows),
            "sequence": sequence,
            "generator_score": float(fields.get("score", "nan")),
            "global_score": float(fields.get("global_score", "nan")),
        })
    return rows


def generate_proteinmpnn(config, component, canonical_pdb, manifest, seed, work_dir):
    settings = config["generation"]["proteinmpnn"]
    start, end = component["h3_positions_1_indexed"]
    chain_order = [chain for chain in ("H", "L", "P") if chain in manifest]
    fixed = {
        chain: (
            [position for position in range(1, manifest["H"]["length"] + 1)
             if not start <= position <= end]
            if chain == "H" else list(range(1, manifest[chain]["length"] + 1))
        )
        for chain in chain_order
    }
    name = canonical_pdb.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = work_dir / "fixed_positions.jsonl"
    fixed_path.write_text(json.dumps({name: fixed}) + "\n", encoding="ascii")
    output_dir = work_dir / "mpnn"
    fasta = output_dir / "seqs" / f"{name}.fa"
    command = [
        sys.executable,
        str(ROOT / settings["script"]),
        "--pdb_path", canonical_pdb.resolve().as_posix(),
        "--pdb_path_chains", " ".join(chain_order),
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (ROOT / settings["weights"]).resolve().as_posix(),
        "--model_name", settings["model_name"],
        "--num_seq_per_target", str(config["generation"]["candidates_per_seed"]),
        "--batch_size", "1",
        "--sampling_temp", str(settings["temperature"]),
        "--seed", str(seed),
        "--save_score", "1",
        "--suppress_print", "1",
        "--out_folder", output_dir.resolve().as_posix(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    return parse_mpnn_fasta(fasta, component, manifest, seed), command


def generate_bfn(config, component, canonical_pdb, seed):
    from modules import bfn_loader

    settings = config["generation"]["single_conformer_bfn"]
    checkpoint = (ROOT / settings["checkpoint"]).resolve()
    os.environ["DISORDERFLOW_CHECKPOINT"] = str(checkpoint)
    start, end = component["h3_positions_1_indexed"]
    rows = bfn_loader.run_bfn_design(
        str(canonical_pdb),
        f"H:{start}-{end}",
        num_samples=int(config["generation"]["candidates_per_seed"]),
        stochastic=bool(settings["stochastic"]),
        context_chains=[chain for chain in ("L", "P") if chain != "L" or component.get("light_chain")],
        antigen_chains=["P"],
        device=settings["device"],
        disorder_guided=False,
        sampling_seed=seed,
    )
    return [
        {"seed": seed, "sample_index": index, **row}
        for index, row in enumerate(rows)
    ]


def generate_esm_if(config, model, component, canonical_pdb, seed):
    adapter_config = {
        "generation": {
            "candidates_per_seed": config["generation"]["candidates_per_seed"],
            "esm_if": {"temperature": config["generation"]["esm_if"]["temperature"]},
        }
    }
    start, end = component["h3_positions_1_indexed"]
    chain_order = ["H"] + (["L"] if component.get("light_chain") else []) + ["P"]
    rows = generate_esmif(
        adapter_config, model, canonical_pdb, list(range(start - 1, end)), chain_order, seed
    )
    return [
        {"seed": seed, "sample_index": index, **row}
        for index, row in enumerate(rows)
    ]


def validate_rows(rows, component, expected):
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} candidates, found {len(rows)}")
    length = len(component["native_h3"])
    for row in rows:
        validate_candidate(row["sequence"], length)


def write_output(config_path, config, manifest_path, slots, failures, out_path):
    requested_methods = [method for method in METHODS if config["generation"][method]["enabled"]]
    components = json.loads(manifest_path.read_text(encoding="utf-8"))["components"]
    expected = len(components) * len(config["generation"]["seeds"]) * len(requested_methods)
    output = {
        "schema_version": 1,
        "status": "complete" if len(slots) == expected and not failures else "partial",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "component_manifest": str(manifest_path.relative_to(ROOT)),
        "component_manifest_sha256": sha256(manifest_path),
        "requested": {
            "components": len(components),
            "methods": requested_methods,
            "seeds": config["generation"]["seeds"],
            "candidates_per_seed": config["generation"]["candidates_per_seed"],
            "expected_slots": expected,
        },
        "completed_slots": len(slots),
        "failures": failures,
        "slots": slots,
        "claim_boundary": config["claim_boundary"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_matched_generation_dev_v1.yml"
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--components", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["component_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = manifest["components"]
    if args.components:
        requested = set(args.components)
        components = [row for row in components if row["component_id"] in requested]
        if {row["component_id"] for row in components} != requested:
            raise ValueError("Unknown component requested")

    for method in args.methods:
        settings = config["generation"][method]
        checkpoint = Path(settings["checkpoint"])
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        if sha256(checkpoint) != settings["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")

    out_path = ROOT / (args.out or config["output"])
    work_root = out_path.parent / "work"
    slots = []
    failures = []
    if out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != sha256(config_path):
            raise ValueError("Existing output was produced by a different generation config")
        if previous.get("component_manifest_sha256") != sha256(manifest_path):
            raise ValueError("Existing output was produced by a different component manifest")
        slots = previous.get("slots", [])
        failures = previous.get("failures", [])
    completed_keys = {
        (row["component_id"], row["method"], int(row["seed"])) for row in slots
        if row["status"] == "success"
    }
    output = write_output(config_path, config, manifest_path, slots, failures, out_path)
    esm_model = None
    if "esm_if" in args.methods:
        esm_model, _ = load_esmif()

    for component in components:
        component_dir = work_root / component["component_id"]
        canonical_pdb = component_dir / "canonical_HLP.pdb"
        canonical_manifest = canonicalize_component(component, canonical_pdb)
        for method in args.methods:
            for seed in config["generation"]["seeds"]:
                slot_key = (component["component_id"], method, int(seed))
                if slot_key in completed_keys:
                    print(f"{component['component_id']} {method} seed={seed}: cached", flush=True)
                    continue
                failures = [
                    row for row in failures
                    if (row["component_id"], row["method"], int(row["seed"])) != slot_key
                ]
                started = time.perf_counter()
                try:
                    if method == "proteinmpnn":
                        rows, command = generate_proteinmpnn(
                            config,
                            component,
                            canonical_pdb,
                            canonical_manifest["chains"],
                            int(seed),
                            component_dir / method / f"seed_{seed}",
                        )
                    elif method == "single_conformer_bfn":
                        rows = generate_bfn(config, component, canonical_pdb, int(seed))
                        command = None
                    else:
                        rows = generate_esm_if(config, esm_model, component, canonical_pdb, int(seed))
                        command = None
                    validate_rows(
                        rows, component, int(config["validity"]["exact_raw_candidates_per_slot"])
                    )
                    slots.append({
                        "component_id": component["component_id"],
                        "method": method,
                        "seed": seed,
                        "status": "success",
                        "wall_seconds": time.perf_counter() - started,
                        "canonical_structure": canonical_manifest,
                        "command": command,
                        "candidates": rows,
                    })
                    completed_keys.add(slot_key)
                except Exception as error:  # noqa: BLE001
                    failures.append({
                        "component_id": component["component_id"],
                        "method": method,
                        "seed": seed,
                        "status": "failure",
                        "error": str(error),
                    })
                output = write_output(config_path, config, manifest_path, slots, failures, out_path)
                print(
                    f"{component['component_id']} {method} seed={seed}: "
                    f"slots={output['completed_slots']} failures={len(failures)}",
                    flush=True,
                )

    if output["status"] != "complete" and not args.components and set(args.methods) == set(METHODS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
