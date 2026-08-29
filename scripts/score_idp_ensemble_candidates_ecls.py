#!/usr/bin/env python
"""Score matched H3 candidates by ProteinMPNN epitope-conditioning gain."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_h3_epitope_delta import sequence_nll  # noqa: E402
from scripts.generate_idp_ensemble_matched_baselines import canonicalize_component  # noqa: E402
from scripts.score_idp_ensemble_candidates_bfn import summarize  # noqa: E402
from scripts.score_idp_ensemble_matched_candidates import (  # noqa: E402
    candidate_pools,
    leave_one_out,
    pose_rows_by_component,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_apo(complex_path, apo_path):
    lines = [
        line for line in complex_path.read_text(encoding="ascii").splitlines(keepends=True)
        if not line.startswith(("ATOM  ", "HETATM")) or line[21] != "P"
    ]
    apo_path.write_text("".join(lines), encoding="ascii")


def run_conditional(config, pdb_path, h3_indices, heavy_length, work_dir):
    settings = config["proteinmpnn"]
    name = pdb_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    fixed = [
        position for position in range(1, heavy_length + 1)
        if position - 1 not in set(h3_indices)
    ]
    fixed_path = work_dir / "fixed_positions.jsonl"
    fixed_path.write_text(json.dumps({name: {"H": fixed}}) + "\n", encoding="ascii")
    output_dir = work_dir / "mpnn"
    result_path = output_dir / "conditional_probs_only" / f"{name}.npz"
    command = [
        sys.executable,
        str(ROOT / settings["script"]),
        "--pdb_path", pdb_path.resolve().as_posix(),
        "--pdb_path_chains", "H",
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (ROOT / settings["weights"]).resolve().as_posix(),
        "--model_name", settings["model_name"],
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1",
        "--batch_size", "1",
        "--seed", str(settings["seed"]),
        "--suppress_print", "1",
        "--out_folder", output_dir.resolve().as_posix(),
    ]
    if not result_path.exists():
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    result = np.load(result_path)
    for index in h3_indices:
        if result["design_mask"][index] <= 0 or result["mask"][index] <= 0:
            raise ValueError(f"Invalid ProteinMPNN H3 mask at index {index}")
    return result["log_p"][0], {
        "npz": str(result_path.relative_to(ROOT)),
        "npz_sha256": sha256(result_path),
        "command": command,
    }


def score_component(config, component, pose_rows, method_pools, work_dir):
    start, end = component["h3_positions_1_indexed"]
    h3_indices = list(range(start - 1, end))
    conformers = []
    complex_log_probs = []
    apo_log_probs = []
    for index, pose in enumerate(pose_rows):
        pose_component = {
            **component,
            "reference_path": pose["pose_pdb"],
            "antigen_chain": "P",
        }
        pose_dir = work_dir / f"pose_{index}"
        complex_path = pose_dir / "complex_HLP.pdb"
        pose_dir.mkdir(parents=True, exist_ok=True)
        canonical = canonicalize_component(pose_component, complex_path)
        heavy_length = int(canonical["chains"]["H"]["length"])
        apo_path = pose_dir / "apo_HL.pdb"
        write_apo(complex_path, apo_path)
        complex_logp, complex_provenance = run_conditional(
            config, complex_path, h3_indices, heavy_length, pose_dir / "complex"
        )
        apo_logp, apo_provenance = run_conditional(
            config, apo_path, h3_indices, heavy_length, pose_dir / "apo"
        )
        complex_log_probs.append(complex_logp)
        apo_log_probs.append(apo_logp)
        conformers.append({
            "conformer": pose["conformer"],
            "source_pose": pose["pose_pdb"],
            "source_pose_sha256": sha256(ROOT / pose["pose_pdb"]),
            "canonical": canonical,
            "complex": complex_provenance,
            "apo": apo_provenance,
        })

    def gains(sequence):
        return [
            sequence_nll(apo, sequence, h3_indices)
            - sequence_nll(complex_logp, sequence, h3_indices)
            for complex_logp, apo in zip(complex_log_probs, apo_log_probs, strict=True)
        ]

    native = gains(component["native_h3"])
    methods = []
    for method, sequences in method_pools.items():
        methods.append({
            "method": method,
            "sequences": sequences,
            "epitope_conditioning_gain": [gains(sequence) for sequence in sequences],
        })
    return {
        "component_id": component["component_id"],
        "native_h3": component["native_h3"],
        "native_epitope_conditioning_gain": native,
        "conformers": conformers,
        "methods": methods,
    }


def cache_valid(payload, config_hash, generation_hash, component, pools):
    if payload.get("config_sha256") != config_hash:
        return False
    if payload.get("raw_generation_sha256") != generation_hash:
        return False
    scores = payload.get("scores", {})
    if scores.get("component_id") != component["component_id"]:
        return False
    actual = {row["method"]: row["sequences"] for row in scores.get("methods", [])}
    return actual == pools


def pose_rows_from_manifest(components):
    output = {}
    for component in components:
        paths = component.get("pose_paths", [])
        sensitivity_only = bool(component.get("sensitivity_only"))
        if not sensitivity_only and len(paths) != 5:
            raise ValueError(
                f"{component['component_id']} requires exactly five "
                f"manifest pose paths, found {len(paths)}"
            )
        if sensitivity_only and not paths:
            raise ValueError(
                f"{component['component_id']} sensitivity component requires at least "
                "one manifest pose path"
            )
        output[component["component_id"]] = [
            {"conformer": index, "pose_pdb": path}
            for index, path in enumerate(paths)
        ]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_ecls_scoring_dev_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = {
        "component_manifest": config["component_manifest"],
        "raw_generation": config["raw_generation"],
    }
    if "preparation" in config:
        frozen["preparation"] = config["preparation"]
    if "pose_audits" in config:
        frozen.update({
            "abeta_pose_audit": config["pose_audits"]["abeta"],
            "nonabeta_pose_audit": config["pose_audits"]["nonabeta"],
        })
    paths = {name: ROOT / item["path"] for name, item in frozen.items()}
    for name, path in paths.items():
        if sha256(path) != frozen[name]["sha256"]:
            raise ValueError(f"Frozen hash mismatch: {path}")
    checkpoint = ROOT / config["proteinmpnn"]["checkpoint"]
    if sha256(checkpoint) != config["proteinmpnn"]["checkpoint_sha256"]:
        raise ValueError("ProteinMPNN checkpoint hash mismatch")

    manifest = json.loads(paths["component_manifest"].read_text(encoding="utf-8"))
    raw = json.loads(paths["raw_generation"].read_text(encoding="utf-8"))
    if raw["status"] != "complete" or raw["failures"]:
        raise ValueError("Matched generation must be complete")
    components = manifest["components"]
    methods = config["candidate_pool"]["methods"]
    pools = candidate_pools(raw, methods)
    if config.get("pose_source") == "component_manifest_pose_paths":
        poses = pose_rows_from_manifest(components)
    else:
        poses = pose_rows_by_component(
            components,
            json.loads(paths["abeta_pose_audit"].read_text(encoding="utf-8")),
            json.loads(paths["nonabeta_pose_audit"].read_text(encoding="utf-8")),
        )

    out_path = ROOT / (args.out or config["output"])
    work_dir = out_path.parent / "work"
    config_hash = sha256(config_path)
    generation_hash = sha256(paths["raw_generation"])
    component_scores = []
    for component in components:
        component_id = component["component_id"]
        component_pools = {method: pools[(component_id, method)] for method in methods}
        cache_path = work_dir / component_id / "ecls_scores.json"
        cached = None
        if cache_path.exists():
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_valid(candidate, config_hash, generation_hash, component, component_pools):
                cached = candidate["scores"]
        if cached is None:
            scores = score_component(
                config,
                component,
                poses[component_id],
                component_pools,
                work_dir / component_id / "states",
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "config_sha256": config_hash,
                "raw_generation_sha256": generation_hash,
                "component": component,
                "scores": scores,
            }, indent=2) + "\n", encoding="ascii")
            cached = scores
        component_scores.append(cached)
        print(f"ECLS scoring complete: {component_id}", flush=True)

    loo_rows = []
    aggregation = config["primary"]["ensemble_aggregation"]
    for component in component_scores:
        native = np.asarray(component["native_epitope_conditioning_gain"], dtype=np.float64)
        for method_row in component["methods"]:
            matrix = np.asarray(method_row["epitope_conditioning_gain"], dtype=np.float64)
            loo_rows.extend(leave_one_out(
                component["component_id"],
                method_row["method"],
                method_row["sequences"],
                matrix,
                native,
                aggregation,
            ))
    component_ids = [row["component_id"] for row in components]
    method_summary = summarize(
        loo_rows, methods, component_ids, config["statistics"], config["development_gates"]
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
            "config_sha256": config_hash,
            "inputs": {name: sha256(path) for name, path in paths.items()},
            "checkpoint_sha256": sha256(checkpoint),
        },
        "method_summary": method_summary,
        "leave_one_conformer_out": loo_rows,
        "component_scores": component_scores,
        "claim_boundary": config["claim_boundary"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": output["status"], "method_summary": method_summary}, indent=2))


if __name__ == "__main__":
    main()
