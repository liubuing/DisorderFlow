#!/usr/bin/env python
"""Score matched H3 candidates across five poses with the fixed-sequence BFN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import bfn_loader  # noqa: E402
from scripts.generate_idp_ensemble_matched_baselines import canonicalize_component  # noqa: E402
from scripts.score_idp_ensemble_matched_candidates import (  # noqa: E402
    bootstrap_mean,
    candidate_pools,
    exact_sign_flip_p,
    leave_one_out,
    pose_rows_by_component,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_value(output, batch):
    generated = batch["generate_flag"][0].bool()
    plddt = output["plddt"][0][generated]
    pae = output["pae"][0][generated][:, generated]
    return {
        "iptm": float(output["iptm"].float().mean().item()),
        "generated_region_plddt": float(plddt.float().mean().item()),
        "generated_region_pae": float(pae.float().mean().item()),
        "state_compatibility": float(output["state_compatibility"].float().mean().item()),
    }


def score_component(config, component, pose_rows, method_pools, model, work_dir):
    canonical_poses = []
    batches = []
    start, end = component["h3_positions_1_indexed"]
    region = f"H:{start}-{end}"
    for index, pose in enumerate(pose_rows):
        pose_component = {
            **component,
            "reference_path": pose["pose_pdb"],
            "antigen_chain": "P",
        }
        output_path = work_dir / f"pose_{index}_HLP.pdb"
        provenance = canonicalize_component(pose_component, output_path)
        batch = bfn_loader.build_region_batch(
            str(output_path),
            region,
            context_chains=["L", "P"],
            antigen_chains=[config["bfn"]["antigen_chain"]],
            device=config["bfn"]["device"],
        )
        canonical_poses.append({
            "conformer": pose["conformer"],
            "pose_pdb": pose["pose_pdb"],
            "pose_sha256": sha256(ROOT / pose["pose_pdb"]),
            "canonical": provenance,
        })
        batches.append(batch)

    native_scores = []
    for batch in batches:
        bfn_loader.inject_candidate_sequence(batch, component["native_h3"])
        with torch.no_grad():
            native_scores.append(score_value(
                model.score(batch, fixed_t=float(config["bfn"]["fixed_t"])), batch
            ))

    methods = []
    for method, sequences in method_pools.items():
        candidate_rows = []
        for sequence in sequences:
            conformer_scores = []
            for batch in batches:
                bfn_loader.inject_candidate_sequence(batch, sequence)
                with torch.no_grad():
                    conformer_scores.append(score_value(
                        model.score(batch, fixed_t=float(config["bfn"]["fixed_t"])), batch
                    ))
            candidate_rows.append({"sequence": sequence, "conformer_scores": conformer_scores})
        methods.append({
            "method": method,
            "candidates": candidate_rows,
        })
    return {
        "component_id": component["component_id"],
        "native_h3": component["native_h3"],
        "poses": canonical_poses,
        "native_scores": native_scores,
        "methods": methods,
    }


def component_cache_valid(payload, config_hash, generation_hash, component, pools):
    if payload.get("config_sha256") != config_hash:
        return False
    if payload.get("raw_generation_sha256") != generation_hash:
        return False
    if payload.get("component", {}).get("component_id") != component["component_id"]:
        return False
    expected = {method: sequences for method, sequences in pools.items()}
    actual = {
        row["method"]: [candidate["sequence"] for candidate in row["candidates"]]
        for row in payload.get("scores", {}).get("methods", [])
    }
    return actual == expected


def summarize(rows, methods, component_ids, statistics, gates):
    output = {}
    for index, method in enumerate(methods):
        values = []
        component_rows = []
        for component_id in component_ids:
            selected = [
                row for row in rows
                if row["component_id"] == component_id and row["method"] == method
            ]
            value = float(np.mean([row["ensemble_minus_single_state"] for row in selected]))
            values.append(value)
            component_rows.append({
                "component_id": component_id,
                "mean_ensemble_minus_single_state": value,
                "mean_ensemble_minus_native": float(np.mean([
                    row["ensemble_minus_native"] for row in selected
                ])),
            })
        ci = bootstrap_mean(
            values,
            int(statistics["bootstrap_trials"]),
            int(statistics["bootstrap_seed"]) + index,
        )
        positive_fraction = sum(value > 0 for value in values) / len(values)
        gate_results = {
            "minimum_valid_components": len(values) >= int(gates["minimum_valid_components_per_generator"]),
            "minimum_positive_component_fraction": positive_fraction
            >= float(gates["minimum_positive_component_fraction"]),
            "bootstrap_ci95_lower_above_zero": bool(ci and ci[0] > 0),
        }
        output[method] = {
            "valid_components": len(values),
            "mean_ensemble_minus_single_state": float(np.mean(values)),
            "median_ensemble_minus_single_state": float(np.median(values)),
            "positive_component_fraction": positive_fraction,
            "component_bootstrap_ci95": ci,
            "exact_sign_flip_p": exact_sign_flip_p(values),
            "gate_results": gate_results,
            "passed": all(gate_results.values()),
            "components": component_rows,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_bfn_fixed_scoring_dev_v1.yml"
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
    checkpoint = ROOT / config["bfn"]["checkpoint"]
    if sha256(checkpoint) != config["bfn"]["checkpoint_sha256"]:
        raise ValueError("BFN checkpoint hash mismatch")

    manifest = json.loads(paths["component_manifest"].read_text(encoding="utf-8"))
    raw = json.loads(paths["raw_generation"].read_text(encoding="utf-8"))
    if raw["status"] != "complete" or raw["failures"]:
        raise ValueError("Matched generation must be complete")
    components = manifest["components"]
    methods = config["candidate_pool"]["methods"]
    pools = candidate_pools(raw, methods)
    poses = pose_rows_by_component(
        components,
        json.loads(paths["abeta_pose_audit"].read_text(encoding="utf-8")),
        json.loads(paths["nonabeta_pose_audit"].read_text(encoding="utf-8")),
    )

    os.environ["DISORDERFLOW_CHECKPOINT"] = str(checkpoint.resolve())
    model, _ = bfn_loader.load_bfn(config["bfn"]["device"])
    out_path = ROOT / (args.out or config["output"])
    work_dir = out_path.parent / "work"
    config_hash = sha256(config_path)
    generation_hash = sha256(paths["raw_generation"])
    component_scores = []
    for component in components:
        component_id = component["component_id"]
        component_pools = {
            method: pools[(component_id, method)] for method in methods
        }
        cache_path = work_dir / component_id / "fixed_scores.json"
        cached = None
        if cache_path.exists():
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if component_cache_valid(
                candidate, config_hash, generation_hash, component, component_pools
            ):
                cached = candidate["scores"]
        if cached is None:
            scores = score_component(
                config,
                component,
                poses[component_id],
                component_pools,
                model,
                work_dir / component_id / "poses",
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
        print(f"BFN fixed scoring complete: {component_id}", flush=True)

    loo_rows = []
    aggregation = config["primary"]["ensemble_aggregation"]
    for component in component_scores:
        native = np.asarray([row["iptm"] for row in component["native_scores"]])
        for method_row in component["methods"]:
            sequences = [row["sequence"] for row in method_row["candidates"]]
            matrix = np.asarray([
                [score["iptm"] for score in row["conformer_scores"]]
                for row in method_row["candidates"]
            ])
            loo_rows.extend(leave_one_out(
                component["component_id"],
                method_row["method"],
                sequences,
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
